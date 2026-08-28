# #5 — Why the gap exists: a numerical account

Companion to `benchmarks/BUG_CLASS_THEORY.md`, which establishes the empirical
result. This derives it. Every threshold below is computed from the operator's
arithmetic and then checked against the recorded search history; where the two
disagree, or where a derivation does not cover a case, it is said so.

**Status: drafted 2026-08-21 under a time box, extended the same evening.
§1-§4 and §6 are complete and their numbers are verified against the recorded
search history. **§5 (`matmul`) is PARTIAL: three of its four masking conditions
hold — one of them confirmed out-of-sample against a control — and the fourth
(`skip_boundary_tiles`) was FALSIFIED by that same test and is open (§5.4).**
§7 is an explicit sketch. §8 lists what is uncovered: the two attention
operators, plus `skip_boundary_tiles`. Not yet a
paper section: no related work, and no treatment of the online-softmax
recurrence.**

---

## 1. Setup

Let `f` be the reference kernel, `f̃` a mutant, `x` an input. The baseline the
checker is measured against is `torch.allclose` with `atol = a = 1e-3`,
`rtol = r = 1e-2` (`executor.py`, `_evaluate_kernel`).

Define the **blindness predicate**: the baseline cannot see the bug on `x` iff

```
    ∀j :  |f̃(x)_j − f(x)_j|  ≤  a + r·|f(x)_j|                          (1)
```

Two properties of (1) drive everything that follows.

**It is elementwise, and its right-hand side is not constant.** The tolerance
available at coordinate `j` is proportional to the *reference's own output* at
`j`. Where `f(x)_j` is large, `r` grants a wide margin; where `f(x)_j → 0`, the
envelope collapses to the constant `a`. A bug is therefore easiest to hide in
coordinates where the reference output is large, and hardest to hide where the
reference output vanishes — regardless of how small the bug's absolute error is
there.

**A HIT requires (1) to hold.** By construction, a HIT is: reference valid,
mutant fails the checker, mutant passes the baseline. `BUG_CLASS_THEORY.md`
measures that the middle term is satisfied on every valid, baseline-blind
proposal in the corpus — **0 counterexamples in 120** — so predicting HITs
reduces to predicting (1). That is what makes a numerical account possible at
all: the question "what input produces a hit" becomes the question "what input
satisfies (1)", which is arithmetic rather than search behaviour.

---

## 2. Tolerance invariance: which gaps survive a perfect baseline

This section is the load-bearing one. The rest of the document derives *when*
particular bugs hide; this establishes that they hide in two categorically
different ways, only one of which is a statement about the checker at all.

### 2.1 The residual, and the claim

For a mutant `f̃`, reference `f`, and input `x`, define the **residual**

```
    R(x)_j  =  f̃(x)_j − f(x)_j
```

**Claim.** The baseline `allclose(f̃(x), f(x); a, r)` is blind on `x` for *every*
tolerance pair `(a, r) ≥ 0` — including `a = r = 0` — **if and only if
`R(x) ≡ 0`.**

**Proof.** (⇐) If `R(x) ≡ 0` then `|R_j| = 0 ≤ a + r|f(x)_j|` for all `j` and all
`(a, r) ≥ 0`, the right-hand side being non-negative. (⇒) Contrapositive:
suppose `R(x)_k ≠ 0` for some `k`. At `a = r = 0` condition (1) requires
`|R_k| ≤ 0`, which fails. So some admissible baseline sees the bug. ∎

The proof is one line. **The content is not the proof — it is that the
partition it induces is non-empty on both sides, and that the inputs landing on
the invariant side are ordinary rather than contrived.**

### 2.2 The partition, and a decision procedure

The claim splits every hit into:

- **Class T (tolerance straddling)** — `R(x) ≠ 0`, but `|R_j| ≤ a + r|f(x)_j|`
  for the specific `(a, r)` in use. The gap is a joint property of the checker
  **and the baseline's tolerance**. Some tighter allclose would close it: any
  `(a', r')` with `a' + r'|f(x)_k| < |R(x)_k|` for a single coordinate `k`.
- **Class E (exact masking)** — `R(x) ≡ 0`. The mutant and the reference are the
  *same function* at `x`. **No allclose-based baseline, at any tolerance, in any
  precision, can detect this mutant on this input** — there is no output
  difference to detect.

This is decidable offline and cheaply: re-run the simulation with `a = r = 0`
and record which hits survive. That is a decision procedure, not an
interpretation, and it is what produced the numbers below.

### 2.3 Measured split

Over the 20 simulated confirmed hits:

| class | count | survives `a = r = 0` | example |
|---|---:|---|---|
| **E — exact masking** | **9** | yes | `rmsnorm:ignore_gamma` at `γ ≡ 1` |
| **T — tolerance straddling** | **11** | no | `gelu:sigmoid_approx` at `x ≈ 1` |

**Precision caveat, because the boundary is not tolerance-dependent but
precision-dependent.** The simulation runs float64; the kernels run float32. The
absorption threshold of §4.2 is `v > 40.9` in float64 and `v > 21.5` in float32,
and two recorded proposals sit at `v = 20` — classified T here, but they would
be **class E on the real float32 kernels**. So the on-hardware split is likely
**11 / 9**, not 9 / 11. The qualitative claim is unaffected; the counts are not,
and should not be quoted without this sentence.

### 2.4 Why class E is not a curiosity

The obvious objection to class E is that `R(x) ≡ 0` sounds like a measure-zero
contrivance the search had to hunt for. It is the opposite. All three mechanisms
that produce it (derived in §4) are triggered by the **ordinary** operating
regime, not an adversarial corner:

| mechanism | the masking input | how exotic is it? |
|---|---|---|
| algebraic identity | `γ ≡ 1`, `β ≡ 0` | the **defaults** — what an untrained layer holds, and what anyone writes in a smoke test |
| discrete uniqueness | no tied maxima | the **generic** case; ties have measure zero under any continuous distribution |
| fp absorption | saturated softmax | routine whenever logits have a wide spread |

So the class-E bugs are precisely the ones that survive casual output
comparison **in normal use**, which is where a correctness checker earns its
keep. `argmax:tiebreak` is the sharpest illustration and is worth stating in the
paper as a standalone anecdote: 20 of 21 valid proposals contained deliberate
ties — the intuitive way to expose a tiebreak bug — and every one **failed** to
produce a gap, because a tie makes the two kernels visibly disagree and the
baseline catches it too. The single hit came from an input with **no ties at
all**.

### 2.5 Consequence for the project's claims

Two claims of very different strength are being conflated whenever "N bugs
allclose misses" is quoted as a single number:

1. **Class T (11/20, or 9/20 on hardware) is contingent on `rtol = 1e-2`.**
   Anyone may respond "then tighten your baseline", and they would be right:
   these gaps close. Any headline citing them must state the baseline tolerance
   it was measured against, because the number moves with it.
2. **Class E (9/20, or 11/20 on hardware) is unconditional.** It is not a
   statement about a tolerance being loose; it is a statement that output
   comparison is the wrong instrument. This is the claim that survives the
   tightening objection, and it is the one that argues for property checking as
   a *category* rather than as a better-tuned comparator.

**The honest headline is therefore not one number but two**, and the second is
the interesting one. A checker that only delivered class T would be a tolerance
argument; a checker that delivers class E is doing something output comparison
provably cannot.

## 3. Tolerance straddling, derived

### 3.1 `softmax:wrong_reduction` — a normalisation-constant perturbation

Reference: `y_j = e_j / S`, with `e_j = exp(x_j − max x)` and `S = Σ_k e_k`.
The mutant divides by a truncated sum `S_P = Σ_{k<P} e_k`, `P = 64`.

So `ỹ = κ·y` with `κ = S/S_P ≥ 1`: the mutant's output is the reference scaled
by a single scalar. Substituting into (1):

```
    y_j(κ−1) ≤ a + r·y_j   ∀j     ⟺     κ − 1  ≤  r + a / max_j y_j        (2)
```

The binding coordinate is the **largest** output, because that is where the
constant `a` buys the least relative slack.

For the corpus's hit shape — a spike of `m = 32` equal maxima at value `v`
against a zero background, `n` columns — we have `max y = 1/(32 + (n−32)e^{−v})
≈ 1/32`, so `a / max y ≈ 0.032`, and

```
    κ − 1  ≈  (n − 64)·e^{−v} / 32   ≤   r + 0.032 = 0.042
```

giving a closed-form condition on the spike height:

```
    e^{−v} ≤ 1.344 / (n − 64)                                             (3)
```

For `n = 2048`: **`v ≥ 7.30`**. Every recorded softmax hit used `v ∈ {10, 20,
100}` — all satisfy (3). ✔

**The same formula explains the non-hits, and this is where it earns its keep.**
Move the spike outside the first `P = 64` columns and `S_P` contains only
background: `S_P = 64e^{−v}`, `S ≈ 32`, so `κ = e^{v}/2`. At `v = 10` that is
`κ ≈ 1.1 × 10⁴`, violating (2) by four orders of magnitude. Hand-checked against
the two real proposals:

| proposal | spike at | value | error | tolerance | baseline |
|---|---|---:|---:|---:|---|
| `435708ba` (HIT) | `[:, :32]` | 10 | 8.77e-05 | 1.31e-03 | blind |
| `f9f19cce` (miss) | `[:, -32:]` | 10 | 3.44e+02 | 1.31e-03 | sees it |

**Same bug, same magnitude, same shape — position alone moves the error by a
factor of ~4 × 10⁶.** The relevant quantity is not "is the input extreme" but
"does the truncated reduction still contain the probability mass".

### 3.2 `softmax:first_tile` — a support truncation

The mutant computes softmax over the first `c = n/2` columns and writes zeros
beyond. For `j ≥ c`, `Δ_j = −y_j`, so (1) becomes

```
    y_j ≤ a + r·y_j    ⟺    y_j ≤ a/(1−r) ≈ 1.01e-3      ∀ j ≥ c          (4)
```

**The tail mass must sit below `atol`.** With the spike in the first 32 and a
zero background, `y_j ≈ e^{−v}/S ≈ 1.4e-6` for `v = 10` — blind by three orders
of magnitude. With the spike in the last 32, `y_j ≈ 1/32 = 3.1e-2` — visible by
a factor of 31. Same input geometry, opposite verdict, for a different reason
than §3.1: here it is an absolute-tolerance argument, there a relative one.

### 3.3 `gelu:sigmoid_approx` — an approximation-error envelope

Reference `g(x) = x·Φ(x)`; mutant `g̃(x) = x·σ(1.702x)`. The constant 1.702 is
the classical choice minimising `sup_x |σ(1.702x) − Φ(x)|`; computed here that
supremum is **0.00949, attained near |x| = 2**. The pointwise error is
`Δ(x) = x·[σ(1.702x) − Φ(x)]`, so (1) reads

```
    |x|·|σ(1.702x) − Φ(x)|  ≤  a + r·|x·Φ(x)|                             (5)
```

Solving (5) numerically over `x ∈ [−10, 10]`:

```
  blind:   x ∈ [0.70, 10]   and   x ∈ [−0.30, 0.44]   and   x ≤ −5
  visible: x ∈ [−5, −0.30]  and   x ∈ [0.44, 0.70]
```

The structure is the collapsing envelope from §1. For `x ≳ 0.7`, `Φ(x)` is
`O(1)` so the `r·|g(x)|` term dominates and easily covers an error of ~1% of
`x`. For `x ∈ [−5, −0.3]`, `Φ(x) → 0` faster than the approximation error does:
at `x = −4`, `g(x) = −1.3e-4` while `|Δ| = 4.3e-3` — **the mutant's error is 33×
the reference's own output**, and the envelope has shrunk to `a`.

Against the corpus: all four gelu hits used `fill=ones` at scale 1 or 2, or a
patch at 1.5 — every sampled value inside the blind band. Every gelu non-hit
sampled the visible band: `randn × 3` and `randn × 5` span it, the explicit
patches sit at `−4`, `−8`, `±5`, `±10`, and `arange × 0.01` sweeps through
`[0.44, 0.70]` on its way up. **4 hits and 14 non-hits, all accounted for by
(5).** ✔

### 3.4 `instancenorm:skip_eps` — a conditioning argument

The mutant computes `rsqrt(σ²)` where the reference computes `rsqrt(σ² + ε)`,
`ε = 1e-5`. The relative output error is

```
    (1 + ε/σ²)^{1/2} − 1  ≈  ε / (2σ²)      for σ² ≫ ε                    (6)
```

so the bug's visibility is governed by `ε/σ²` — a condition number in the
ordinary sense: the sensitivity of the normalisation to a perturbation of the
variance. The three recorded hits all used `fill=randn`, giving `σ² ≈ 1` and a
relative error of `≈ 5e-6`, three orders of magnitude inside `r = 1e-2`. Blind.

This is the one class where the checker's advantage is *not* about the input the
search chose: the spec's own battery supplies `near_zero_variance`
(`torch.full_like(x, 3.0) + x*1e-6`), driving `σ² → 0` and `ε/2σ² → ∞`. The
search's contribution is only to hand the baseline an input where `σ² ≈ 1`.

---

## 4. Exact masking, derived

### 4.1 Algebraic identity

`rmsnorm:ignore_gamma` computes `x/rms` where the reference computes
`(x/rms)⊙γ`. Then `Δ = (x/rms)⊙(γ−1)`, and

```
    γ ≡ 1   ⟹   Δ ≡ 0   for every x, exactly.
```

Likewise `layernorm:ignore_gamma_beta` under `γ ≡ 1, β ≡ 0`. These are not
small errors; the mutant and the reference are the *same function* restricted
to that parameter slice. The bug lives in a direction of parameter space the
input never excites.

The checker catches it because `check_gamma_correctness` does not use the
proposed `γ` at all: it supplies its own `γ = 1` and `γ = 2` and tests the
homogeneity identity `f(x, 2γ) = 2f(x, γ)`. That is a *property* of the operator,
not a comparison of two outputs on one input, which is exactly why it survives
where output comparison cannot.

### 4.2 Floating-point absorption — the mechanism I did not predict

Five of the nine exact-masking hits are `softmax:wrong_reduction` at spike value
`v = 100`, which §3.1 classifies as merely straddling. They are exact, and the
reason is arithmetic rather than algebra.

The truncated sum `S_P` omits `(n − P)` terms each of size `e^{−v}`. In floating
point, that omission is **exactly lossless** when the omitted total falls below
half an ULP of the retained partial sum:

```
    (n − P)·e^{−v}  <  ½·ulp(m)  =  m·2^{−(t+1)}                          (7)
```

with `m = 32` the retained mass and `t` the mantissa width. For `n = 2048`:

| precision | `t` | threshold on `v` |
|---|---:|---:|
| float64 (this simulation) | 52 | `v > 40.9` |
| **float32 (the actual kernels)** | 23 | **`v > 21.5`** |

At `v = 100` the omitted terms are `~10⁻⁴¹`, absorbed with vast margin; at
`v = 10` they are `~10⁻¹`, plainly retained. The recorded hits split at exactly
this boundary — `v = 100` exact, `v = 10` and `v = 20` straddling. ✔

**Caveat, stated because the boundary is precision-dependent:** the simulation
is float64 and the kernels are float32, and the two thresholds (40.9 vs 21.5)
straddle the recorded `v = 20` cases. Those two proposals are classified as
straddling here but would be **exact** on the real float32 kernels. The 9/11
split in §2 is therefore float64-specific; on hardware it is likely 11/9. This
does not affect the qualitative claim, and it does affect the counts.

### 4.3 The discrete case — `argmax:tiebreak`

The reference returns `min{j : x_j = max x}`, the mutant `max{...}`. So
`Δ = 0` iff **the row maximum is unique**, and the masking condition is
combinatorial rather than metric.

This inverts the intuition the word "adversarial" carries, and the corpus shows
it plainly: **20 of 21 valid argmax proposals contained deliberate ties** — the
obvious way to expose a tiebreak bug — and every one failed to produce a hit,
because a tie makes reference and mutant return visibly different indices and
the baseline catches it too. The single hit came from `arange`, an input with
**no ties anywhere**, where the mutation cannot change the answer.

A second-order effect is worth recording because it will recur wherever an
operator returns indices: applying (1) to integer outputs makes the tolerance
`a + r·|j|` grow **with the index**. Two adjacent tied positions differ by 1,
which is inside `a + r·j` as soon as `j ≥ 100`. So on a wide row, near-adjacent
ties at high indices are masked *by the relative tolerance itself*. The
`argmax` spec's own docstring records being bitten by this: a 2-tie case gave
"reference index 2 vs mutant index 11, diff=9", absorbed as within-tolerance.
**`rtol` is meaningless on a coordinate whose units are positions, and applying
it there silently rescales the test with the array size.**

---

## 5. `matmul`, derived — and one derivation falsified by its own test

Added after §1-§4. The tiling makes this more tractable than attention, and it
turns out to be the sharpest example of class E in the corpus.

The reference tiles `C = A@B` with `BLOCK_M = BLOCK_N = BLOCK_K = 32`,
accumulating in fp32 and masking both loads and the final store. Four mutants
perturb it. Each has a **structural** masking condition — a property of the
shapes and strides, not of the values:

| mutant | what it changes | `R(x) ≡ 0` exactly when |
|---|---|---|
| `skip_boundary_tiles` | drops `C_mask` on the final `tl.store` | ~~`M ≡ 0` and `N ≡ 0 (mod 32)`~~ — **FALSIFIED, see §5.4. The real condition is not yet known.** |
| `swapped_strides` | indexes `A` with `B`'s strides and vice versa | `A.stride() == B.stride()` — true for any two contiguous arrays of the **same shape** |
| `wrong_dtype` | accumulates in fp16 instead of fp32 | every partial sum is exactly representable in fp16 (integers up to 2048) |
| `partial_k_reduct` | contracts only `k < K/2` | `\|Σ_{k≥K/2} A_ik B_kj\| ≤ a + r\|C_ij\|` — the tail of the contraction must be negligible |

Only the last is metric; the first three are exact-masking conditions of the
kind §2 calls class E.

### 5.1 Checked against both recorded matmul proposals

The corpus contains exactly two reference-valid matmul proposals, and between
them they exercise all four conditions in both directions:

| mutant | `0fa23d50`: A `[64,256]`, B `[256,64]`, `ones × 100` | `0b6cc4c1`: A `[256,256]`, B `[256,256]`, `ones` |
|---|---|---|
| `skip_boundary_tiles` | 64, 64 both ≡ 0 mod 32 → **masked** | 256, 256 ≡ 0 mod 32 → **masked** |
| `swapped_strides` | strides `(256,1)` vs `(64,1)` differ → **visible** | both `(256,1)`, identical → **masked** |
| `wrong_dtype` | `Σ = 256 × 100² = 2.56e6` > fp16 max 65504 → **overflows, visible** | `Σ = 256`, exact in fp16 → **masked** |
| `partial_k_reduct` | all-positive, tail is exactly 50% of the sum → **visible** | same → **visible** |
| **predicted credited** | `{skip_boundary}` | `{skip_boundary, swapped_strides, wrong_dtype}` |
| **actually credited** | `['skip_boundary']` ✔ | `['skip_boundary', 'swapped_strides', 'wrong_dtype']` ✔ |

**Eight of eight cells predicted correctly, in both directions — but see §5.4: the `skip_boundary` row is now known to be right for the WRONG REASON, so this is 6 of 8 on conditions that survive testing.** The
`scale = 100` difference between the two proposals is what flips `wrong_dtype`
from masked to visible, and the non-square shape is what flips
`swapped_strides` — so the two proposals are not redundant, they are a
two-point ablation that the derivation reproduces.

### 5.2 The observation worth putting in the paper

**A 256×256 matrix of ones masks three independent bugs simultaneously.** Two
of the reasons are derived and hold up: it is square-and-contiguous (so the two
stride pairs coincide, masking `swapped_strides`) and small-summed (so fp16
accumulation is exact, masking `wrong_dtype`). The third bug,
`skip_boundary_tiles`, is also masked there — but §5.4 shows the reason is NOT
the tile alignment this section originally credited, and the true reason is
unknown. The observation stands; one of its three explanations does not.

What survives intact is that none of these is a coincidence the search
engineered: they are properties of the most obvious test matrix anyone would
write.

This is §2.4's argument in its strongest form. The intuition that a "simple"
input is a weak test is usually about *coverage*; here it is about
**degeneracy** — a simple input collapses distinctions the kernel is supposed
to respect, and every collapsed distinction is a bug that becomes invisible to
output comparison at any tolerance.

### 5.3 A prediction, and its out-of-sample test — **CONFIRMED**

`partial_k_reduct` is the one matmul mutant never masked in the corpus, and the
derivation says why: with an all-positive fill the omitted half contributes ~50%
of every entry, four orders of magnitude outside `r = 1e-2`. But its masking
condition is satisfiable — set `A[:, K/2:] = 0` and the omitted contraction is
**identically zero**, making it class E.

That input had never been proposed. The prediction was written down first and
then tested (`verification_runs/matmul_prediction_2026-08-21/`), with a paired
negative control so that a hit could not be credited to the shape instead of the
zeroing:

| proposal | `A[:, 128:] = 0`? | `partial_k_reduct` |
|---|---|---|
| P1 `[256,256]²`, tile-aligned | yes | **credited** |
| P2 `[100,256]×[256,100]`, unaligned | yes | **credited** |
| P4 `[100,256]×[256,100]`, non-constant `C` | yes | **credited** |
| **P3 CONTROL — identical to P2** | **no** | **not credited** |

**Credited on 3 of 3 zeroed proposals and on 0 of 1 un-zeroed control, first
attempt, exactly as derived.** The control is what makes this evidence: it holds
shape, fill and strides fixed and varies only the zeroing, so the masking is
attributable to the residual being identically zero and not to anything else
about the input.

This is the first claim in this project made **before** the data rather than
fitted to it. `swapped_strides` also behaved as derived — visible on P2/P3/P4,
where `A.stride() ≠ B.stride()`, and masked on P1 where they coincide.

### 5.4 The same test FALSIFIED a different part of this section

`skip_boundary_tiles` came back **masked on all four proposals**, including
`M = N = 100`, where §5's condition (`M ≡ 0 and N ≡ 0 mod 32`) predicts it
should be plainly visible. The condition is wrong.

A second hypothesis — that the all-ones fills make `C` constant, so an
out-of-bounds store writes the same value it clobbers — motivated P4, which
holds everything else fixed and makes `C` vary along `j`. **`skip_boundary`
stayed masked there too.** So that hypothesis is refuted as well, and no third
one is offered here: two mechanisms have been proposed and both are dead, and
inventing a third without testing it is exactly the move this project keeps
having to retract.

**What this costs §5.1.** The "8 of 8 cells" agreement is now known to include
**two cells that were right for the wrong reason** — every recorded matmul
proposal is simultaneously tile-aligned *and* constant-output, so the data could
not distinguish the alignment condition from any other property those inputs
share. This is the same confound as "9 of 9 softmax hits are patched" (74% of
non-hits are too) and "non-power-of-two shapes" (retracted in `BENCHMARK_RESULTS.md` §8.3.1): **a
condition that fits every observation is not thereby the operative one, and the
only way to find out is to construct the input that separates them.** Doing so
here cost one GPU run and overturned a claim that had looked perfectly
supported.

The honest state of §5: `partial_k_reduct` and `swapped_strides` are derived and
now confirmed out-of-sample; `wrong_dtype`'s fp16-exactness condition is
consistent with all six observations but has not been separately falsified;
`skip_boundary_tiles` is **open**.

## 6. What the derivations buy

**The gap is a joint property of the checker and the baseline, not of the
checker alone.** Eleven of twenty hits exist because `rtol = 1e-2`; they would
vanish under a tighter comparison. Any headline of the form "the checker finds N
bugs allclose misses" is quoting a number that moves with the baseline's
tolerance, and the tolerance therefore belongs in the claim.

**Nine of twenty are tolerance-invariant, and those are the load-bearing ones.**
For `γ ≡ 1`, unique maxima, or absorbed reduction terms, the mutant *is* the
reference on that input. No allclose-based test at any tolerance can detect
them. That is the argument for property checking as a category, and it does not
rest on the checker being tuned tighter than someone else's tolerance.

**The search's job is the inverse of what it appears to be.** It is not finding
inputs that expose bugs to the checker — the checker's per-operator batteries
supply their own inputs and catch these mutants on every valid proposal (0
misses in 120). It is finding inputs that *conceal* bugs from the baseline. The
search is adversarial against the comparator, not the kernel.

---

## 7. Sketch — not yet done

- A general statement of when property checking strictly dominates output
  comparison. The three masking mechanisms (algebraic identity, absorption,
  discrete uniqueness) look like instances of "the bug lies in the kernel of the
  evaluation map at `x`", but that is a conjecture here, not a theorem.
- Whether the blind set has positive measure for each mutant class. For gelu it
  clearly does (§3.3 gives intervals); for `ignore_gamma` the blind set is the
  measure-zero slice `γ ≡ 1`, which the search nonetheless finds immediately
  because it is the natural default a person would write.

## 8. Not covered

**`matmul` is PARTIALLY covered — see §5.** `partial_k_reduct` and
`swapped_strides` are derived and confirmed out-of-sample; `wrong_dtype` is
consistent with all six observations but untested against a separating case;
**`skip_boundary_tiles` is OPEN — two proposed conditions, both falsified
(§5.4).** What also remains is
`flash_attention` and `causal_flash_attention` — 132 proposals and 1 of the 23
confirmed hits. Deriving their masking conditions needs the online-softmax
recurrence (the running max/denominator rescaling), which is genuinely harder
than matmul's tiling because the mutants perturb a *sequential* update rather
than a term in a sum: `approx_denom`, `skip_rescaling` and `drop_last_tile`
each change the recurrence's invariant, so the residual does not decompose
into omitted terms the way §5's does. Not attempted here rather than guessed
at. The `causal_flash_attention` NO_HIT result (0 hits in 51
valid proposals) is consistent with §2's framing — a wrong causal mask perturbs
whole output rows, so no input brings it inside the envelope — but that is an
argument, not a derivation.
