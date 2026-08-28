# Phase 2 — convolution: 8 operators, one closed form covers all of them, 48/48 sandwich. Conv is the best-behaved family in the corpus and needs no new exception category.

**Measured 2026-08-27 on a Colab T4** (`torch 2.11.0+cu128`, `triton 3.6.0`,
Tesla T4 cap 7.5 — same environment as the 2026-08-25 and Phase-1 rounds, so
every number is directly comparable). Session `kccphase2`, provisioned and
stopped. Kernels `kernels/conv_kernels.py`; probes `probes/`; raw output in
`native_run/`. **Nothing in the checker's verdict path was changed.**

---

## Verdict up front

| | result |
|---|---|
| Operators added | **8** (56 → **64**); covers **35 of 100** KernelBench L1 problems |
| Triton conv kernels written and verified vs torch | **36 / 36 configurations**, worst relative error **2.528e-07** (Phase 1's was 2.9e-07) |
| Closed forms | **ONE identity covers all 8 forms.** Autograd-verified on 19 configurations, max rel err **3.8e-16** |
| Sandwich, fully native | **48 / 48** invocations, both sides, 0 errors |
| Adversarial regime | **35 / 35** both sides — **0 violations, 0 vacuous variants**. The only family in the corpus with a clean sweep |
| Closed-form `L` vs converged probe (K=20000) | **1.011 – 1.020, median 1.014** |
| **M3 R², full 62-operator corpus** | **0.8564** (was 0.8567 on 54) — conv is **neutral**, and it is the *second-tightest* family |
| Mutant coverage | **16 / 16 caught, 0 false positives** — after one spec defect was found and fixed |
| Exception taxonomy | **No new category, and conv fits none of the three existing ones** — §6 |
| Determinism | `det_floor = 0` on all 48. Gather formulation, no atomics |

---

## 1. The closed form — one identity, eight forms

Derived rather than assumed, per the task. A convolution is linear in `x`, and
for a fixed output element `o` the map tap → input position is **injective**
(two distinct taps of the same output read two distinct inputs), so

```
    y_o = Σ_τ W[τ] · x[φ(o,τ)]     ⟹     ‖J_o‖₂² = Σ over IN-BOUNDS taps of W[τ]²
```

That right-hand side is itself a convolution: feed an **all-ones input** through
the **same operator** with `W²` in place of `W`. In-bounds taps each contribute
`W[τ]²·1`; out-of-bounds taps contribute 0, exactly as zero-padding does. Hence
for every variant, with identical stride/padding/dilation/groups:

```
        ‖J_o‖₂  =  √( F(ones_like(x), W², same hyperparameters)[o] )
```

For the **transposed** forms the same argument applies with `F = conv_transpose`:
`y_o = Σ_{(i,k): i·s − p + k·d = o} W[k]·x[i]`, and for fixed `(i,o)` the tap `k`
is uniquely determined, so the summation index is again injective.

**Why this is the whole derivation.** Stride, padding, dilation, groups and
asymmetric kernels need no special cases — they are *already encoded in `F`*.
The plan budgeted "~4 derivations … composable, but each needs care"; the
measured answer is **one**.

Verified against an autograd-exact Jacobian on 19 configurations spanning every
combination (`probes/derive_conv.py`): **max relative error 3.8e-16, 19/19**.

Two properties, both checked rather than asserted:

- **Input-independent** — recomputing on a completely different `x` (scaled 37×,
  shifted) is bitwise identical. Conv joins matmul/batchnorm in that class, as
  the plan predicted.
- **NOT shape-only** — it needs `W`, and padding makes the profile genuinely
  non-constant across `o` because border outputs tap fewer weights. Even
  unpadded, `‖J_o‖` varies by output channel. So conv does **not** join
  `STATIC_OPS`.

---

## 2. The kernels — the phase's real cost, as predicted

`TritonBench/reference/` had no convolutions. Eight `@triton.jit` kernels were
written (`kernels/conv_kernels.py`, 323 lines): conv1d/2d/3d, conv_transpose1d/2d/3d,
depthwise_conv2d, pointwise_conv2d. Direct convolution, deliberately — an
im2col+`tl.dot` formulation would be faster and would also insert a second,
separately-fallible transformation between the operator and its measured Jacobian.

Taps are looped at **runtime**, so one kernel body per form covers every
(stride, padding, dilation, groups, asymmetric-kernel) combination. That is what
reduces 35 KernelBench problems to 8 kernels.

**Transposed forms are implemented as gathers, not scatters.** For output `o` and
tap `k` the contributing input is `i = (o + p − k·d)/s`, used only when that
division is exact and `i` is in range. No atomics, no write conflicts — hence
`det_floor = 0` on every invocation, avoiding `frobenius_norm`'s known
non-bitwise-determinism (GPU_NATIVE.md §3a).

**Verification gate, before any measurement:** 36 configurations covering the
full hyperparameter matrix — **36/36 correct, worst relative error 2.528e-07**.

---

## 3. The table — fully GPU-native

Each operator sweeps all 5 of its `valid_shapes` configs (distinct stride /
padding / dilation / groups / 1×1 regimes), 6 invocations each.

| operator | n | m | tol | L | tol/3σL | defect | slope | sandwich | L_mc(20k)/L_cl | M3 | spread | zero_frac |
|---|--:|--:|--:|--:|--:|--:|--:|:-:|--:|--:|--:|--:|
| `conv1d` | 6 | 346 | 3.924e-02 | 4.304e+00 | 3.091 | 0.0115% | 1.0000 | **6/6** | 1.015 | 1.003 | 1.37 | 0.000 |
| `conv2d` | 6 | 1316 | 8.272e-02 | 7.586e+00 | 3.310 | 0.0205% | 1.0000 | **6/6** | 1.019 | 1.000 | 1.33 | 0.000 |
| `conv3d` | 6 | 664 | 7.824e-02 | 8.696e+00 | 3.164 | 0.0296% | 1.0000 | **6/6** | 1.013 | 1.013 | 1.40 | 0.000 |
| `conv_transpose1d` | 6 | 372 | 3.939e-02 | 4.451e+00 | 2.980 | 0.0135% | 1.0000 | **6/6** | 1.011 | 0.984 | 1.65 | 0.000 |
| `conv_transpose2d` | 6 | 2880 | 6.730e-02 | 7.138e+00 | 3.246 | 0.0235% | 1.0000 | **6/6** | 1.012 | 1.014 | 1.42 | 0.000 |
| `conv_transpose3d` | 6 | 720 | 5.985e-02 | 7.452e+00 | 3.080 | 0.0290% | 1.0000 | **6/6** | 1.012 | 1.023 | 1.43 | 0.000 |
| `depthwise_conv2d` | 6 | 2036 | 4.582e-02 | 4.700e+00 | 3.173 | 0.0057% | 1.0000 | **6/6** | 1.017 | 1.014 | 1.56 | 0.000 |
| `pointwise_conv2d` | 6 | 2686 | 3.896e-02 | 3.625e+00 | 3.475 | 0.0080% | 1.0000 | **6/6** | 1.020 | 1.033 | 1.65 | 0.000 |

**TOTAL: 48 / 48 invocations satisfy both sides, across 8 operators, 0 errors.**

**Conv is the most linear family in the corpus by a wide margin.** Linearisation
defect across all 48: **min 0.0037%, median 0.0146%, max 0.0805%** — against
Phase 1's median 0.039% and max 100%. That is exactly what should happen: conv
*is* exactly linear, so the only defect is float rounding. Every slope is
1.0000. This is a positive control on the whole directional-derivative method —
an operator with no curvature measures as having no curvature.

---

## 4. `L` is structural — the same convergence signature, a third time

| `L_mc(K) / L_closed` | min | median | max | within ±5% |
|---|---:|---:|---:|---:|
| K = 400 *(what the main run used)* | 1.062 | **1.100** | 1.137 | 0 / 8 |
| K = 4000 | 1.021 | 1.034 | 1.044 | **8 / 8** |
| K = 20000 | 1.011 | **1.014** | 1.020 | **8 / 8** |

Prior rounds: original 27 gave 1.081 → 1.025 → 1.010; Phase 1 gave
1.107 → 1.027 → 1.012. **Three independent operator sets, three independent
derivations, the same convergence.** The K=400 gap is estimator bias; the closed
forms are right.

---

## 5. M3 with convolution included — conv is neutral, and unusually tight

| corpus | n | ops | **R²** | median pred/meas | spread | ±10% |
|---|--:|--:|--:|--:|--:|--:|
| original 27 | 38 | 27 | 0.9579 | 1.022 | 1.26× | 36/38 |
| Phase-1 27 | 162 | 27 | 0.8388 | 1.014 | 1.86× | 124/162 |
| **Phase-2 conv 8** | 48 | 8 | 0.8398 | **1.009** | **1.36×** | **46/48** |
| 54-operator corpus (pre-conv) | 200 | 54 | 0.8567 | 1.016 | 1.86× | 160/200 |
| **FULL 62-operator corpus** | **248** | **62** | **0.8564** | 1.015 | 1.86× | 206/248 |

**R² moves 0.8567 → 0.8564. Conv is neutral.** It neither rescues the fit nor
degrades it.

### Does conv's linear, input-independent Jacobian help the fit?

By the statistic that actually measures per-family agreement, **yes — it is the
second-tightest family in the corpus**:

| family | n | median pred/meas | spread |
|---|--:|--:|--:|
| matmul-variant | 24 | 1.0059 (+0.6%) | **1.13×** |
| activation | 54 | 1.0013 (+0.1%) | 1.19× |
| **conv (Phase 2)** | **48** | **1.0089 (+0.9%)** | **1.36×** |
| loss (`m=1`) | 31 | 0.9891 (−1.1%) | 1.83× |
| **scan** | 24 | **1.2467 (+24.7%)** | 1.25× |

46 of 48 conv invocations land within ±10% — a 96% hit rate, the best of any
family. Per operator the residuals span just **−1.6% to +3.3%**
(`conv2d` −0.01%, `conv1d` +0.28%, `conv3d` +1.28%, `conv_transpose2d` +1.37%,
`depthwise` +1.42%, `conv_transpose1d` −1.56%, `conv_transpose3d` +2.28%,
`pointwise` +3.29%).

**A caveat on conv's standalone R² of 0.8398, because reading it as "conv fits
badly" would be wrong.** R² is scaled by the variance of the *actual* values,
and conv's `y` range is narrow — 2.53 to 3.94, against 1.4–4.8 for the corpus as
a whole. A family whose true values barely vary cannot score a high R² however
small its residuals are. Its residuals are in fact the second-smallest in the
corpus. **For within-family agreement, cite the ±10% rate and the spread; R² is
the wrong statistic at fixed operator family.**

The mechanistic reading is consistent with everything prior rounds established:
M3 assumes orthogonal Jacobian rows, and it over-predicts in proportion to how
correlated the real rows are. Conv rows overlap only where receptive fields
overlap — bounded by the kernel size, and zero for stride ≥ kernel — so the
correlation is mild and local. That places conv between the near-orthogonal
families (matmul, elementwise) and the maximally-correlated one (prefix scans,
whose rows are nested subsets and which over-predict by 24.7%).

---

## 6. Exception taxonomy — no new category, and conv fits none of the three

Tested explicitly rather than assumed, because the task asked whether conv's
local receptive fields and potentially sparse Jacobians create a new failure mode.

| existing category | diagnostic | conv's measurement | fits? |
|---|---|---|:-:|
| absolute `1e-6` floor | is `tol` clamped? | min raw `3σ·q95` = **6.99e-03 = 6985× the floor** | **no** |
| fp32 quantisation floor | `s`/ulp ≈ 2–3, defect exactly 900% | min s/ulp over all 48 = **3054**; defect max **0.0805%** | **no** |
| `m=1` diagnostic blindness | scalar output | `m` ranges **18 … 30720**; **zero** invocations at `m=1` | **no** |

> **CORRECTED 2026-08-27:** `../theory_audit_2026-08-27/` showed the "three
> existing categories" are not disjoint regions — the two floor rows are the
> absolute and relative arms of one resolvability criterion (and overlap on
> the real data), and m=1 invocations can trip the relative arm too. **This
> section's conclusion is unaffected**: conv clears the *unified* criterion by
> the same margins shown here (min s/ulp 3054 vs the 32 screen, min raw tol
> 6985× the 1e-6 floor, m ≥ 18), so "conv fits none of the exception
> mechanisms" stands as measured.

And no *new* category is needed: **35/35 adversarial variants satisfy both
sides, with 0 violations and 0 vacuous (`L ≤ 0`) cases.** For comparison Phase 1
had 11 vacuous variants and 4 upper-bound deviations. **Conv is the only family
in the corpus that sweeps clean.**

### The sparsity question, answered — and it is real, just benign

Conv's Jacobian *matrix* is extremely sparse (each row has at most
`C_in·∏k` nonzeros out of `numel(x)`). That is not what the profile sees: the
profile is the vector of row *norms*, and a row norm is zero only when an output
element receives **no** taps at all. On 45 of 48 invocations that never happens
— `zero_frac = 0.000` exactly.

It does happen on 3, and the pattern is structural:

| operator | config | `zero_frac` | predicted `1 − s^(−nd)` | `y` | sandwich |
|---|--:|--:|--:|--:|:-:|
| `conv_transpose1d` | 2 | 0.4889 | 0.5000 | 3.074 | **OK** |
| `conv_transpose2d` | 2 | 0.7495 | 0.7500 | 3.219 | **OK** |
| `conv_transpose3d` | 2 | 0.8846 | 0.8750 | 3.294 | **OK** |

These are the strided (`s=2`) transposed configs. A strided transposed
convolution scatters into an output grid where only every `s`-th position per
axis receives any contribution, so a fraction `1 − s^(−nd)` of outputs is
**structurally dead** — the classic checkerboard artifact. The measured
fractions match that prediction to within 1.1% (the residual is border effects),
and **the closed form reproduces the dead positions exactly**, since
`F(ones, W²)` is zero precisely there.

**All three still satisfy both sides**, with `y` in 3.074–3.294 — squarely inside
the dense-profile range of 2.528–3.944. So sparsity moves nothing.

**Why it is benign here when `hardsigmoid`'s zeros were flagged.** The
distinction is *structural* versus *input-dependent*:

- `hardsigmoid` (Phase 1, 96.7% zeros): the dead rows are wherever `|x| ≥ 3`, so
  they **move as the input moves**. The profile is unstable, and M3 simulating
  over an all-but-empty, shifting profile is not a validated regime.
- conv: the dead outputs are fixed by `(stride, dilation, padding, shape)` and
  are **identical for every input**. A structurally dead output carries no
  information for *any* kernel — correct or buggy — so no verdict can depend on
  it. `y_profile` filters `rn[rn > 0]` before simulating, which is exactly the
  right behaviour here.

**Recommended classification: conv requires no caveat.** It is the cleanest
family in the corpus and belongs in the "passes" column without qualification.

---

## 7. Mutant coverage, and the spec defect found by measuring it

**16 / 16 caught, 0 false positives on the correct implementation.**

Two mutants per operator, each a documented real conv bug — `flipped_kernel`
(correlation vs. true convolution, the single most common conv error and
invisible for symmetric kernels), `ignores_dilation`, `wrong_padding`
(divisibility off-by-one in the transposed gather), `not_grouped` (channel leak
in depthwise), `partial_channels`, `transposed_weight`. Torch-level, same
rationale as Phase 1; Layer-1 AST checks deliberately not run.

### D4 — four operators never exercised dilation

`conv3d/ignores_dilation` escaped the entire battery on the first run. Cause:
**not one of `conv3d`'s five configs used `dilation > 1`**, so a kernel that
silently ignores the dilation argument produced bit-identical output on every
input the spec could generate — undetectable by construction.

Auditing the rest found the same hole in `conv_transpose2d`,
`conv_transpose3d` and `depthwise_conv2d`. Only `conv1d`, `conv2d` and
`conv_transpose1d` had a dilated config. `pointwise_conv2d` is legitimately
exempt: its kernel is 1×1, so dilation is a no-op by definition.

**This is the same class as Phase 1's D1** — a hyperparameter present in the
signature but never varied — and it was found the same way, by a mutant
escaping. Fixed by setting `dilation = 2` on config index 2 of the four affected
operators; catch rate went 15/16 → **16/16**.

The general lesson now has two instances and is worth stating as a rule: **a
spec that accepts a hyperparameter must vary it in `valid_shapes`, or a kernel
that ignores that hyperparameter is undetectable.** A cheap audit is one line
per spec — enumerate the config column and assert at least two distinct values.

---

## 8. Corpus state after Phase 2

| | count |
|---|--:|
| operator specs | **64** |
| with closed-form `L` (`SUPPORTED_OPS`) | **62** |
| shape-only (`STATIC_OPS`) | 14 |
| excluded (`argmax`, `argmin` — `int64`, `J = 0` a.e.) | 2 |
| KernelBench L1 problems covered by an operator | **80 / 100** |

The 20 L1 problems still uncovered are the remaining matmul layout variants
(6–11, 13, 16–18), `cumprod` (excluded by derivation — unbounded condition
number), `softsign`/`hardtanh`/`hinge`/`triplet` (dropped in Phase 1 as absent
from both real corpora), and `conv_depthwise_separable_2D` (#86), which is a
*composition* of two operators already covered and belongs to the L2 fusion
question, not to L1.

---

## 9. Reproduce

```bash
python probes/derive_conv.py           # 19 configs vs autograd, CPU, no GPU

export HOME=~/.colab-home
colab new --gpu T4 -s <name>
colab upload -s <name> kernels/conv_kernels.py /content/conv_kernels.py
colab upload -s <name> kcc.tgz /content/kcc.tgz       # verification/ only
colab exec  -s <name> -f probes/verify_conv.py --timeout 900   # gate: 36/36
colab exec  -s <name> -f probes/conv_native.py --timeout 900   # 48 invocations, ~11s
colab exec  -s <name> -f probes/conv_pass2.py  --timeout 900   # K-ladder + adversarial, ~34s
colab stop  -s <name>

python probes/run_conv_mutants.py      # catch/FP, CPU
```

Total GPU compute for the round: **under 1 minute.** As in Phase 1, the cost is
writing the kernels, not measuring them.

---

## 10. What this changes elsewhere

- `structural_l.py`: `SUPPORTED_OPS` 54 → **62**. `STATIC_OPS` unchanged at 14 —
  conv is input-independent but needs `W`, so it is not shape-only.
- `base_spec.py`: adds `ConvKernelSpec`, whose `valid_shapes` entries are
  **structured configs** `(N, C_in, C_out, spatial, kernel, stride, padding,
  dilation, groups)` rather than flat tuples. `CORPUS_EXPANSION_PLAN.md` §3.2
  change 6 (refactor `valid_shapes` to a config dataclass) is **still open** and
  was deliberately not done inside this phase — it touches all 56 pre-existing
  specs and should not ride along with a corpus addition.
- `CORPUS_EXPANSION_PLAN.md` §4.4's estimate of "~4 derivations" for conv is
  superseded: **one identity covers all eight forms.** Its estimate that
  authoring the kernels, not the maths, is the cost **held**.
