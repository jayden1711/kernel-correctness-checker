# The CPU/GPU gap was my measurement error. Full 29-operator coverage, on the corpus's real inputs

**Follow-up to `FINDINGS.md` §8, 2026-08-25.** Probes: `probes/replay.py`,
`probes/coverage2.py`, `probes/gputest.py`, `probes/attn.py`,
`probes/saturation.py`. **Nothing in the checker was changed.**

**Scope limit, stated first.** This machine is an Apple M3: `torch.cuda.is_available()`
is `False` and `triton` is not installed. **No Triton kernel was executed in this
session.** What made the work possible anyway is that the banked run's inputs are
exactly reproducible — see §1 — so every comparison below is against *real
Triton-on-T4 measurements on the identical input*. Where a claim still rests on a
CPU-side quantity, it says so.

---

## Verdict up front

| step | answer |
|---|---|
| 1. Cause of the 0.41–1.67× gap | **Measurement artifact, entirely mine.** I compared different inputs. Not floating point, not a Triton smoothness violation. |
| 2. Does the sandwich hold on real Triton kernels? | **Yes.** Verified against the banked Triton-measured tolerance on **210/210** evaluable invocations, 27 operators. Benign, with margin quantified in §3. |
| 3. Remaining 10 operators | All checked. **27 of 29 satisfy the assumptions; 2 (`argmax`, `argmin`) fail structurally and predictably.** No unpredictable failure found. |
| 4. New finding not in the original | **Three attention operators leave the linear regime on saturating-softmax *adversarial* inputs** — the argmax mechanism again, input-conditional rather than operator-conditional. §4. |

---

## 1. Cause of the discrepancy: I compared different inputs

`probes/validate.py` (the original cross-check) drew inputs from
`verification/specs/*.valid_shapes` — `(512,512)`, `(256,1024)`, `(1,512)` … and
set `gamma = ones`, `beta = zeros`. **The banked corpus does not use those.**
`benchmarks/autokernel/files/tritonbench_registry.py` builds every input from
fixed, much smaller shapes with *random* affine parameters:

| family | banked corpus input | what `validate.py` used |
|---|---|---|
| row-wise (`softmax`, `layernorm`, …) | `(64, 128)` | `(512, 512)` … `(2048, 128)` |
| `matmul` | `(32,16) @ (16,32)` | `(512,512,512)` … |
| attention | `(64, 32)` ×3 | `(128,64)` … |
| `layernorm`/`rmsnorm` γ, β | `rng.normal(...)` | `ones` / `zeros` |

`m` differs by up to 32× and the tolerance depends on `m` through
`sqrt(2 ln 2m)`, so the CVs were never comparable. **There was no CPU/GPU
numerical phenomenon to explain.**

### The fix: replay the exact inputs

`probe_redundancy.py` builds inputs from `np.random.default_rng(0)` via
`entry["input_fn"](rng)`, and makes exactly **6 counted draws per corpus entry**
— 1 mutant call then 5 reference calls, the 2 warm calls being rolled back by
the snapshot/restore around them. numpy's `Generator` is device-independent, so
entry `k`'s inputs are draws `6k … 6k+5` **reproducible bit-for-bit on any
machine**. `probes/replay.py` reconstructs all 40 entries (order verified
against the banked JSON: exact match) and regenerates every input.

The perturbation deltas are *not* reproducible (CUDA `randn_like`), so
per-sample `s_k` cannot match. But `adaptive_tol` is what the bound is about,
and with `n=40` its own sampling error is only ≈ `CV/sqrt(40)` ≈ 2–3%.

### Result, same input, Triton-on-T4 vs torch-on-CPU

**27 of 29 operators agree to within 0.87 – 1.11×** — at or below the estimator's
own sampling noise. Median ratio across operators: **1.000**.

The two that did not, and why — both explained, neither a GPU effect:

- **`cross_entropy`, initially 12.99×.** My error. The Triton host wrapper ends
  `return per_sample_loss.mean()` — the output is a **scalar** (`m = 1`), not the
  per-row vector I had written. With the correct semantics the ratio is **0.872**,
  and the theory predicts the banked value directly: `sigma·||mean_i g_i||·Phi^-1(0.975)·3`
  ≈ `1e-3 · 0.125 · 1.96 · 3` = `7.4e-4` against a banked **7.5e-4**. It is a
  *confirmation*, not an outlier — and its CV of **0.735** against the
  half-normal prediction of **0.7555** is the `m = 1` case of §4's ceiling
  landing exactly where the model says it should.
- **`argmin`, 3.00×.** `int64` index output; the 40-sample vector takes values
  like `{0, 47, 104}` — integer index jumps. Structurally outside the theorem
  (§5), and a ratio between two such draws is not a meaningful quantity.

**Candidate explanations explicitly ruled out.** Floating-point precision
(fp32 accumulation order, tensor-core rounding) cannot produce factor-2
disagreements when the measured agreement on identical inputs is 0.87–1.11×
across 210 invocations — fp effects here are ~1 ulp, and the sensitivities are
10²–10⁵ ulp. Triton-specific smoothness violation is ruled out for the same
reason, plus the direct C¹ test in §3. What remains is the input mismatch,
which is demonstrated rather than inferred.

---

## 2. Does the bound hold on real Triton kernels? Yes — benign, with margin

The sandwich was re-evaluated **using the banked Triton-measured `adaptive_tol`**,
with `sigma` computed from the (identical) input and `L`, `m` from the reference's
mathematics:

```
    2.023 sigma L   <=   adaptive_tol_TRITON   <=   3 sigma L (sqrt(2 ln 2m) + sqrt(2 ln(n/eta)))
```

| | result |
|---|---|
| evaluable primary invocations (27 in-scope operators) | **210** |
| passing **both** sides | **210 / 210** |
| `adaptive_tol_TRITON / (3 sigma L)` | 2.07 – 4.06 across operators |

The guarantee is `1 - eta - (n+1)/2^n` = `1 - 0.05 - 2.0e-5` at `n=40, eta=0.05`;
observed violations 0, consistent. **The discrepancy is benign and does not
threaten the guarantee** — indeed there is no discrepancy left to be benign
about: on identical inputs the two agree to within sampling noise.

**What is still CPU-side:** `L` (the Jacobian `2->inf` norm) and `m` are computed
from the math-equivalent reference, not from the Triton kernel. This is sound
because `L` is a property of the mathematical function and the two implementations
are shown to agree on `adaptive_tol` — which is `3 sigma` times a statistic *of*
`L` — to within 0.87–1.11×. A native GPU run would tighten this from "agrees to
within sampling noise" to "identical", and is still worth doing, but it can no
longer change the verdict.

---

## 3. C¹ / linearisation on the corpus's own inputs, all 29 operators

`|| f(x+d) - f(x) ||_inf` vs `|| J d ||_inf` (`torch.func.jvp`), on the replayed
corpus inputs, median over 10 deltas per invocation:

| median C¹ relative error | across the 27 float-output operators |
|---|---|
| **0.001% – 0.387%** | largest: `cross_entropy` 0.387% (`m=1`, so no max-smoothing) |

`argmax` / `argmin` have `int64` outputs and admit no Jacobian at all.

### A GPU-side test of the assumption that needs no GPU

Under linearisation `s = max_i |Z_i|` with `Z` centred jointly Gaussian. **Claim:
for any such `Z`, `CV(max_i |Z_i|) <= CV(|N(0,1)|) = sqrt(pi/2 - 1) = 0.7555`** —
the half-normal (`m_eff = 1`) case is the worst case. Verified by Monte Carlo over
400 random covariance structures (ranks 1–8, `m` 1–60, log-normal row scaling):
max observed **0.7586**, consistent with the ceiling to MC error.

This is a **falsification test applicable directly to the banked Triton
vectors**. With `n=40`, `SE(CV) ≈ 0.12` at the ceiling, so the 2σ decision
threshold is ≈ 1.0. Applied to all 854 banked invocations, the operators whose
real Triton sensitivity CV exceeds it are exactly: **`argmin`**,
**`flash_attention`**, **`causal_flash_attention`** — and for the latter two only
on specific adversarial inputs (§4). On the **primary** input, **4 of 222
invocations violate, all of them `argmin`**.

---

## 4. New finding: three attention operators are *input-conditionally* out of scope

Attributing every banked vector to the check record that produced it
(`probes/attn.py`) localises all attention violations to two named adversarial
inputs:

| operator | input | n | CV median | over ceiling |
|---|---|---:|---:|---:|
| `flash_attention` | `perturbation_tolerance` (primary) | 22 | 0.212 | 0 |
| `flash_attention` | `adversarial_multi_tile_rescaling` | 22 | **1.728** | **22 / 22** |
| `causal_flash_attention` | `perturbation_tolerance` | 5 | 0.221 | 0 |
| `causal_flash_attention` | `adversarial_large_magnitude_qk` | 5 | 0.932 | 1 / 5 |
| `scaled_dot_product_attention` | `adversarial_large_magnitude_qk` | 5 | 0.740 | 0 / 5 |

Both inputs inflate the attention scores: `_multi_tile_rescaling` multiplies
`K`'s tiles 3–6 by **1e4** (`verification/specs/flash_attention.py:65`);
`large_magnitude_qk` is `(Q*20, K*20)`. Direct confirmation
(`probes/saturation.py`):

| input | peak attention weight | C¹ rel. error | CV |
|---|---:|---:|---:|
| `flash_attention` primary | 0.327 | **0.03%** | 0.19 |
| `flash_attention` / `multi_tile_rescaling` | 1.000000 | **85.28%** | 1.24 |
| `causal` / `large_magnitude_qk` | 1.000000 | **15.00%** | 0.69 |
| `sdpa` / `large_magnitude_qk` | 1.000000 | **27.44%** | 0.99 |

**This is the `argmax` mechanism, not a new one.** `softmax(beta·s) -> argmax` as
`beta -> inf`; once the peak weight reaches 1.0 the operator is a hard select,
piecewise constant, Jacobian ≈ 0 a.e. It is **predictable and detectable in
advance** — peak attention weight is computable before the check runs.

*(One caveat on that diagnostic: `causal` primary also shows peak weight 1.0,
because row 0 of a causal mask attends to exactly one key by construction. Peak
weight alone is therefore not sufficient for causal variants; its C¹ error is
0.04%, i.e. fully in scope. The C¹ error is the reliable criterion.)*

---

> **SUPERSEDED IN PART, 2026-08-25 — see `GPU_NATIVE.md`.** The bound is now
> verified fully natively on Triton kernels (228/228, 27 operators), so the
> "still CPU-side" caveats in §2 and §6 are closed. **§4 of this document is
> corrected there:** native measurement found a *second* out-of-scope mechanism
> (the float32 quantisation floor, distinct from softmax saturation), falsified
> the peak-attention-weight predictor proposed below, and showed the
> `CV <= 0.7555` screen to be insufficient and unstable at n=40. Read §4 here
> only alongside `GPU_NATIVE.md` §4.

---

## 5. Final coverage table

`assumptions` = (A1) Gaussian deltas, (A2) i.i.d. — both true by construction on
any device — and (A3) `C^1` with negligible second order, tested per operator.
`bound verified` = the sandwich evaluated against the **banked Triton-measured**
tolerance.

| # | operator | m | A3 holds? | C¹ rel err | bound verified (Triton tol) | CPU/GPU tol ratio | notes |
|---:|---|---:|:---:|---:|:---:|---:|---|
| 1 | `avg_pool1d` | 48 | yes | 0.001% | **6/6** | 1.077 | |
| 2 | `avg_pool2d` | 96 | yes | 0.002% | **6/6** | 1.059 | |
| 3 | `avg_pool3d` | 384 | yes | 0.001% | **6/6** | 0.940 | |
| 4 | `batchnorm` | 256 | yes | 0.002% | **6/6** | 1.011 | |
| 5 | `causal_flash_attention` | 2048 | yes* | 0.031% | **5/5** | 0.984 | *primary only — §4 |
| 6 | `cross_entropy` | 1 | yes | 0.387% | **6/6** | 0.872 | scalar output; half-normal case |
| 7 | `flash_attention` | 2048 | yes* | 0.037% | **22/22** | 1.012 | *primary only — §4 |
| 8 | `frobenius_norm` | 8192 | yes | 0.008% | **5/5** | 0.981 | |
| 9 | `gelu` | 8192 | yes | 0.015% | **6/6** | 1.028 | |
| 10 | `groupnorm` | 256 | yes | 0.016% | **6/6** | 1.105 | |
| 11 | `instancenorm` | 128 | yes | 0.028% | **6/6** | 1.014 | |
| 12 | `l1norm` | 8192 | yes | 0.007% | **5/5** | 0.989 | |
| 13 | `l2norm` | 8192 | yes | 0.014% | **5/5** | 0.983 | |
| 14 | `layernorm` | 8192 | yes | 0.010% | **16/16** | 1.006 | |
| 15 | `log_softmax` | 8192 | yes | 0.012% | **6/6** | 0.986 | |
| 16 | `matmul` | 1024 | yes | 0.002% | **22/22** | 1.000 | exactly linear |
| 17 | `max_pool1d` | 48 | yes | 0.001% | **6/6** | 0.946 | |
| 18 | `max_pool2d` | 96 | yes | 0.001% | **6/6** | 1.006 | |
| 19 | `max_pool3d` | 384 | yes | 0.001% | **6/6** | 1.042 | |
| 20 | `max_reduction` | 64 | yes | 0.003% | **6/6** | 1.003 | |
| 21 | `mean_reduction` | 64 | yes | 0.003% | **5/5** | 0.995 | |
| 22 | `min_reduction` | 64 | yes | 0.002% | **6/6** | 0.985 | |
| 23 | `rmsnorm` | 8192 | yes | 0.012% | **15/15** | 0.992 | |
| 24 | `scaled_dot_product_attention` | 2048 | yes* | 0.037% | **5/5** | 0.873 | *primary only — §4 |
| 25 | `softmax` | 8192 | yes | 0.058% | **10/10** | 0.972 | |
| 26 | `sum_reduction` | 64 | yes | 0.003% | **5/5** | 1.078 | |
| 27 | `swish` | 8192 | yes | 0.022% | **6/6** | 0.971 | |
| 28 | `argmax` | 64 | **NO** | n/a | **excluded** | 1.000 (both at floor) | `int64` output, `J = 0` a.e. |
| 29 | `argmin` | 64 | **NO** | n/a | **excluded** | 3.000 (meaningless) | `int64` output, `J = 0` a.e. |

**Totals: 27 / 29 operators in scope; 210 / 210 evaluable primary invocations
satisfy the bound against the Triton-measured tolerance; 2 operators excluded,
both structurally and predictably.**

Behaviour of the two excluded operators is itself predicted by the theory rather
than merely observed: `J = 0` a.e. ⇒ `s ≡ 0` ⇒ the `1e-6` floor takes over.
`argmax` sits at the floor on 6/6 invocations. `argmin` does not, because integer
index *jumps* make `s` a positive integer — giving it an `adaptive_tol` of **21
to 63 in index units**, a nonsense tolerance that the check nonetheless applies.
That is worth fixing independently of any paper.

### The honest paper claim

> The bound holds for operators whose reference is `C^1` at the evaluation point
> with negligible second-order response at the perturbation scale. Verified on
> **27 of 29** operators of the TritonBench corpus, over 210 invocations, against
> tolerances measured from the real Triton kernels. The two exclusions
> (`argmax`, `argmin`) are index-valued and have zero Jacobian almost everywhere;
> the theory predicts their observed behaviour (collapse to the tolerance floor)
> rather than being silent about it. Three attention operators are in scope on
> ordinary inputs but leave the linear regime on adversarial inputs that saturate
> the softmax, by the same mechanism — a condition detectable in advance from the
> peak attention weight.

Do **not** claim uniform coverage of all 29.

---

## 6. Residual limits

- **No Triton kernel ran in this session.** Every "Triton" number is read from
  the banked 2026-08-25 T4 run. The replay establishes that the inputs are
  identical; it does not re-execute the kernels. A native GPU re-run remains
  worthwhile for `L` (§2) and would let the C¹ test run against Triton outputs
  directly rather than against math-equivalent references.
- **`L` is a Monte-Carlo per-row estimate** (K=400), validated against exact
  `jacrev` row norms at 2.1% / 7.1% / 10.0% on three cases in the original
  `FINDINGS.md` §3. Biased slightly low for a maximum.
- **18 of the 240 primary invocations have no `perturbation_tolerance` record**
  (checker short-circuited before layer 3), spread across 12 operators; 222
  remain, of which 210 are evaluable after excluding `argmax`/`argmin`.
- **Coverage is per-operator, not per-input.** The table certifies the corpus's
  primary input and the five reference redraws. §4 shows the assumption is
  input-conditional for at least one family, so operator-level scope claims
  should carry that qualifier.
- The `CV <= 0.7555` ceiling of §3 is verified numerically over 400 random
  structures, not proved. It is used only as a screening test, and every case it
  flagged was independently confirmed by direct C¹ measurement.
