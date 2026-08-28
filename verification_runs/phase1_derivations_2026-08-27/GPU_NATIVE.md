# Phase-1 GPU-native verification — 162/162 sandwich, 27 operators, real Triton kernels. M3 falls to 0.857, and the predicted cause was wrong.

**Measured 2026-08-27 on a Colab T4** (`torch 2.11.0+cu128`, `triton 3.6.0`,
Tesla T4 cap 7.5 — the same environment as the 2026-08-25 round, so every
number here is directly comparable). Session `kccphase1`, provisioned and
stopped. Probes `probes/phase1_native.py`, `probes/pass2.py`,
`probes/verify_kernels.py`; kernels `kernels/phase1_kernels.py`; raw output in
`native_run/`. **Nothing in the checker's verdict path was changed.** One
scale bug in `structural_l.y_profile` was found and fixed — §6.

---

## Verdict up front

| | result |
|---|---|
| Sandwich, fully native, Phase-1 operators | **162 / 162** invocations, **27** operators, both sides |
| Triton kernels written and verified vs torch | **27 / 27** correct before any measurement |
| Closed-form `L` vs converged probe (K=20000) | **1.000 – 1.023, median 1.012** — the closed forms are right |
| **M3 R², full 54-operator corpus** | **0.8567** (was 0.9579 on 27) — **the standing prediction that it falls is CONFIRMED** |
| **Why it falls** | **The scan family, +24.7%.** NOT the `m=1` losses — **that prediction is REFUTED** |
| `m=1` losses under M3 | **−1.1% median.** Essentially unbiased |
| R² with the 4 scan operators excluded | **0.9635** — *better* than the original 27 |
| The `sigmoid`/`tanh`/`swiglu` spread claim (~5e10) | **REFUTED.** Native max spread across all inputs is **16.3** |
| Sandwich on adversarial input | 103/103 lower, 99/103 upper — 4 upper deviations, all the `1e-6` tolerance floor, §5 |
| **`bce_loss`'s 52.6% defect** | **NOT a floor case.** New third category: the `m=1` sensitivity limit — §7 |

---

## 0. A prerequisite that was not in the plan

`TritonBench/reference/` held only the original 29 kernels. The native
methodology differentiates **the kernel that ships**, not a torch stand-in, so
27 real `@triton.jit` kernels had to be written first
(`kernels/phase1_kernels.py`, 475 lines). **All 27 were verified against their
torch reference on the T4 before any measurement ran** — max relative error
2.9e-07, worst case `cumsum` at 2.91e-07. A wrong kernel produces a confident
wrong sandwich result, so this gate came first.

The scalar-output losses use a **two-stage deterministic reduction** (per-row
partials, then one program summing them) rather than `tl.atomic_add`, because
`frobenius_norm`'s atomic reduction is measurably non-bitwise-deterministic
(GPU_NATIVE.md §3a). Result: **det_floor = 0 on all 27 operators**, no repeat
of that finding.

---

## 1. Method

Identical to `adaptive_tol_theory_2026-08-25/probes/gpu_native.py`, constants
unchanged: `NS=40`, `K_MC=400`, `ETA=0.05`, `DELTA_SCALE=1e-3`,
`T_LADDER=[0.01, 0.1, 1.0]`.

`torch.func.jvp` cannot differentiate a `@triton.jit` kernel. The native
substitute is the directional derivative by definition, evaluated with the
kernel itself: `s(t) = ||f(x+td) − f(x)||_inf`, linear along `d` iff
`s(t) = t·s(1)`, so `defect = |s(1) − s(t)/t| / s(1)`.

Inputs come from each operator's own `KernelSpec.make_inputs` at
`valid_shapes[0]`, six draws each, seeded per `(op, invocation)`. The Phase-1
operators are not in the TritonBench corpus and so have no banked draw to
replay. **`valid_shapes[0]` for every invocation, not `valid_shapes[inv % n]`**
— that list is the `cross_shape` sweep and deliberately contains degenerate
edge shapes; a first pass that cycled it crashed on `(1,)` (std undefined → NaN,
and NaN is truthy so `or 1.0` does not catch it) and on `cumsum_exclusive` at
width 1 (output identically zero → `L = 0`). Both fixed at source.

---

## 2. The table — fully GPU-native, all 27 Phase-1 operators

`tol`, `L`, `m`, defect, slope and spread all from live Triton execution.
`L_mc/L_cl` is the K=400 probe over the closed form (biased high by design —
see §3). `M3` is predicted ÷ measured `y`. `spread` is max/median of the native
row-norm profile. `det` = bitwise deterministic over 5 repeats.

| operator | family | n | m | tol | L | tol/3σL | defect | slope | sandwich | L_mc/L_cl | M3 | spread | det |
|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|--:|--:|--:|:-:|
| `relu` | activation | 6 | 4096 | 1.282e-02 | 1.124e+00 | 3.775 | 0.004% | 1.0000 | **6/6** | 1.124 | 0.987 | 1.13 | yes |
| `leaky_relu` | activation | 6 | 4096 | 1.226e-02 | 1.124e+00 | 3.616 | 0.002% | 1.0000 | **6/6** | 1.124 | 1.028 | 1.22 | yes |
| `sigmoid` | activation | 6 | 4096 | 3.011e-03 | 2.765e-01 | 3.625 | 0.040% | 1.0000 | **6/6** | 1.106 | 1.007 | 1.24 | yes |
| `tanh` | activation | 6 | 4096 | 1.182e-02 | 1.099e+00 | 3.548 | 0.034% | 1.0000 | **6/6** | 1.099 | 1.004 | 1.68 | yes |
| `selu` | activation | 6 | 4096 | 1.803e-02 | 1.884e+00 | 3.163 | 0.140% | 1.0003 | **6/6** | 1.072 | 1.011 | 1.81 | yes |
| `elu` | activation | 6 | 4096 | 2.117e-02 | 2.137e+00 | 3.301 | 0.143% | 0.9995 | **6/6** | 1.069 | 0.978 | 2.14 | yes |
| `softplus` | activation | 6 | 4096 | 1.084e-02 | 1.073e+00 | 3.370 | 0.024% | 1.0000 | **6/6** | 1.074 | 1.026 | 2.13 | yes |
| `hardsigmoid` | activation | 6 | 4096 | 2.140e-03 | 1.873e-01 | 3.776 | 0.033% | 1.0000 | **6/6** | 1.124 | 1.018 | 1.13 | yes |
| `new_gelu` | activation | 6 | 4096 | 1.342e-02 | 1.249e+00 | 3.571 | 0.031% | 1.0000 | **6/6** | 1.106 | 0.982 | 2.53 | yes |
| `cumsum` | scan | 6 | 32768 | 2.347e-01 | 2.471e+01 | 3.163 | 0.022% | 1.0001 | **6/6** | 1.092 | 1.229 | 1.55 | yes |
| `cumsum_reverse` | scan | 6 | 32768 | 2.373e-01 | 2.471e+01 | 3.185 | 0.034% | 1.0000 | **6/6** | 1.092 | 1.220 | 1.55 | yes |
| `cumsum_exclusive` | scan | 6 | 32768 | 2.297e-01 | 2.462e+01 | 3.105 | 0.032% | 1.0000 | **6/6** | 1.089 | 1.255 | 1.54 | yes |
| `masked_cumsum` | scan | 6 | 32768 | 1.634e-01 | 1.756e+01 | 3.083 | 0.030% | 1.0000 | **6/6** | 1.049 | 1.257 | 1.55 | yes |
| `matvec` | matmul-var | 6 | 512 | 2.610e-01 | 2.449e+01 | 3.571 | 0.029% | 1.0000 | **6/6** | 1.096 | 0.982 | 1.09 | yes |
| `batched_matmul` | matmul-var | 6 | 65536 | 1.721e-01 | 1.477e+01 | 3.891 | 0.022% | 1.0000 | **6/6** | 1.090 | 1.005 | 1.31 | yes |
| `diagonal_matmul` | matmul-var | 6 | 262144 | 4.092e-02 | 4.605e+00 | 2.927 | 0.005% | 1.0000 | **6/6** | 1.027 | 1.014 | 6.84 | yes |
| `triangular_matmul` | matmul-var | 6 | 262144 | 3.420e-01 | 2.766e+01 | 4.121 | 0.060% | 1.0000 | **6/6** | 1.110 | 1.009 | 1.23 | yes |
| `mse_loss` | loss (m=1) | 6 | 1 | 2.834e-05 | 5.582e-03 | 1.715 | 26.138% | 0.9750 | **6/6** | 1.010 | 1.091 | 1.00 | yes |
| `huber_loss` | loss (m=1) | 6 | 1 | 8.695e-06 | 1.596e-03 | 1.828 | 18.056% | 0.9423 | **6/6** | 1.020 | 1.008 | 1.00 | yes |
| `bce_loss` | loss (m=1) | 6 | 1 | 3.545e-05 | 2.420e-02 | 1.726 | 52.649% | 1.2085 | **6/6** | 1.231 | 0.881 | 1.00 | yes |
| `kldiv_loss` | loss (m=1) | 6 | 1 | 1.668e-05 | 3.293e-03 | 1.732 | 12.242% | 0.9945 | **6/6** | 1.029 | 1.073 | 1.00 | yes |
| `nll_loss` | loss (m=1) | 6 | 1 | 2.691e-04 | 4.391e-02 | 2.063 | 0.137% | 0.9999 | **6/6** | 0.994 | 0.918 | 1.00 | yes |
| `rope` | other | 6 | 65536 | 1.464e-02 | 1.158e+00 | 4.216 | 0.006% | 1.0000 | **6/6** | 1.158 | 1.004 | 1.16 | yes |
| `swiglu` | other | 6 | 131072 | 4.008e-02 | 4.852e+00 | 2.759 | 0.046% | 1.0000 | **6/6** | 1.017 | 1.025 | 13.13 | yes |
| `logsumexp` | other | 6 | 512 | 9.846e-04 | 1.546e-01 | 2.241 | 0.639% | 1.0005 | **6/6** | 1.020 | 1.004 | 2.19 | yes |
| `std_reduction` | other | 6 | 512 | 5.149e-04 | 4.876e-02 | 3.512 | 0.379% | 0.9997 | **6/6** | 1.102 | 0.988 | 1.10 | yes |
| `var_reduction` | other | 6 | 512 | 1.012e-03 | 1.018e-01 | 3.296 | 0.289% | 0.9993 | **6/6** | 1.052 | 1.010 | 1.15 | yes |

**TOTAL: 162 / 162 invocations satisfy both sides, across 27 operators.**
Zero errors, zero non-finite outputs, `det_floor = 0` everywhere.

Together with the banked 27, the closed-form corpus is now **54 operators**
(56 specs minus `argmax`/`argmin`, which return `int64` and have `J = 0` a.e.;
both remain excluded exactly as before and route through `_check_exact_match`).

---

## 3. `L` is structural — confirmed on real kernels

K=400 is biased **high**, exactly as the original round found. Raising K
separates "the formula is wrong" from "the estimator is biased":

| `L_mc(K) / L_closed` | min | median | max | within ±5% |
|---|---:|---:|---:|---:|
| K = 400 *(what the main run used)* | 0.933 | **1.107** | 1.164 | 2 / 12 |
| K = 4000 | 0.995 | 1.027 | 1.046 | **12 / 12** |
| K = 20000 | 1.000 | **1.012** | 1.023 | **12 / 12** |

The original round's figures were 1.081 → 1.025 → 1.010. **The same
convergence, on a different operator set, from a different derivation.** The
closed forms are right; the K=400 gap is estimator bias and nothing else.

Per operator at K=20000: `logsumexp` 1.000, `swiglu` 1.003, `tanh` 1.010,
`sigmoid` 1.011, `mse_loss` 1.011, `new_gelu` 1.011, `cumsum` 1.013,
`cumsum_reverse` 1.013, `std_reduction` 1.014, `relu` 1.016, `matvec` 1.016,
`rope` 1.023.

---

## 4. M3 re-fit — the prediction was right about the direction and wrong about the cause

| corpus | n | ops | **R²** | median pred/meas | spread | ±10% |
|---|--:|--:|--:|--:|--:|--:|
| original 27 *(reproduced from banked data as a control)* | 38 | 27 | **0.9579** | 1.022 | 1.26× | 36/38 |
| Phase-1 27 | 162 | 27 | 0.8388 | 1.014 | 1.86× | 124/162 |
| **FULL 54-operator corpus** | **200** | **54** | **0.8567** | 1.016 | 1.86× | 160/200 |

The control reproduces the banked 0.958 to four digits (0.9579, spread 1.26×,
36/38 within ±10%, min 0.929 / median 1.022 / max 1.167), so the matching
procedure is right before anything new is added.

**R² falls from 0.9579 to 0.8567. The standing prediction is confirmed.**

### But the predicted mechanism is refuted

Per family, median predicted ÷ measured:

| family | n | median | |
|---|--:|--:|---|
| **scan** (`cumsum`, `_reverse`, `_exclusive`, `masked_`) | 24 | **1.247** | **+24.7% — the entire degradation** |
| `loss (m=1)` (5 new + `cross_entropy`) | 31 | 0.989 | −1.1% |
| activation (9 new + `gelu`, `swish`) | 56 | 1.001 | +0.1% |
| matmul-variant (4 new + `matmul`) | 28 | 1.006 | +0.6% |
| other new (`rope`, `swiglu`, `logsumexp`, `std`, `var`) | 30 | 1.006 | +0.6% |

**R² with the four scan operators excluded: 0.9635** — marginally *better* than
the original 27. **24 of 200 invocations account for the whole drop.**

### The `m=1` prediction was based on a misreading, and it was mine

The standing prediction held that adding five `m=1` losses would drag M3 down
because `cross_entropy` was "already M3's worst case at +121%". **That is false.
The +121% belongs to model M1′** — the theorem's leading term `√(2 ln 2m)` used
as an estimator — which the original round reported at **R² = −0.34, worse than
predicting the mean** (`generalization/FINDINGS.md` §B.1). Under **M3**, the
model actually being re-fit, `cross_entropy` in the banked data is
**−1.8%** (`y_M3` 1.8644 vs `y_meas` 1.8989). It was never M3's worst case.

Measured now, the five new `m=1` losses: `mse` +9.1%, `kldiv` +7.3%,
`huber` +0.8%, `nll` −8.2%, `bce` −11.9%. Unbiased as a group, and the corpus
going from 1 to 6 `m=1` operators did not hurt the fit. **`m=1` is not the
problem and never was.**

### What the scans reveal is a confirmation, not a surprise

M3 simulates `E[q95(max_i (‖J_i‖/L)|z_i|)]` **under an orthogonal-rows
assumption**, and the original round stated the consequence exactly: it
"over-predicts exactly where rows are genuinely correlated", with
`flash_attention` (+17%, rows sharing a softmax denominator) as its worst case.

A prefix scan is the most correlated Jacobian there is: row `i` is
`(1,…,1,0,…,0)` with `i+1` ones, so row `i`'s support is a strict **subset** of
row `j`'s for every `i<j`. Nested, not merely correlated. The measured +24.7%
over-prediction extends the stated mechanism past attention's +17% in the
direction the mechanism predicts. *(Note 2026-08-27: the attention "+17%" was
subsequently shown to be mostly single-draw noise — its true structural
correction is +3–4%; see
`../adaptive_tol_theory_2026-08-25/attention_gram/ATTENTION_GRAM.md`. The
scans' +24.7% is unaffected and remains the genuine correlation case.)* **M3's residual remains interpretable and
correctly signed; it is the orthogonality assumption being paid for, at the
worst operator family available.**

This is a scope statement about M3, not a defect in the scan derivations —
those agree with a converged probe to 1.013 (§3).

---

## 5. The spread claim — refuted

The CPU-side round reported `sigmoid`/`tanh`/`swiglu` row-norm spreads of
~**5e10 / 7e9 / 8e9** against a previous corpus maximum of 38.7 (`softmax`),
and flagged it as a reason to expect trouble. **Native measurement does not
reproduce it.**

| | CPU closed-form claim | **native, ordinary input** | **native, adversarial/saturating** |
|---|---:|---:|---:|
| `sigmoid` | 4.8e10 | **1.24** | 3.30 |
| `tanh` | 7.2e9 | **1.68** | 4.28 |
| `swiglu` | 8.0e9 | **13.13** | 16.27 |
| corpus max, any operator, any input | — | 13.13 | **16.27** |

**The largest spread anywhere in the Phase-1 corpus is 16.3 — below the
previous corpus maximum of 38.7.**

**Why the CPU number was wrong.** It computed `max/median` over the *positive*
entries of an autograd row-norm vector in a saturating regime where most
entries are denormal (~1e-40) rather than exactly zero. Dividing by a denormal
median manufactures an arbitrarily large ratio. The native probe forms the
profile from finite-difference kernel responses, where anything below fp32
resolution reads as exactly 0 and is excluded by `prof > 0`. **The 8-orders-of-
magnitude figure was an artifact of the CPU measurement, not a property of the
operators.** Corrected here; the FINDINGS.md claim is superseded.

### Does anything break the sandwich? No.

Across **103 adversarial variants with a usable `L`**: lower bound **103/103**,
upper bound **99/103**. Eleven further variants have `L ≤ 0` — the kernel's
output is genuinely independent of the input there (all-negative `relu`,
saturated `tanh`/`selu`/`elu`/`softplus`, `hardsigmoid` outside both knees,
`std`/`var` under a constant shift) — recorded as **vacuous, not violated**.

The four upper-bound deviations are all one mechanism, and it is not the bound:

| operator / variant | y | defect | s/ulp | mechanism |
|---|--:|--:|--:|---|
| `sigmoid` / `near_zero` | 8.05 | **900%** | 1.0 | fp32 quantisation floor — the exact 900% signature from GPU_NATIVE.md §4(ii) |
| `sigmoid` / `saturating_neg` | 3.9e38 | 100% | 0.0 | response below fp32 resolution; `L → 0` |
| `new_gelu` / `near_zero` | 599 | 0.0% | 6033 | `tol` pinned at its `1e-6` floor while `3σL ≈ 1.5e-9` |
| `swiglu` / `gate_saturating_neg` | 2.1e10 | 2.6% | 354330 | gate saturates, output exactly 0, `tol` at the floor |

In every case `tol = max(3·q95(sens), 1e-6)` has hit its **floor** while the
true sensitivity is ~0. The floor is doing its documented job — without it these
checks would degenerate into exact-match — and the ratio `tol/(3σL)` is
meaningless when the denominator is at machine zero. **This is the floor
interacting with a vacuous input, not the theorem failing.** The lower bound,
which is the side that governs whether a correct kernel can be rejected, held
on every single one of the 103.

---

## 6. A real bug in shipped code, found by this run

`structural_l.y_profile` **silently returned `None` for `diagonal_matmul` and
`triangular_matmul` — via a CUDA OOM.**

Its row-count trim only subsamples the tail *below 1e-3 of the max*. For a
profile that is dense near its maximum — `|B|` for random `B`, which is exactly
`diagonal_matmul`'s closed form — almost nothing is trimmed, and the simulation
allocated `200 × 40 × 262144 × 4 B = 8.4 GB` in one chunk. Measured: 7.81 GiB
requested for `diagonal_matmul`, 3.91 GiB for `triangular_matmul`.

**The failure was silent.** `structural_adaptive_tol` catches the exception and
returns `None`, which the caller reads as "this path declines to answer" — so
it presented as *missing coverage*, not as an error. It did not surface on the
original 27 because their largest `m` was 8192; the Phase-1 operators reach
262144.

**Fixed by making the chunk size adaptive** (`ELEM_BUDGET = 32M` elements, ~128
MB). The estimator is **unchanged** — only the batching of it is. Capping rows
instead would have altered the estimator, which is why that was not the fix.
Both operators now predict, and predict well: `diagonal_matmul` 1.014,
`triangular_matmul` 1.009.

---

## 7. `bce_loss` — RESOLVED. Not a floor case; a new `m=1` sensitivity limit

The first draft of this section left `bce_loss` ambiguous: 52.6% linearisation
defect, slope 1.209, yet sandwich 6/6. That ambiguity is now closed by analysis
of the banked per-invocation records (`probes/bce_classification.py`, no GPU).

### Verdict

**`bce_loss` passes on its own merits. It is NOT the `equal_attention_weights`
floor exception, and it must not be folded into that footnote.** But the pass
carries almost no information about linearity, for a reason that is structural
to `m = 1` and applies to all six `m = 1` operators.

### Both floor mechanisms are ruled out

| test | `equal_attention_weights` | `last_tile_dropped` / `skip_rescaling` | **`bce_loss`** |
|---|---|---|---|
| `tol` on the `1e-6` absolute clamp? | **yes — exactly `1.000e-06`** | no | **no — `3.14e-05` to `3.85e-05`, 31–38× the floor** |
| q95 sample vs fp32 ulp | — | **2–3 ulp** | **106 ulp** |
| defect signature | exactly 900% | exactly 900% | **44–79%, varying** |
| CV | 0.000 | 0.283 / 6.293 | 0.618–0.679 (inside the 0.7555 screen) |
| `det_floor` | 0 | 0 | 0 |

The constant **900%** defect is the signature of `s(t)` being *independent of
`t`* — the response pinned to representable-number spacing. `bce_loss`'s defect
varies invocation to invocation (43.8, 47.4, 49.0, 56.3, 62.9, 79.5%), which is
what genuine curvature looks like. Every quantity in its tolerance is a real
measurement.

### Why the bound holds anyway — the actual mechanism

At `m = 1` the output is a scalar, so `base.numel() == 1` and the `L` estimator
`L = √(E[(f(x+d) − f(x))²]) / σ` and the tolerance `tol = 3·q95(s)` are two
statistics of **the same scalar response distribution**. The sandwich ratio
therefore reduces algebraically to

```
        y  =  tol / (3 σ L)  =  q95(s) / √(E[s²])
```

— a pure **shape ratio**, with `σ`, `L` and every overall scale cancelling.
Confirmed numerically: recomputing `q95(s)/RMS(s)` from the banked `sens` arrays
reproduces the banked `y` to 0.6–1.9% for `bce`/`mse`/`huber` (the residual is
sampling noise, since `L` uses an independent `K_MC = 400` draw while `q95` uses
the 40 perturbation samples).

Under exact linearity `s ~ σL·|Z|` is half-normal and `q95/RMS = 1.9600`.
**Curvature deforms that ratio, monotonically, but weakly:**

| operator | defect | `q95/RMS` | skew | vs half-normal |
|---|--:|--:|--:|--:|
| `nll_loss` | 0.14% | 1.9586 | 0.972 | −0.1% |
| `cross_entropy` | 1.52% | 1.9166 | 0.842 | −2.2% |
| `kldiv_loss` | 12.24% | 1.8905 | 0.799 | −3.5% |
| `huber_loss` | 18.06% | 1.8623 | 0.760 | −5.0% |
| `mse_loss` | 26.14% | 1.7484 | 0.599 | −10.8% |
| **`bce_loss`** | **52.65%** | **1.7369** | 0.736 | **−11.4%** |
| *(exact linearity)* | 0% | 1.9600 | 0.995 | — |

The trend is real and orderly: `q95/RMS = 1.9306 − 0.4247·defect`, **R² = 0.829**,
and the fit's zero-defect intercept of **1.9306 lands within 1.5% of the
half-normal 1.9600** — an independent check that the model is the right one.
`nll_loss` is the control: it is linear in its input by construction (it
gathers), and it sits on the half-normal value.

**The bound is simply far wider than that deformation.** At `m = 1` the sandwich
admits `y ∈ [0.6745, 4.8338]`, a **7.17× window**, and the linear prediction
1.9600 sits near its geometric centre. Extrapolating the fit, `y` would not
reach the lower bound until a linearisation defect of **≈296%**. `bce_loss` is
at 53%. Curvature can only push `y` *down* (the fit is decreasing), so the upper
bound is unreachable by this mechanism at all.

### Classification for the paper

**A third category, distinct from both floor mechanisms. Do not extend the
`equal_attention_weights` footnote to cover it.**

| category | mechanism | status of the bound |
|---|---|---|
| absolute-floor exception (`equal_attention_weights`) | `tol` clamped to `1e-6`; bound compares against a constant | **vacuous** — bound not actually tested |
| fp32 quantisation floor (`last_tile_dropped`, `skip_rescaling`) | `s` at 2–3 ulp; measuring representable spacing, not `‖Jd‖` | **not applicable** — `L` is not a Jacobian estimate |
| **`m = 1` sensitivity limit** (all 6 `m=1` operators; `bce_loss` extreme) | `y` degenerates to `q95/RMS`, a shape statistic that curvature barely moves | **sound and genuinely tested — but near-blind to nonlinearity** |

> **CORRECTED 2026-08-27 — the categories are mechanisms, not disjoint
> regions.** `../theory_audit_2026-08-27/` (probes/taxonomy_median.py, 205
> banked invocations) measured what this table only asserted:
>
> - **The two floor categories are the absolute and relative arms of ONE
>   resolvability criterion**, `min(s_med/ulp(‖f‖∞)/32, tol/1e-6) ≤ 1` —
>   15/15 known cases flagged, 0 missed. And the absolute-floor cases
>   (`equal_attention_weights`) *also* sit at 2.5–6 ulp, i.e. inside the
>   relative arm: the first two rows overlap on the real data.
> - **The `m = 1` category does NOT sit cleanly outside those axes.** 8 of 18
>   m=1 invocations have *median* s/ulp of 12–25, below the 32 screen (the
>   106-ulp figure above is the q95 sample; the median — the statistic the
>   scope round validated — is 2.9× smaller). m=1 blindness is a genuinely
>   different mechanism (output-dimension degeneracy), but it **co-occurs**
>   with near-resolution-limited responses rather than excluding them.
> - `flash_attention/multi_tile_rescaling` sits at 2 ulp on some invocations —
>   a genuine fp32-floor member missing from this table's variant list.
>
> Everything below this box — the q95/RMS mechanism, the sound-but-low-powered
> recommendation, keeping `bce_loss` in the "passes" column — **stands
> unchanged.** What is corrected is only the "distinct, non-overlapping
> categories" reading of the table above.

The recommended wording is that for `m = 1` operators the sandwich is **sound
but low-powered**: it holds, and it holds on a real measurement, but *passing
must not be cited as evidence that the linearisation assumption is satisfied*.
`bce_loss` should stay in the "passes" column — moving it to an exclusion list
would be wrong, since its tolerance is a valid measured `3σ·q95` and the check
functions — but it should be excluded from any count of operators whose
**linearisation** was validated. On present evidence that count is 48 of 54:
the six `m = 1` operators are sound-but-unvalidated on this axis.

This is a limitation of the sandwich's **diagnostic power at `m = 1`**, not of
`bce_loss`, and not of the theorem. It generalises: any operator with a scalar
output inherits it.

---

## 8. Reproduce

```bash
export HOME=~/.colab-home
colab new --gpu T4 -s <name>
colab upload -s <name> kernels/phase1_kernels.py /content/phase1_kernels.py
colab upload -s <name> kcc.tgz /content/kcc.tgz        # verification/ only
colab exec  -s <name> -f probes/verify_kernels.py --timeout 600   # gate: 27/27
colab exec  -s <name> -f probes/phase1_native.py  --timeout 900   # 162 invocations, ~49s
colab exec  -s <name> -f probes/pass2.py          --timeout 900   # K-ladder + adversarial, ~49s
colab stop  -s <name>
```

Then `python probes/final.py` locally for the M3 fit.

Total GPU time for the whole round: **under 3 minutes of compute.** The cost is
in staging and in writing the kernels, not in the measurement.

---

## 9. What this supersedes

- **`FINDINGS.md` §5's spread claim is superseded by §5 here.** The
  ~5e10/7e9/8e9 figures are a CPU denormal-division artifact; native spread
  never exceeds 16.3.
- **`FINDINGS.md` §4's prediction is superseded by §4 here.** The direction was
  right (R² falls, 0.958 → 0.857); the attributed cause (`m=1` losses,
  `cross_entropy` at +121%) was wrong and rested on reading an M1′ residual as
  an M3 one. The actual cause is the scan family's nested-prefix Jacobian.
- **`CORPUS_EXPANSION_PLAN.md` §4.2** carries the same wrong attribution and
  should be corrected the same way.
- Phase-1 operators are no longer "derivation-verified, probe-unverified". They
  are **probe-verified on real Triton kernels**, and `structural_l.py`'s
  Phase-1 note should be updated to say so.
