# Fully GPU-native sandwich verification — 228/228, 27 operators, no CPU in the measurement path

**Measured 2026-08-25 on a Colab T4** (`torch 2.11.0+cu128`, `triton 3.6.0`).
Probes `probes/gpu_native.py`, `probes/attn_native.py`, `probes/floor_native.py`;
raw output in `native_run/`. Session `kccnat`, provisioned and stopped.
**Nothing in the checker was changed.** The `argmin` nonsense-tolerance bug
remains a separate flagged item, untouched.

This closes the last CPU-side gap in `GPU_COVERAGE.md`. It also **corrects two
claims** from that round — see §4, which supersedes `GPU_COVERAGE.md` §4.

> **ADDENDUM 2026-08-27 — `attention_gram/ATTENTION_GRAM.md`.** The attention
> family's M3 residual is now fully accounted for by the exact Gram-matrix
> law (36 banked + 108 fresh out-of-sample GPU measurements; true structural
> correction +3–4%, the "+17%" was single-draw noise). That round's
> falsification run also found a **real bug in the shipped `flash_attention`
> and `scaled_dot_product_attention` reference kernels**: padded key columns
> enter the softmax denominator whenever `N % 32 ≠ 0` (up to ~97% output
> error at N=1; spec `valid_shapes` includes affected shapes). Flagged there,
> not fixed. **Fixed 2026-08-27 with a clean 40/200 regression —
> `verification_runs/attention_mask_fix_2026-08-27/`.**

---

## Verdict up front

| | result |
|---|---|
| Sandwich, fully native | **228 / 228** invocations, **27** operators, both sides |
| `adaptive_tol` native vs banked Triton | 0.861 – 1.084, **median 1.000** |
| Linearisation defect (native, kernel-differentiated) | 0.0028% – 3.66%, **median 0.0196%** |
| Homogeneity slope | 0.9922 – 1.0050 |
| Remaining CPU-side caveats for the 27 | **none** |
| Prior round's peak-attention-weight predictor | **falsified** — §4 |
| Prior round's `CV <= 0.7555` screen | **necessary but not sufficient** — §4 |

---

## 1. How the Jacobian was taken natively

`torch.func.jvp` cannot differentiate a `@triton.jit` kernel — there is no
autograd registration. The native substitute is the directional derivative by
its definition, evaluated **with the kernel itself**. Writing
`s(t) = || f(x + t d) - f(x) ||_inf`, linearity along `d` is exactly
`s(t) = t · s(1)`, so:

```
    defect  =  | s(1) - s(0.1)/0.1 |  /  s(1)          slope = log10( s(1) / s(0.1) )
```

This is a **stronger** test than the CPU `jvp` comparison of the prior round:
`jvp` differentiates the mathematical reference, whereas this differentiates the
kernel that actually ships. `L` was likewise taken from the kernel, via
`E[(J d)_i^2] = sigma^2 ||J_i||^2` with K=400 native launches per invocation.

Inputs are the corpus's own, replayed bit-for-bit from `np.random.default_rng(0)`
with the same 6-draws-per-entry sequence, so every number is directly comparable
to the banked and CPU-derived rounds.

**Coverage is 228, not the prior round's 210** — the native probe calls the
perturbation mechanism directly rather than through `KernelChecker.run()`, whose
layer short-circuiting had dropped 18 records from the banked run. Every
in-scope invocation is now measured: 38 entries × 6 = 228.

---

## 2. The table — fully GPU-native

`tol`, `L`, `m`, defect and slope all from live Triton execution.
`vs bank` = native `adaptive_tol` ÷ the banked Triton median from `GPU_COVERAGE.md`.

| operator | n | m | tol (native) | L (native) | tol/3σL | defect | slope | sandwich | vs bank |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| `avg_pool1d` | 6 | 48 | 5.038e-03 | 5.411e-01 | 2.929 | 0.011% | 1.0000 | **6/6** | 1.069 |
| `avg_pool2d` | 6 | 96 | 2.485e-03 | 2.701e-01 | 3.078 | 0.019% | 1.0000 | **6/6** | 1.032 |
| `avg_pool3d` | 6 | 384 | 3.962e-03 | 3.917e-01 | 3.334 | 0.013% | 1.0000 | **6/6** | 0.917 |
| `batchnorm` | 6 | 256 | 1.648e-02 | 1.944e+00 | 3.014 | 0.012% | 1.0000 | **6/6** | 1.021 |
| `causal_flash_attention` | 6 | 2048 | 7.297e-03 | 1.093e+00 | 2.258 | 0.071% | 1.0000 | **6/6** | 1.076 |
| `cross_entropy` | 6 | 1 | 7.142e-04 | 1.234e-01 | 1.961 | 1.518% | 1.0027 | **6/6** | 0.952 |
| `flash_attention` | 24 | 2048 | 4.231e-03 | 6.134e-01 | 2.220 | 0.103% | 0.9999 | **24/24** | 1.017 |
| `frobenius_norm` | 6 | 8192 | 1.477e-04 | 1.248e-02 | 3.965 | 0.010% | 1.0000 | **6/6** | 0.966 |
| `gelu` | 6 | 8192 | 1.409e-02 | 1.245e+00 | 3.771 | 0.021% | 1.0000 | **6/6** | 1.063 |
| `groupnorm` | 6 | 256 | 1.322e-02 | 1.601e+00 | 2.957 | 0.023% | 1.0000 | **6/6** | 1.084 |
| `instancenorm` | 6 | 128 | 1.570e-02 | 1.855e+00 | 2.589 | 0.029% | 0.9999 | **6/6** | 0.943 |
| `l1norm` | 6 | 8192 | 1.380e-04 | 1.239e-02 | 3.659 | 0.008% | 1.0000 | **6/6** | 0.998 |
| `l2norm` | 6 | 8192 | 1.223e-03 | 1.112e-01 | 3.709 | 0.010% | 1.0000 | **6/6** | 0.976 |
| `layernorm` | 18 | 8192 | 2.888e-02 | 3.210e+00 | 2.958 | 0.014% | 1.0000 | **18/18** | 1.014 |
| `log_softmax` | 6 | 8192 | 1.341e-02 | 1.132e+00 | 3.937 | 0.053% | 1.0000 | **6/6** | 0.985 |
| `matmul` | 24 | 1024 | 5.547e-02 | 5.859e+00 | 3.156 | 0.013% | 1.0000 | **24/24** | 1.016 |
| `max_pool1d` | 6 | 48 | 9.902e-03 | 1.092e+00 | 3.048 | 0.006% | 1.0000 | **6/6** | 0.977 |
| `max_pool2d` | 6 | 96 | 1.026e-02 | 1.076e+00 | 3.215 | 0.009% | 1.0000 | **6/6** | 1.017 |
| `max_pool3d` | 6 | 384 | 1.087e-02 | 1.102e+00 | 3.316 | 0.007% | 1.0000 | **6/6** | 0.969 |
| `max_reduction` | 6 | 64 | 9.442e-03 | 1.078e+00 | 2.924 | 0.023% | 1.0000 | **6/6** | 0.948 |
| `mean_reduction` | 6 | 64 | 8.503e-04 | 9.654e-02 | 2.957 | 0.024% | 1.0000 | **6/6** | 1.000 |
| `min_reduction` | 6 | 64 | 9.485e-03 | 1.074e+00 | 2.973 | 0.020% | 1.0000 | **6/6** | 0.949 |
| `rmsnorm` | 18 | 8192 | 2.784e-02 | 3.307e+00 | 2.897 | 0.012% | 1.0000 | **18/18** | 1.013 |
| `scaled_dot_product_attention` | 6 | 2048 | 3.676e-03 | 6.094e-01 | 2.312 | 0.089% | 1.0000 | **6/6** | 0.861 |
| `softmax` | 12 | 8192 | 1.000e-03 | 1.597e-01 | 2.080 | 0.072% | 1.0000 | **12/12** | 1.005 |
| `sum_reduction` | 6 | 64 | 1.139e-01 | 1.236e+01 | 3.066 | 0.024% | 1.0000 | **6/6** | 1.045 |
| `swish` | 6 | 8192 | 1.280e-02 | 1.198e+00 | 3.561 | 0.031% | 0.9999 | **6/6** | 1.000 |
| `argmax` | — | 64 | — | — | — | — | — | **excluded** | `int64`, `J = 0` a.e. |
| `argmin` | — | 64 | — | — | — | — | — | **excluded** | `int64`, `J = 0` a.e. |

**TOTAL: 228 / 228 invocations satisfy both sides, across 27 operators.**

### Divergence from the prior round

Native `adaptive_tol` ÷ banked Triton median: **0.861 – 1.084, median 1.000**.
No operator diverges meaningfully. The two extremes are
`scaled_dot_product_attention` (0.861, n=6) and `groupnorm` (1.084, n=6); both
have native linearisation defects of 0.089% and 0.023%, i.e. squarely in scope,
and both sit within the q95 estimator's own sampling spread at n=40 on a
6-invocation median. **Nothing here needs explaining beyond sampling noise.**

`L` differs from the CPU-derived round only through the same channel, and the
ratio `tol/(3σL)` (1.633 – 4.157, median 2.983) is consistent with the CPU
round's 2.07 – 4.06.

---

## 3. Two native-only findings, invisible to a CPU reference

**(a) `frobenius_norm` is not bitwise deterministic.** 2 of its 6 invocations
return a different result on repeat, by exactly **1 ulp (3.73e-09)**. This is
the `atomic_add` cross-block reduction the kernel's own docstring warns about
(`TritonBench/reference/frobenius_norm.py`: *"reduces across the WHOLE tensor …
give this file extra scrutiny"*) — atomics do not fix summation order. Every
other operator is bitwise deterministic over 12 repeats (226/228).

It is **benign here**: the nondeterminism is 4 orders of magnitude below the
measured sensitivity (min `s`/ulp = 10226 for this operator), so it cannot move
`adaptive_tol` or any verdict. But it means `f` is strictly not a function on
this operator, which no CPU-reference study could have seen.

**(b) `cross_entropy`'s smallest sensitivity sample sits at the fp floor.**
Its minimum over 40 samples is **2.0 × ulp**; median 360 × ulp. This is a
consequence of `m = 1`: the scalar output makes the sensitivity half-normal
(§`FINDINGS.md`), which has real mass near zero, so the smallest of 40 draws
lands in the quantisation floor. **`q95` is the second-largest and is
unaffected**, so the tolerance and the bound are fine — but it is also why
`cross_entropy` carries the largest linearisation defect of the 27 (1.518%
median, 3.66% worst).

Every other operator has **min `s`/ulp ≥ 3350** across all 40 samples, and all
228 outputs are finite.

---

## 4. Attention, re-measured natively — this supersedes `GPU_COVERAGE.md` §4

The prior round attributed all attention out-of-scope behaviour to softmax
saturation, and proposed peak attention weight as an advance predictor. **Native
measurement shows there are two distinct mechanisms, and the predictor does not
work.** Adversarial inputs seeded (the spec's `_make_qkv` uses bare
`torch.randn`), 5 independent draws each.

| op | variant | peak wgt | s/ulp | defect | CV median | in scope? |
|---|---|---:|---:|---:|---:|:---:|
| `flash_attention` | primary | 0.370 | 8912 | 0.2% | 0.247 | **yes** |
| `flash_attention` | `approx_denominator` | **1.000** | 91595 | 0.3% | 0.270 | **yes** |
| `flash_attention` | `wrong_causal_mask` | **1.000** | 6408 | 0.1% | 0.228 | **yes** |
| `causal_flash_attention` | primary | 0.291 | 15947 | 0.1% | 0.262 | **yes** |
| `scaled_dot_product_attention` | primary | 0.374 | 15250 | 0.1% | 0.181 | **yes** |
| `flash_attention` | `multi_tile_rescaling` | 1.000 | 2220 | **99.3%** | 1.869 | no — saturation |
| `causal_flash_attention` | `large_magnitude_qk` | 1.000 | 118355 | **23.7%** | 1.033 | no — saturation |
| `scaled_dot_product_attention` | `large_magnitude_qk` | 1.000 | 7699 | **24.0%** | 1.074 | no — saturation |
| `flash_attention` | `last_tile_dropped` | 1.000 | **2.00** | **900%** | 0.315 | no — fp floor |
| `flash_attention` | `skip_rescaling` | 1.000 | **2.00** | **900%** | 0.080 | no — fp floor |
| `flash_attention` | `equal_attention_weights` | 0.016 | **3.00** | **900%** | 0.000 | no — fp floor |

**Kernel nondeterminism is ruled out** as an explanation: `det_floor = 0` on
every attention variant over 12 repeats.

### Mechanism (i): softmax saturation — as previously described

`multi_tile_rescaling` (K's tiles 3–6 × 1e4) and `large_magnitude_qk` (Q,K × 20)
collapse softmax to a hard select. Genuine nonlinearity: defect 24–99%, CV above
the `0.7555` ceiling. This part of the prior round stands.

### Mechanism (ii): float32 quantisation floor — NEW, and previously missed

`last_tile_dropped`, `skip_rescaling` and `equal_attention_weights` give
`s = 2–3 ulp`. The perturbation response is **below float32 granularity**, so
the check is measuring representable-number spacing, not `||J d||_inf`. The
signature is a defect of exactly **900%**, which is what
`|s(1) - s(0.1)/0.1| / s(1)` returns when `s(t)` is *constant* in `t` — the
response is pinned to the quantisation step regardless of perturbation size.
`last_tile_dropped` reaches it because it sets `V[-1] = 1e4`, putting the output
at magnitude 1e4 where one ulp is 9.77e-04.

This is a different failure from saturation and needs a different screen.

### The peak-attention-weight predictor is falsified

Peak weight reaches **1.000 in 6 of 9 non-primary variants**, but two of those —
`approx_denominator` (defect 0.3%) and `wrong_causal_mask` (defect 0.1%) — are
**fully in scope**. The predictor has false positives and is not usable. The
prior round proposed it on 3 data points; 5-seed native measurement across 9
variants does not support it.

### The `CV <= 0.7555` ceiling is necessary but not sufficient

The ceiling is a correct property of the linear regime, but as a *screen on 40
samples* it fails twice over:

- **It misses mechanism (ii).** `skip_rescaling` has CV median **0.080** — far
  below the ceiling — while its defect is 900%.
- **It is unstable at n=40 on these inputs.** Across 5 seeds, `wrong_causal_mask`
  ranged CV **0.164 – 5.496** and `skip_rescaling` **0.080 – 3.333**. This also
  explains the prior round's banked CV of 1.7276 for `multi_tile_rescaling`
  against a native 0.30 on the first draw: **seed variance, not a device
  difference.** Single-seed CV numbers for adversarial attention inputs should
  not be quoted.

**The reliable native screen is the linearisation defect itself** — one extra
kernel call per delta — **paired with `s/ulp`** to catch the quantisation floor.
Both are cheap and both are computed from the kernel.

### The two sandwich failures, and what they mean

| variant | side failed | cause |
|---|---|---|
| `skip_rescaling` | lower | `L = 343` is inflated: the MC row norms are computed from ulp-floor differences, so `L` is not a Jacobian estimate at all |
| `equal_attention_weights` | upper | `tol` is exactly `1.000e-06` — the **absolute floor** clamped it |

The second is worth stating precisely, because it is a limit of the theorem's
statement rather than of the operator: **the sandwich bounds `3σ·q95(s)`, but
the shipped `adaptive_tol` is `max(3σ·q95(s), 1e-6)`.** When the floor binds,
the upper bound can be violated by the clamp. No in-scope primary invocation
ever hits the floor, which is why this never surfaced before.

---

## 5. Final coverage table — the paper version

| # | operator | assumptions (A1–A3) | bound verified, GPU-native | invocations | exclusion reason |
|---:|---|:---:|:---:|---:|---|
| 1–27 | `avg_pool{1,2,3}d`, `batchnorm`, `cross_entropy`, `frobenius_norm`, `gelu`, `groupnorm`, `instancenorm`, `l1norm`, `l2norm`, `layernorm`, `log_softmax`, `matmul`, `max_pool{1,2,3}d`, `max_reduction`, `mean_reduction`, `min_reduction`, `rmsnorm`, `softmax`, `sum_reduction`, `swish` | **hold** | **yes** | **204/204** | — |
| — | `flash_attention`, `causal_flash_attention`, `scaled_dot_product_attention` | **hold on ordinary inputs** | **yes** | **24/24** | out of scope on saturating or fp-floor adversarial inputs (§4) |
| 28 | `argmax` | **A3 fails** | excluded | — | `int64` output, `J = 0` a.e.; theory predicts the observed floor collapse |
| 29 | `argmin` | **A3 fails** | excluded | — | as above; separately, its tolerance is 21–63 *in index units* (flagged bug, not fixed this pass) |

*(The 27 in-scope operators comprise 204 non-attention + 24 attention primary
invocations = 228.)*

### The claim to make in the paper

> The bound holds for operators whose reference kernel is `C^1` at the evaluation
> point, with a perturbation response that is both above the floating-point
> quantisation floor and below the onset of saturation. Verified **natively on
> Triton kernels on a T4**, over **228 invocations across 27 of the corpus's 29
> operators**, with the Jacobian taken by directional differentiation of the
> kernel itself. Two operators (`argmax`, `argmin`) are index-valued with zero
> Jacobian almost everywhere and are excluded; the theory predicts their observed
> collapse to the tolerance floor. Three attention operators are in scope on
> ordinary inputs and leave it on adversarial inputs by two distinct mechanisms —
> softmax saturation and the float32 quantisation floor — both detectable in
> advance by the linearisation defect together with the response-to-ulp ratio.

Do **not** claim the peak attention weight predicts scope, and do **not** quote
single-seed CV values for adversarial attention inputs.

---

## 6. Residual limits

- **`L` remains a Monte-Carlo per-row estimate** (K=400 native launches),
  now computed from Triton output but still an estimate. In the prior round it
  validated against exact `jacrev` row norms at 2.1% / 7.1% / 10.0%. It is biased
  slightly low for a maximum, which makes the reported upper-bound looseness
  slightly optimistic and the lower bound slightly conservative. An exact native
  `L` would need `m` backward passes through a kernel that has no backward.
- **`m` and `sigma` are exact**, read from the kernel's own output and input.
- **Six invocations per operator** (the corpus's mutant draw plus five reference
  redraws), one input shape per operator. Coverage is per-operator at the
  corpus's configuration, not across the shape space — and §4 shows scope is
  input-conditional for at least one family.
- **The `0.7555` ceiling is verified numerically**, not proved (400 random
  centred-Gaussian structures, prior round). §4 now shows it is in any case
  insufficient as a screen.
- **Attention adversarial inputs are seeded here but unseeded in the shipped
  spec** (`_make_qkv` uses bare `torch.randn`), so the checker's own runs draw a
  different input every time. Given the CV instability measured in §4, that is
  worth knowing independently of this work.
