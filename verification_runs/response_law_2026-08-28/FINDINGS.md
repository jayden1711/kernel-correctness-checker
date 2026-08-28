# The response-curve law: P(catch | bug magnitude) is a derived functional of the structural parent — all 25 banked (op, margin) points pass exact binomial tests, the pooled curve reproduces 0/6/42/90/100% as 0/4/41/96/100%, and the v-series straddle widths follow from the same object

**Derived and validated 2026-08-28, CPU only** against the banked GPU
near-miss records (`../near_miss_2026-08-28/data/near_miss_gpu.json`, 250
m-series perturbation records; `../near_miss_verdict_2026-08-28/data/
v_series_gpu.json`, 250 v-series verdict records). Probe
`probes/response_law.py`, results `data/response_law.json`. This is the
round's answer to the lead *"what does the near-miss corpus make newly
measurable that was previously unidentifiable"*: it made P(miss) a
measurable quantity — and the measurable quantity turns out to be a
**derivable** one. The checker now has a predicted ROC against bug
magnitude, with no fitted constants.

## The law

For an m-series mutant (uniform epilogue mis-scale DELTA = m·ρ₀), the
perturbation check catches iff DELTA·M(x) > max(3·q95₂₀(s), 1e-6), where
over a fresh harness input x and fresh delta draws:

- M(x) = max|f(x)| — an extreme-value functional of the output field,
- s_k = σ(x)·max_i w_i(x)·|z_ik| — the structural parent (closed-form row
  norms; the H1/Gram object),
- q95₂₀ — torch.quantile's interpolated order statistic of 20 draws.

Both random objects on the right are derived from operator structure;

    P_catch(m) = P[ m·ρ₀·M(x) > tol(x, z) ]

is evaluated by Monte Carlo *of the law* (fresh CPU input draws + parent
inverse-transform draws, deterministic seed — the operative form the
attention onset law established for record-level statistics). Nothing from
the GPU data being predicted enters the computation.

## Validation

**T1 — 25 points, exact binomial two-sided tests, 25/25 pass** (worst tail
p = 0.094, sum_reduction m125). Representative rows:

| op | m080 pred/obs | m100 pred/obs | m125 pred/obs |
|---|---|---|---|
| layernorm | 0.2% / 0/10 | 34% / 4/10 | 99.5% / 10/10 |
| softmax | 11% / 1/10 | 45% / 5/10 | 91% / 8/10 |
| gelu | 1.6% / 0/10 | 49% / 6/10 | 99.7% / 10/10 |
| l2norm | 0.2% / 0/10 | 30% / 2/10 | 99.3% / 10/10 |
| sum_reduction | 8.7% / 2/10 | 45% / 4/10 | 89% / 7/10 |

**T2 — the realized-margin distributions match to the percentile.**
Predicted p5/p50/p95 vs banked 10-seed min/med/max at m100:
layernorm 0.862/0.967/1.116 vs 0.840/0.966/1.106; gelu 0.875/0.999/1.188
vs 0.876/1.017/1.195; sum_reduction 0.751/0.980/1.318 vs 0.677/0.939/1.415.
The per-op straddle *widths* — which ops are knife-edge and which are wide —
are the law's own CVs: predicted 8.1/19.6/9.3/8.2/17.5% for
ln/softmax/gelu/l2norm/sum, reproducing the design round's measured 7–15%
(and softmax/sum as the wide pair, their M(x) being a saturating peak and a
64-row Gumbel max respectively).

**T3 — pooled response curve**: predicted 0.0/4.3/40.7/95.6/100.0% vs
observed 0/6/42/90/100% at margins 0.5/0.8/1.0/1.25/2.0.

**T4 — the v-series (verdict-level) straddles follow from the same
machinery** applied to each op's *binding comparator* (binding-check law,
`../binding_law_2026-08-28/`), input-draw randomness only:

| op (binding) | v100 pred | v100 obs | note |
|---|---|---|---|
| layernorm (affine) | 24% | 3/10 | |
| l2norm (cross_shape) | 23% | 2/10 | |
| sum_reduction (cross_shape) | 82% | 9/10 | the knife-edge is *predicted*: sum's cross_shape δ* is rtol-dominated (1e-4 + 1e-4/max\|f\| with max\|f\| ≈ 90), so the input-draw sensitivity of the boundary is ~1%, and the mutant at design margin 1.0 sits above it in ~4 of 5 draws |

Every other (op, v-margin) point is predicted 0% or 100% and observed
0/10 or 10/10. **gelu and softmax are excluded by derivation, not by
failure**: their binding tolerances are the 1e-6 floor ± fp-quantization
noise — the s/ulp < 32 regime the scope round proved outside the parent's
validity domain. The banked gelu offset (design 1.3–1.5× off, floor-draw
wobble) is that exclusion seen from the mutant side.

## What is new here

1. **The checker's detection probability is now a curve you can compute,
   not a table you must measure.** For any operator with closed-form row
   norms and any uniform-scaling bug size, P(catch) comes from CPU
   arithmetic. The m-series' GPU campaign validated it at 25 points; new
   operators inherit the prediction for free.
2. **P(miss) at the boundary is structural, not incidental**: the ~40%
   catch at design margin 1.0 is forced by the two derived CVs (M-draw and
   q95-draw); no retuning of seeds or sample counts moves it without moving
   the tolerance itself. Equivalently: the transition width of the
   checker's response is set by the parent's dispersion — operators with
   flat profiles and stable output maxima (layernorm, l2norm) get sharp
   boundaries; saturating peaks and small-row-count maxima (softmax, sum)
   get wide ones. This was visible in the banked data as unexplained
   variation; it is now the law's random variable, exactly as the onset
   law's "scatter" was.
3. Combined with the binding-check law, the near-miss *design* pipeline is
   now fully closed-form: predict the binding check, predict δ*, predict
   the response curve, and only then spend GPU time confirming.

## Limits

- Same scope as the family it predicts: uniform relative scaling, corpus
  shape (64, 128), five ops; 10 GPU seeds per point bound the test's power
  (a systematic mis-prediction of ≤ ~15 percentage points would not be
  detected at n = 10).
- The law's Monte Carlo uses the CPU fp32 emulations for M(x); GPU output
  maxima differ at fp-noise level, invisible at these margins.
- The v-series comparator emulation ignores the GPU kernel's own e0 (fp
  residual vs the CPU reference) — correct to ~e0/atol ≈ 0.1% of the
  boundary.
- Floor-adjacent bindings (gelu, softmax v-series) are excluded by the
  validity domain; a law for the floor regime would need the fp-noise
  distribution of s near ulp scale, which the taxonomy deliberately
  fences off.

## Reproduce

```bash
.venv/bin/python probes/response_law.py   # ~6 min CPU, deterministic seeds
```
