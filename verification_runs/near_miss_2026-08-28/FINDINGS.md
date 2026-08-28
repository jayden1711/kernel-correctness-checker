# Near-miss mutants: 25 margin-targeted kernels land on their design margins on GPU, the response curve is now 0→6→42→90→100% across 0.5–2.0×, and the scale multiplier's (1.642, 4.360) dead interval is populated with 152 flip points

**Built and GPU-validated 2026-08-28** (T4, session `nearmiss`, stopped).
Design probe, generator, GPU probe and analysis in `probes/`; raw results
`data/near_miss_gpu.json`. This closes the standing gap named by the
n_samples, adaptive_tol, tol_floor and l3_margins rounds: the published
corpus fails by median 461× (closest 1.45×, P(miss) ≤ 1.9e-3 everywhere),
so no tolerance experiment could see anything move.

## The family

`TritonBench/near_miss/<op>/m{050,080,100,125,200}.py` — 5 operators
(layernorm, softmax, gelu, l2norm, sum_reduction: normalization,
saturating, smooth-elementwise, scalar-denominator, linear-reduction
response types) × 5 design margins {0.5, 0.8, 1.0, 1.25, 2.0}×. Each is
the reference kernel verbatim with a **mis-scaled epilogue**
(`out * (1 + DELTA)` inside the kernel — a real bug shape, not a wrapper
shim), with DELTA set per operator from the measured ratio
ρ = tol/max|f| (CPU design probe, `design_deltas.py`): the
perturbation-check error is exactly DELTA·max|f|, so the margin
DELTA·max|f|/tol is a ratio of statistics of the same input draw and is
stable across seeds by construction. Registry:
`benchmarks/autokernel/files/near_miss_corpus.py` (same entry shape as
the published corpus; **deliberately never merged into it** — the
published 40/40, 0/200 is defined on the original mutant set).

## GPU validation (10 seeds per mutant, corpus shape (64,128))

1. **Margins land on design.** Median realized margins: 0.475–0.540 at
   target 0.50, 0.949–1.080 at target 1.00, 1.898–2.159 at target 2.00 —
   CPU-designed deltas transfer to the GPU within ~8%. Seed-to-seed CV
   7–15% (sum_reduction 25%, its P95-of-max statistic is intrinsically
   noisier), matching the design probe's prediction.
2. **The response surface is no longer flat.** Perturbation-check catch
   rate pooled over ops: **0/50, 3/50, 21/50, 45/50, 50/50** at design
   margins 0.5 / 0.8 / 1.0 / 1.25 / 2.0. The m100 family genuinely
   straddles the boundary (42%, per-op 2–6 of 10 seeds) — P(miss) is now a
   measurable quantity instead of ≤ 1.9e-3.
3. **The enabled experiment, demonstrated.** Per record the verdict-flip
   scale is s* = 3.0·margin; across the 250 records s* spans
   [1.01, 8.49] with P10–P90 = [1.48, 5.91] and **152 flip points inside
   (1.642, 4.360)** — the interval the adaptive_tol round measured as
   verdict-dead on the published corpus. Any future re-derivation of
   `scale` (equivalently `delta_scale`, §2.2 identifiability) can now be
   scored against real flips. The same applies to the 1e-6 floor
   (tol_floor round): its (5e-7, 6.4e-3) dead zone can be probed by the
   same method with floor-targeted deltas.

## What the full battery says — reported, not hidden

At the **verdict** level every near-miss mutant is still caught 3/3, by
checks *other than* the one it targets: `cross_shape` (atol 1e-4) and the
L2 property checks (`unit_l2_norm`, `unit_variance`, `rows_sum_to_one`,
atol ~1e-3) are **20–30× tighter than the adaptive tolerance** for a
uniform output scaling, so they fire even at the 0.5× perturbation margin.
Two consequences, both useful:

- The family is a **check-level** instrument (which is what tolerance
  experiments operate on — per-check margins), not a verdict-level
  stealth corpus. Anyone wanting verdict-level near-misses must target
  the *binding* check; for scaling-type bugs that is cross_shape/property
  atols, and the same design method transfers directly
  (δ = m·atol_check/max|f| gives cross_shape-marginal mutants at
  δ ≈ 3e-5). *(Built, later on 2026-08-28: the v-series does exactly
  this — verdict-level straddle 0/0/42/100/100% validated on GPU;
  `../near_miss_verdict_2026-08-28/FINDINGS.md`.)*
- The margin *hierarchy* of the checker is itself now measured on a
  controlled family: for uniform scaling, property checks < cross_shape ≪
  adaptive tolerance — the adaptive tolerance is nowhere near the binding
  constraint for this bug class, quantifying the "real headroom" note the
  adaptive_tol round made abstractly.

## Limits

- The family covers one error *shape* (uniform relative scaling). Bugs
  with input-dependent error profiles (the interesting ones for the
  adversarial variants) get margins that vary per variant — measured
  here only at the base input; extending the ladder per-variant is
  mechanical with the same method.
- 10 seeds per point: the response-curve percentages carry ±~15%
  binomial error; the shape (0 → 100% across 4× of margin, centered at
  1.0) is not in doubt.
- sum_reduction's wider CV means its m100 file is effectively an
  m0.99±0.25 instrument; use layernorm/gelu/l2norm (CV ≤ 9%) when tight
  margin control matters.

## Reproduce

```bash
.venv/bin/python probes/design_deltas.py       # measures rho, picks deltas
.venv/bin/python probes/generate_mutants.py    # writes TritonBench/near_miss/
# GPU: upload kcc9.tgz (incl. TritonBench/near_miss) + probes/near_miss_gpu.py
#      run with PYTHONPATH=/content; download /content/nm/near_miss_gpu.json
.venv/bin/python probes/analyze_near_miss.py
```
