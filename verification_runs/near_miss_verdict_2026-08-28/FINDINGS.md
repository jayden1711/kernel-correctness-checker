# The verdict-level near-miss family (v-series): the whole-pipeline boundary is located per op, the 25 new mutants straddle it (0/0/42/100/100% verdict catch across the ladder), and nothing beyond the design-binding check fires — 100 sub-boundary runs, zero catches by any other check

**Designed, built and GPU-validated 2026-08-28** (T4 session `nmv`,
stopped). Design `probes/design_verdict_deltas.py`, generator
`probes/generate_v_mutants.py`, GPU probe `probes/v_series_gpu.py`, raw
results + analysis in `data/`. This closes the m-series' stated gap
(`../near_miss_2026-08-28/FINDINGS.md`): those mutants straddle the
adaptive-tolerance boundary but are verdict-caught by tighter checks, so
they were a check-level instrument only.

## Design: the binding check, found through the shipped code

For a uniform (1+δ) scaling, the verdict flips at
δ*_verdict = min over every check in the pipeline of that check's flip-δ.
Rather than an analytic table, δ* was found by **bisection through the
shipped check functions themselves** (property checks, `_check_cross_shape`,
`check_weight_magnitude`, `check_perturbation_tolerance` on the base input
and every spec adversarial variant, floors included), run on CPU with fp32
emulations of the reference kernels in the specs' own signatures, 5 RNG
draws per stochastic check. The full per-check tables are in
`data/design_verdict.json`; the binding structure:

| op | binding check | δ* | second check | gap |
|---|---|---|---|---|
| layernorm | `affine_correctness` (L2) | 1.94e-5 | cross_shape | 6.3× |
| softmax | `adversarial_max_in_last_tile` (floor-bound, tol=1e-6, peak output ≈1) | 1.01e-6 | `adversarial_extreme_range` — **lockstep, same δ*** | 1.0× |
| gelu | `adversarial_near_global_min` (floor-bound) | 8.76e-6 | cross_shape | 14× |
| l2norm | `cross_shape` | 3.41e-4 | unit_l2_norm | 3.0× |
| sum_reduction | `cross_shape` | 1.01e-4 | weight_magnitude | 9.9× |

Notable in itself: **for three of five ops the verdict boundary is not a
property check but a floor-bound adversarial variant or cross_shape** —
the binding constraint is 4–3400× tighter than the adaptive perturbation
tolerance, and for softmax/gelu it is the 1e-6 floor (tol_floor round)
seen from the mutant side. The affine_correctness δ* also has a clean
closed form (δ* = rtol + atol/max|2·norm+3| = 1.9e-5), matching the
bisection.

The v-series: `TritonBench/near_miss/<op>/v{050,080,100,125,200}.py`,
δ = m·δ*_binding, same mis-scaled-epilogue mechanism as the m-series.
Registry updated (`near_miss_corpus.py`, both series; still never merged
into the published corpus).

## GPU validation (full KernelChecker battery, 10 seeds per mutant)

**Verdict response curve, pooled: 0/50 → 0/50 → 21/50 → 50/50 → 50/50
(0% / 0% / 42% / 100% / 100%) at design margins 0.5 / 0.8 / 1.0 / 1.25 /
2.0.** The family straddles the verdict boundary, and the transition is
where it was designed to be.

**The "does anything tighten beyond it" question, answered: no.** In all
100 sub-boundary runs (v050 + v080, every op) the failing-check set is
**empty** — not one catch by any check anywhere in the battery. And in
every caught run the catching check is **exactly the design-predicted
binding check** (25/25 mutants; softmax caught by its predicted lockstep
pair, both variants together, 10/10 each).

Per-op realized margins and honest calibration notes:

| op | realized margin at v100 (min/med/max) | v100 catch | note |
|---|---|---|---|
| sum_reduction | 1.000 / 1.002 / 1.003 | 9/10 | knife-edge straddle, near-perfect calibration |
| layernorm | 0.966 / 0.991 / 1.020 | 3/10 | true stochastic straddle |
| l2norm | 0.950 / 0.982 / 1.025 | 2/10 | true stochastic straddle |
| softmax | 1.013 (identical all seeds) | 0/10 (v125: 10/10) | the variant input is deterministic, so the transition is a **step between mutants**, not a within-mutant straddle; boundary located to within 25% |
| gelu | 1.489 (identical all seeds) | 7/10 | design offset ~1.3–1.5×: the CPU-emulated variant response sits slightly above the 1e-6 floor while the GPU response floors harder, and the per-seed GPU tolerance itself wobbles around the floor (0/10 at realized 1.19, 7/10 at 1.49). The curve still transitions inside the ladder. |

Three of five ops give genuine within-mutant straddles (P(miss) a
measurable probability); softmax gives a sharp deterministic step; gelu
transitions with a known offset. Reported as measured — the gelu offset
is the one place the CPU design missed by more than a few percent, and
its mechanism (floor-adjacent tolerance draw-noise) is exactly the
resolvability-boundary behaviour the taxonomy predicts there.

## What this enables

- Verdict-level tolerance experiments: any change to cross_shape's
  atol/rtol, the affine/property atols, or the 1e-6 floor now moves real
  verdicts (the m-series already covers the adaptive-tolerance scale/
  delta_scale knobs at the check level).
- The binding-check table doubles as a measured margin hierarchy per op —
  the l3_margins round's dead-zone statements now have a mutant family
  sitting at the edges instead of 461× away.

## Limits

- One error shape (uniform relative scaling), 5 ops; binding checks for
  other bug shapes (input-dependent errors) can differ.
- gelu/softmax margins quoted from the probe's tolerance re-measurement at
  offset seeds; the battery's own per-seed tolerances differ near the
  floor (visible in gelu's 0-at-1.19 / 7-at-1.49 pattern).
- 10 seeds per point: response percentages carry ±~15% binomial error.
- The softmax lockstep pair means its two variants cannot be separated by
  this family (they flip together by construction of the shared floor).

## Reproduce

```bash
.venv/bin/python probes/design_verdict_deltas.py   # binding-check tables
.venv/bin/python probes/generate_v_mutants.py      # writes the v-series
# GPU: upload kcc10.tgz + probes/v_series_gpu.py, PYTHONPATH=/content,
#      download /content/nmv/v_series_gpu.json
# analysis one-liner banked in data/analysis.log
```
