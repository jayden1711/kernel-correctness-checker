# The binding-check law: every flip-δ in the verdict-boundary table is a closed form — 34/34 banked entries reproduced at ratio p5/p50/p95 = 0.97/1.00/1.06, binding check predicted 5/5, and a pre-registered blind test on two unseen operators lands 2/2 bindings with 14/16 entries at ratio ≈ 1.00

**Derived and validated 2026-08-28, CPU only.** Probe
`probes/predict_bindings.py`, logs and prediction/truth JSONs in `data/`.
This answers the round's lead question — *is there a general law predicting
which check binds for a given operator, from its structure rather than by
bisection?* — in the affirmative, with the law stated below, and it answers
the companion lead (*does the Gram/parent structure predict the binding
boundary?*) with a precise partial: the parent predicts exactly one column
of the table, and whether that column binds is itself derivable (§3).

## 1. The law

For a uniform (1+δ) output scaling — the near-miss family's bug shape —
every check in the pipeline compares a pair (a, b) via an
allclose-comparator with constants (atol, rtol). Under scaling, the
deviation is linear (or exactly polynomial) in δ with a **derived
velocity**, and the flip-δ is

    δ*_check = min over compared elements i of
               (atol + rtol·|b_i| − |e0_i|) / (k·|s_i|)

where s_i is the compared statistic, e0 = a−b the baseline residual at
δ = 0, and k the **scaling degree** of the statistic, read off the
comparator's structure:

| comparator class | k | consequence |
|---|---|---|
| output or degree-1 statistic vs fixed target (row sums, norms, RMS, affine expectation, torch-reference comparisons: cross_shape, weight_magnitude, precision_coercion's atol arm) | 1 | δ* = rtol + atol/max\|s\| (minus baseline-residual room) |
| variance vs 1 (unit_variance) | 2 | δ* = (atol + rtol)/2 to first order; solved exactly as a quadratic |
| both sides computed from the same scaled candidate (every *_invariance, *_equivariance, monotonicity, permutation, scale_linearity, zero_at_origin, **gamma_correctness**) | 0 | **derived inert** — the check cannot see a uniform scaling, by construction |
| perturbation family (base + adversarial variants) | 1 | δ* = max(3·σ·L·E[q95_n], 1e-6)/max\|f_variant\| — the threshold is the **structural parent's** closed-form order-statistic mean (§3) |
| precision_coercion with an fp16 arm (softmax, rmsnorm, matmul) | 1 | two-arm crossing of errP(δ) = max_i \|e0P_i + δ·fP_i\|, solved on the derived model |

Inputs to the law: the comparator constants (including **torch.allclose's
silent default rtol = 1e-5**, see §4), ONE forward evaluation of the
reference per check input (for max|s| and e0), and the parent integral. No
bisection anywhere.

## 2. Validation against the banked bisection tables

Scored against `../near_miss_verdict_2026-08-28/data/design_verdict.json`
(the v-series design tables, found by bisection through the shipped check
functions). All five ops, every entry:

- **Finite entries: 34/34 within the 5-seed draw spread** — ratio
  pred/truth p5/p50/p95 = **0.97/1.00/1.06**. Deterministic-comparator
  entries agree to 3–4 significant figures (e.g. softmax rows_sum_to_one
  1.099e-4 both; layernorm unit_variance 5.086e-4 both; every cross_shape
  and weight_magnitude entry at ratio 1.00).
- **Inert entries: every one derived, none measured-and-missed** — the k=0
  class plus the tiny-statistic cases (zero_mean at max\|means\| ~ 1e-7).
- **Binding check: 5/5**, including softmax's lockstep pair
  (max_in_last_tile + extreme_range: shared floor and shared peak ≈ 1 give
  the same δ* by the law — lockstep is a *prediction*, not an observation).
- The parent-composed perturbation column lands at 0.96–1.07 of truth on
  every variant — including softmax/adversarial_equal_logits (1.325e-2
  pred vs 1.316e-2), where the entire number is parent-generated
  (uniform-p Jacobian, m = 8192 profile, E[q95_20] integral).

## 3. What the parent/Gram structure predicts — the companion lead, answered

The perturbation column of the table is **exactly** the structural parent
pushed through the checker's own order statistic:
δ* = max(3σL·E[q95_20], 1e-6)/max|f|. Validated per-entry above and
absolutely against the native bank (R²(log) = 0.997, median ratio 1.006,
`../direct_tol_2026-08-28/`). So:

- **When the binding check is perturbation-family, the parent predicts the
  verdict boundary outright.** On the seven ops now measured this happens
  two ways: the parent's q95 falls below the floor and the boundary is
  1e-6/max|f| (softmax, gelu — the *floor* flavour: the parent's
  J-degenerate directions decide that the floor takes over), or the parent
  q95 itself is the boundary (frobenius_norm's dominant_outlier variant,
  predicted 1.494e-4 vs truth 1.491e-4 — a fully parent-generated binding).
- **When it is not** (layernorm's affine, l2norm/sum/rmsnorm's
  cross_shape), the parent still predicts its own column, and the *reason
  it does not bind* is derivable: 3·P95 adaptive tolerances are 4–3400×
  looser than the tightest deterministic comparator, so the binding goes to
  whichever fixed-constant check has the smallest effective relative
  tolerance ρ_eff = rtol + atol/max|s|.

One sentence version: **argmin over checks of ρ_eff decides the binding;
the parent supplies ρ_eff for the stochastic column; constants and one
forward evaluation supply the rest.**

## 4. The blind test (pre-registered)

Predictions for **rmsnorm** and **frobenius_norm** — neither in the
v-series, tables never bisected before — were written to
`data/blind_predictions.json` *before* the ground-truth bisection ran
(same probe machinery as the v-series design, REFS extended with the two
fp32 emulations; `data/blind_truth.json`).

- **Binding check 2/2**: rmsnorm → cross_shape (pred 1.213e-4, truth
  1.213e-4); frobenius_norm → adversarial_dominant_outlier (pred 1.494e-4,
  truth 1.491e-4).
- **14 of 16 finite/inert entries agree at ratio p50 = 1.00**, including a
  structural prediction with no analogue in the validated five:
  **rmsnorm's gamma_correctness is derived inert** (unlike layernorm's
  affine check it compares candidate-to-candidate, k = 0) — and the
  bisection confirms it never fires. The law thus predicted a qualitative
  binding-structure difference between layernorm and rmsnorm before
  measurement: layernorm binds at 1.9e-5 on its affine check, rmsnorm has
  no such check to bind on and sits 6× looser at cross_shape.
- **The two misses, adjudicated rather than averaged away:**
  1. `precision_coercion` pred 1.374e-3 vs truth 2.612e-3 (ratio 0.53,
     non-binding, 21× above the binding boundary). Identified mechanism:
     the model treats the (1+δ) scaling of the fp16 arm as exact, but the
     scaled emulation multiplies *in half precision*, quantizing the
     effective δ to fp16 ulps (≈ 9.8e-4 steps at out ≈ 1) — the two-arm
     crossing model needs the fp16-rounded δ_eff, which was not modeled.
  2. `adversarial_constant_rows` pred "inert in range" vs truth 2.56e-1.
     Not a law failure but a range-ceiling artifact on a heavy-tailed
     statistic: the per-seed predictions are 0.32/0.63/1.76/**0.258**/0.86
     (driven by the min row magnitude of the constant-rows draw), the
     probe's bisection ceiling is 0.3, and the seed the truth found flips
     at 0.256 — **0.8% from that seed's prediction**. The medians land on
     opposite sides of an arbitrary ceiling.

## 5. What the law says about the checker (three structural corollaries)

1. **layernorm's verdict boundary is set by an unset default.** Its binding
   δ* = 1.9e-5 comes from `torch.allclose`'s implicit rtol = 1e-5 in
   `check_affine_correctness` — no one chose that constant; every
   explicitly-chosen constant in the battery is ≥ 1e-4. Anyone re-tuning
   layernorm's sensitivity should know the knob is a library default.
2. **Floor-bound bindings are a joint property of the parent and the output
   scale**: they occur exactly where the parent's q95 < 1e-6/3 while some
   variant's output peak is O(1). This is computable in advance for any new
   operator — the law says softmax-like (saturating, peak-1) operators will
   always bind at ≈ 1e-6/1, and it did so for gelu's near-global-min
   variant within its floor-adjacent draw wobble.
3. **The uniform-scaling bug shape can be ranked per op without a GPU**: the
   full table costs one CPU forward evaluation per check input plus a ~1 ms
   parent integral per perturbation variant — the v-series design step
   (bisection through every check, 5 seeds) is no longer the only way to
   build near-miss families for new operators.

## Limits

- **One error shape.** Everything here is for uniform relative scaling —
  the same scope limit the v-series carries. Input-dependent error shapes
  have different velocities v_i and need their own derivation (the k=0
  class in particular is only inert for *uniform* scaling).
- The property/cross-shape validations are against CPU fp32 emulations (the
  same instrument the v-series design used); GPU kernels shift e0 by fp
  noise, which matters only within ~e0/atol ≈ 0.1% of the boundary.
- The fp16 two-arm model needs the δ-quantization correction before its
  precision_coercion predictions are trusted below 2× (miss #1 above).
- max|s| and e0 are *measured* per check input (one forward evaluation) —
  the law is closed-form in the comparator structure, not input-free.
- Floor-adjacent perturbation entries (gelu near_global_min) inherit the
  resolvability-boundary caveat: predictions there are a band
  [1e-6/M, ~3e-6/M], not a point (taxonomy, tol_floor round).

## Reproduce

```bash
.venv/bin/python probes/predict_bindings.py   # validate + blind, ~10 min CPU
# data/validate_predictions.json, blind_predictions.json (written first),
# blind_truth.json (bisection ground truth), run2.log (full scoring)
```
