# The 1e-6 tolerance floor: arbitrary but safe across nearly four decades — benign deviations at the floor are zero, the smallest floor-guarded catch is 6.4e-3, and no verdict anywhere moves for any floor in [0, 0.1]

**Analyzed 2026-08-28** by replaying every banked perturbation-family
verdict under a moving floor (`probes/floor_sensitivity.py`, log in
`data/`). Data: the current arms (`layernorm_mask_fix_2026-08-28`,
A and G — identical results on both), 854 perturbation-family records each,
every one carrying `max_err` and `adaptive_tol` at 6-decimal print
precision. The floor is `adaptive_tol = max(3·P95(s), 1e-6)`
(perturbation.py:210); no derivation or sensitivity analysis existed for
the constant before this round.

## 1. Who the floor binds

**126 of 854 records** print `adaptive_tol = 0.000001` in the current arms
(print precision cannot distinguish clamped from unclamped-just-above, so
this is an upper bound; the 2026-08-25 round, with full-precision access,
counted 98/854 — both numbers are the same population plus/minus the print
ambiguity and the variant-set drift between rounds). By op: argmax 6,
argmin 1, flash_attention 44, gelu 6, groupnorm 6, instancenorm 6,
log_softmax 6, rmsnorm 15, softmax 30, swish 6 — the J = 0 / degenerate-
response population (index-valued outputs, saturated softmax tails,
near-zero-variance and near-global-min adversarial variants), as the
2026-08-25 theory predicted.

## 2. The margin structure at the floor — measured, and bimodal

On the 126 floor-bound records:

- **Every benign record is at zero.** All 116 reference records AND all
  floor-bound passing-mutant records print `max_err = 0.000000`
  (≤ 5e-7 at print precision; exactly 0 for the exact-integer argmax/argmin
  outputs). There is no benign deviation anywhere near the floor.
- **Seven records are floor-guarded catches** — checks whose tolerance IS
  the floor (unclamped 3·P95(s) ≈ 0) and whose mutant error is macroscopic:
  gelu/near_global_min **0.0064**, swish/near_global_min 0.053,
  flash/equal_attention_weights 0.32, groupnorm/near_zero_variance 1.19,
  instancenorm/near_zero_variance 2.76, flash/skip_rescaling 3.11,
  flash/drop_last_tile-skip_rescaling 3.21.
- The smallest **unclamped** tolerance anywhere in the corpus is 1.4e-5 —
  a 14× gap above the floor. Below that, moving the floor touches only the
  clamped population.

The distribution of floor-bound max_err is exactly bimodal:
{0} ∪ [0.0064, 3.21]. Nothing lives between 5e-7 and 6.4e-3.

## 3. Sensitivity: verdicts, catch, FP, taxonomy vs floor

Replayed at F ∈ {0, 1e-9, …, 1e-1} with the unknown unclamped values
bracketed by u = 0 and u = 1e-6 (brackets agree everywhere):

- **Catch stays 40/40, FP stays 0/200, and zero (entry, trial)-level
  verdicts flip at every tested floor from 0 to 0.1**, both arms. The
  first *check-level* attribution change appears at F > 6.4e-3 (the gelu
  floor-guarded catch flips to pass); the first at the *verdict* level
  never appears in the tested range because every affected mutant is
  multiply caught.
- **Exception category 1 is irreducible by floor choice.** Its size stays
  126 for every F ≤ 3e-6 including F = 0 — the clamped population has
  u ≈ 0, so it binds at ANY positive floor; lowering the floor cannot
  shrink the exception class, because the class is a property of the
  operators (J = 0), not of the constant. It grows only once F reaches
  ordinary tolerances: 127 at 1e-4, 189 at 1e-3, 368 at 1e-2, 609 at 1e-1.

## 4. What the floor should be — the derivation

The floor's job is to separate benign candidate-vs-reference deviation on
degenerate-response records (measured: 0) from real-bug deviation
(measured: ≥ 6.4e-3). Two principled candidates:

1. **Relative form**: F = 32·ulp(‖f‖∞) — the scope round's validated
   resolvability threshold. At the corpus's O(1) output scales this is
   ≈ 3.8e-6; the shipped 1e-6 is **8.4 ulp at unit output scale**, i.e.
   the absolute constant is the relative form with an implicit
   unit-output-scale assumption that holds corpus-wide (worst case:
   argmax indices up to 2048 are exact integers with benign deviation
   exactly 0 and mutant signal ≥ 1, so any F ∈ (0, 1) is correct there).
2. **Measured safe interval**: F ∈ (5e-7, 6.4e-3) preserves every
   check-level outcome; F ∈ [0, 0.1] preserves every verdict.

**Verdict: 1e-6 is arbitrary but safe** — it sits ~2× above the print-
precision ceiling on benign deviation and 6400× below the smallest
floor-guarded catch. There is no derivation that singles out 1e-6, and
none is needed at current margins; if the checker ever runs on operators
with output scale far from 1, the relative form (1) is the correct
generalization and is already validated by the s/ulp = 32 work.

## 5. The connection to item 7 (stated, not manufactured)

The flatness IS the finding, and it is also the limitation: this corpus
cannot discriminate between floors anywhere in (5e-7, 6.4e-3) because
nothing in it fails near a boundary — the same flat-response-surface
property the near-miss-mutant round (item 7 of this pass) exists to fix.
Any future floor experiment needs those mutants first; re-running this
sweep after they exist is the designed follow-up.

## Limits

- Print precision truncates max_err/adaptive_tol at 1e-6; "benign = 0"
  means ≤ 5e-7, and the 126-vs-98 floor-bound count carries the same
  ambiguity (both stated in §1–§2).
- The replay recomposes verdicts from banked per-check outcomes; it
  assumes floor changes don't alter which checks *run* (true: the floor
  enters only the pass/fail comparison, and layer short-circuits are
  driven by earlier layers whose outcomes are unchanged — verified by the
  zero-flip result itself).
- Category-1 sizes for F < 1e-6 use u = 0 for clamped records (exact for
  the J=0 ops, upper bound otherwise).

## Reproduce

```bash
.venv/bin/python probes/floor_sensitivity.py
```
