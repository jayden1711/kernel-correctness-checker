# Layer-3 constant margins: the 0.9 factor lives in a (0.58, 0.998) dead zone with only 4 live records, the 3e-2/1e-3 variance asymmetry moves the validity domain rather than any verdict, and the non-round probe constants are structurally inert

**Measured 2026-08-28** from the banked current arms
(`layernorm_mask_fix_2026-08-28/arms/G_lnfix.json.gz`) plus CPU fp32
emulation where 6-decimal print precision truncates pass-side values.
Probe `probes/margins_l3.py`, log `data/margins_l3.log`. The question, per
constant: how far can it move before any outcome flips ("nearest miss"),
at both the check level and the verdict level.

## A. The 0.9 precision-coercion factor (matmul / flash / rmsnorm / softmax)

The check fails iff `err_fp32 > 1e-3 AND err_fp32 ≥ 0.9·err_fp16`. In the
entire corpus the factor is **live on only 4 of 52** recorded
precision-coercion invocations with an fp16 arm — the other 48 are gated
off by the atol arm (err_fp32 ≤ 1e-3), and flash_attention's 20 records
never test the factor at all (fp16 unsupported → auto-pass, with fp32
errors up to 3.0 recorded but unused; noted as a check blind spot, not a
margin).

Flip points r = err_fp32/err_fp16 of the live records:

| record | r | outcome |
|---|---|---|
| matmul/wrong_dtype | **0.5803** | pass (factor-protected) |
| rmsnorm/wrong_norm | **0.9984** | fail |
| rmsnorm/partial_reduction | 1.00003 | fail |
| matmul/swapped_strides | 1.00004 | fail |

**Dead zone: (0.5803, 0.9984).** The factor can rise 11% (to 0.998) before
losing its first check-level catch, or fall 36% (to 0.580) before creating
its first new one (which would be a *true* additional catch on the
wrong_dtype mutant, not an FP). Verdict-level margin is unbounded in the
tested range — every affected mutant is multiply caught. Two readings,
both honest: the constant is nowhere near a boundary, and the three
records it "catches" all sit at r ≈ 1.0 (the genuine coercion signature —
fp32 error equal to fp16 error), so any factor in (0.58, 0.99) behaves
identically; 0.9's specific value is doing no work that 0.95 or 0.7
wouldn't. The atol arm of the same check: nearest err_fp32 above 1e-3 is
3.8× (wrong_dtype, 0.0038), nearest below is 45× (2.2e-5) — a (÷45, ×3.8)
dead zone of its own.

## B. The unit-variance asymmetry (groupnorm/instancenorm 3e-2 vs layernorm 1e-3)

Measured deviations at corpus inputs: groupnorm ≤ 1.4e-5, instancenorm
≤ 1.6e-5 (banked, mutants and refs alike), layernorm ≤ 1.478e-5 and
rmsnorm-RMS ≤ 7.3e-6 (emulated, 10 seeds — pass records print no value).
Margins: layernorm **68×** at its 1e-3; groupnorm/instancenorm **~2000×**
at their 3e-2 — and they would sit at ~70× under layernorm's 1e-3 with
**zero outcome changes**. Nothing in the corpus motivates the asymmetry.

What the asymmetry *does* move is the **validity-domain boundary** from
the l3_validity round: unit-variance false-alarms on the correct operator
once input variance ≲ eps/atol, so 3e-2 tolerates in-domain inputs down to
var ≈ 3.3e-4 while 1e-3 needs var ≳ 1e-2 — a 30× wider input domain for
groupnorm/instancenorm. That is the only effect the constant has at
current margins; if harmonizing, harmonize deliberately against that
domain trade-off, not against corpus outcomes (which are indifferent).

A side observation the emulation surfaced: at corpus inputs the four
layernorm variants (ref + 3 mutants) produce **identical** unit-variance
deviations to 7 digits — every mutant is a correct normalizer at benign
inputs, so unit_variance is structurally mutant-blind there; its catch
value lives entirely in the adversarial-input variants, consistent with
the check-ablation tables.

## C. The non-round probe constants (4.2 l1/l2/frobenius, 3.1 instancenorm, 2.9 groupnorm)

Every mutant these positive-scale-invariance checks ever see is an exact
1-homogeneous normalization (x/S(x) with S 1-homogeneous — including
l1/partial_reduction, l2/wrong_norm, frobenius/wrong_norm,
instancenorm/skip_eps, groupnorm/ignore_affine), so
f(cx) = f(x) **identically** for every c > 0: the checks are
**structurally inert in c**. Verified by sweep: c ∈ {1.3, 2.9, 3.1, 4.2,
10, 100, 1000} changes the measured deviation by nothing above fp noise
(≤ 4.8e-7, i.e. ≥ 2000× inside atol = 1e-3) for every reference and every
mutant. There is no nearest miss — no value of c in [1.3, 10³] flips any
outcome, and none can until the corpus contains a homogeneity-breaking
mutant (an additive-bias bug would be the first customer). The
non-roundness of 4.2/3.1/2.9 is cosmetic.

The eps-domain caveat from the l3_validity round applies at the extremes:
for inputs near the eps boundary the c-dependence returns
(dev ∝ (ε/S)(1−1/c)); at corpus scales this is 10⁴ away.

## Summary table

| constant | nearest miss (check level) | verdict level |
|---|---|---|
| 0.9 pc factor | +11% / −36% (dead zone 0.58–0.998) | no flip in tested range |
| pc atol 1e-3 | ×3.8 above / ÷45 below | no flip |
| gn/inst uv 3e-2 | ÷2000 (to first new attribution) | no flip; domain shrinks 30× if set to 1e-3 |
| ln uv 1e-3 | ÷68 | no flip |
| c = 4.2 / 3.1 / 2.9 | none — structurally inert | none |

Every constant sits in a wide dead zone; none is calibrated, and at
current corpus margins none needs to be. As with the tolerance floor
(tol_floor round), the flat response is also the limitation: near-miss
mutants (item 7) are what would make these constants empirically
calibratable at all.

## Limits

- Banked print precision truncates at 1e-6; emulation fills layernorm/
  rmsnorm pass values, CPU fp32 vs GPU tree order noted (margins ≥ 41×,
  far beyond reduction-order noise).
- The 0.9 analysis covers records where the fp16 arm ran on this corpus;
  a future fp16-supporting flash kernel would activate its 20 dormant
  records.
- Emulated instancenorm normalizes per (N,C) over trailing dims — the
  reference's documented semantics; groupnorm's c-inertness is by the
  same 1-homogeneity argument, not separately emulated.

## Reproduce

```bash
.venv/bin/python probes/margins_l3.py
```
