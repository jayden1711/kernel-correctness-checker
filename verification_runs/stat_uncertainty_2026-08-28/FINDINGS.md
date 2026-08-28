# Uncertainty on two reported statistics — 2026-08-28

**Item:** the unaddressed half of theory-audit flag #6: (A) the 1.9×
context-effect claim (CFA_NONHIT_ROOTCAUSE.md §6, n=9 operators, no CI) and
(B) the leakage-ablation counts 112/120 and 83/120 (BUG_CLASS_THEORY.md),
used comparatively without error bars. Probe:
`probes/uncertainty.py` (exact permutation + seeded bootstrap + Wilson +
exact McNemar; offline, deterministic).

## A. Context effect — claim SURVIVES directionally; point estimate CORRECTED

**The published 10.8 context-group mean is not reproducible.** Canonical
banked artifacts (`adversarial_results/*_search_result.json`) give
{softmax 12, layernorm 12, matmul 4, flash_attention 12, rmsnorm 4} → mean
**8.8**. The history DB's per-run totals allow at most 11.2 (using the first
of softmax's four runs, 23, and layernorm's 13). No natural combination of
banked numbers produces 10.8. The no-context mean 20.0 reproduces exactly
(gelu 18, instancenorm 18, argmax 24; causal_flash_attention censored at
≥120, excluded — consistent with the doc's "1 non-hit" framing).

With banked numbers the ratio is **2.27×**, not 1.9×. Uncertainty:

- Exact permutation test (are the 3 no-context totals unusually high under
  label exchangeability?): the observed split is the single most extreme of
  C(8,3)=56 → one-sided **p = 0.018**. Including the censored operator at
  its lower bound 120 (conservative): 1/126 → **p = 0.008**.
- Bootstrap 95% CI for the ratio of means: **[1.67, 3.93]**; 0 of 10⁵
  resamples put the ratio ≤ 1.

**Verdict:** the comparative claim (context reduces proposals-to-hit)
survives, and is in fact understated by the erroneous 10.8. The magnitude is
loose — the honest statement is "a factor of ~2, anywhere in ~1.7–3.9" — and
the design is observational (operators were not randomized to groups; the
doc itself notes the no-context group skews toward harder conventions).
Corrected in place: dated annotation added to CFA_NONHIT_ROOTCAUSE.md §6.

## B. Leakage ablation — both comparative claims SURVIVE; one sharpened

Reproduced the per-proposal records with `bug_class_theory.py`'s own
simulation (asserted equal to the banked 120/120, 112/120, 83/120 before
computing anything).

- **Offline accuracy 112/120 = 93.3%, Wilson 95% CI [87.4%, 96.6%]**;
  operator-level cluster bootstrap widens it to ≈[80%, 99%] (proposals
  cluster in 6 operators; softmax alone is 48/120).
- **Degradation vs the recorded-validity predictor (120/120) is real, not
  noise**: on the paired 120 items all 8 disagreements fall the same
  direction; exact McNemar two-sided **p = 0.0078**.
- **Term agreement 83/120 = 69.2%, Wilson CI [60.4%, 76.7%]** — but the
  aggregate hides the structure: on softmax, where the range<300 rule was
  fitted, agreement is 46/48 = 95.8%; on the other five operators it is
  **37/72 = 51.4%, CI [40.1%, 62.6%] — statistically indistinguishable from
  a coin flip**. The doc's "does not transfer" warning was, if anything, too
  mild; the rule transfers *no information at all* off-softmax.

Corrected in place — the right way for a generated doc: BUG_CLASS_THEORY.md
carries "do not hand-edit — re-run it", so the uncertainty paragraph was
added to **`bug_class_theory.py`'s report generator** and the doc
regenerated. Two side-findings from doing that:

1. **A prior hand-edit had violated the doc's own contract and was silently
   destroyed by regeneration**: the 2026-08-27 NORM_ADJUDICATION block
   (instancenorm 3/3 / rmsnorm 2/2 confirmed; layernorm "1 of 1" → "1 of
   ≥4") existed only in the .md. It is now embedded in the generator (with a
   dated since-FIXED bracket for the layernorm kernel bug) and survives
   re-runs. `data/BUG_CLASS_THEORY.pre.md` preserves the pre-round file; the
   final diff is exactly {uncertainty paragraph, restored block + fix
   bracket}.
2. The generator's controls all still fire after regeneration (A/B/C/D
   degrade the score as required).

## Files

- `probes/uncertainty.py` — all computations, seeded.
- `data/uncertainty_results.json` — machine-readable results.
- `data/BUG_CLASS_THEORY.pre.md` — pre-round snapshot of the generated doc.
- Edits: `adversarial_results/CFA_NONHIT_ROOTCAUSE.md` (dated annotation),
  `benchmarks/bug_class_theory.py` (uncertainty emission + adjudication
  block), `benchmarks/BUG_CLASS_THEORY.md` (regenerated).
