# Claims-affected sweep over the 42 reference-suspect verdicts — corpus/theory claims clean; one reported claim's denominators are exactly the flagged sets; priority order revised

**Swept 2026-08-27, flag-only** — same discipline as the attention blast-radius
check: this determines *which reported numbers depend on these records*, and
deliberately does **not** adjudicate whether any flag is a real reference bug
or a check false-alarm (that remains the FINDINGS §4 follow-up). Record
extraction in `data/flagged42_detail.log`.

---

## Verdict up front

| claim family | overlap with the 42? |
|---|---|
| Theory rounds — sandwich coverage table, closed-form L, M3 fit (gpu_native / gen_native / replay / phase 1–2) | **None.** All use registry corpus inputs + spec batteries with ordinary fills; no flagged configuration (constant / near-zero-variance fills, shifts 100–1000, ±1e3–1e4 patches, γ≠1 coercion probes) appears in any of them. |
| AutoKernel comparison, corpus catch/FP tables, n_samples / scope / timing arms, latency work | **None.** Reference trials passed everywhere (0 FP) including at layernorm (512,512) — the one flagged *shape* that is also a spec shape; the flagged failures require the extreme search fills, which no corpus battery uses. |
| BUG_CLASS_THEORY **120/120** + "0 falsifying cases" | **None.** Its scope is the causal_flash_attention run's 120 proposals (BUG_CLASS_THEORY.md:55); none of the 42 is in that set. |
| BUG_CLASS_THEORY §4 — *"rmsnorm hit on 2 of 2 valid proposals, layernorm on 1 of 1, instancenorm on 3 of 3"* | **DIRECT — the strongest dependency found.** The "valid" denominators are precisely the complements of the flagged sets: instancenorm 18 proposals − **15 flagged** = 3; layernorm 12 − **11 flagged** = 1; rmsnorm 4 − **2 flagged** = 2. The claim's arithmetic *is* the validity labeling this sweep questions. |
| `RANGE_LIMIT = 300` derivation (*"exact and wide separation over the 48 softmax proposals"*) | **Yes, contained.** The 12 flagged softmax records sit inside the fitted negative class (all carry ±1e3–1e4 patches → range ≥ 1000, verified). The claim is already scoped in-repo as softmax-only and non-transferring. |
| Search-efficiency table (mean 13 proposals-to-hit) + context-effect (10.8 vs 20.0) | **Second-order only.** Flagged proposals are counted as spent; the recorded numbers stand under any adjudication — only a flash-style counterfactual (a *real* reference bug changing the trajectory) could shift interpretation. |

## 1. The flagged shapes/configs, per operator

| operator | n | shapes | configuration pattern |
|---|--:|---|---|
| instancenorm | 15 | (4,8,16), (4,8,16,16), (8,16,32), (8,16,64) | **all** constant (`ones×{1,5}`) or near-zero-variance (`randn×1e-4…1e-6` + shift) — variance ≈ 0, the documented eps-domain regime (`run_random_baseline.py:34-51`) |
| layernorm | 11 | (512,512)×8, (256,512)×2, (512,333), (64,64) | shifts 100–1000 with small scale (cancellation), γ=2/β=3 `precision_coercion` probes, ones+patches |
| softmax | 12 | (512,2048)×10, (512,333), (512,777) | zeros/randn + one ±1e3–1e4 patch; all fail `tile_coverage_softmax_positivity` |
| matmul | 2 | (64,256)@(256,32), (65,65)@(65,65) | **plain randn, scale 1 — ordinary in-domain inputs**, `scalar_associativity` |
| rmsnorm | 2 | (512,512)×2 | one `randn×1e-8` (eps-dominated), one ordinary randn with γ=2 (`precision_coercion`) |

## 2. Why the corpus/theory side is clean

Every theory and benchmark claim draws inputs from the registry corpus
(`default_rng(0)` ordinary fills) or the spec batteries; the sandwich/M3/L
tables never touch these five operators outside those inputs. Where a flagged
*shape* coincides with a spec shape (layernorm (512,512)), the corpus record
is a reference-vs-itself pass — counterfactual-invariant, and 0 FPs were
recorded everywhere. The 42 records exist only inside the 2026-07-23 search
runs, so only search-derived claims can depend on them.

## 3. The one direct dependency, stated precisely

BUG_CLASS_THEORY §4's quick-hit prediction — *identity-parameter operators
hit almost immediately* — is evidenced by hit-rates **per valid proposal**,
and "valid" is exactly the label the 42 flags challenge:

- If the flags are check false-alarms on genuinely degenerate inputs (the
  natural reading for instancenorm's constant inputs — a variance-zero
  instance makes the operator's own invariant ill-posed), the labels are
  right and the claim stands as written.
- If any were real reference bugs (the flash precedent), "3 of 3" drifts
  toward "3 of 18", and the claim's support changes materially.

No other reported number has this property: everything else either excludes
reference-failed records by construction (hit analyses, masking-class
derivations, matmul_prediction) or counts them in ways adjudication cannot
change (proposals-to-hit denominators).

## 4. Revised priority for the deferred adjudication

1. **instancenorm + layernorm + rmsnorm, together** (28 of the 42): one
   adjudication pass over the shared mechanism class (variance/eps domain,
   cancellation, `precision_coercion` on extreme inputs) resolves all three
   denominators of the §4 claim at once — the only direct reported-claim
   dependency. Likely quick: the mechanism is already documented for the
   norm family.
   > **DONE 2026-08-27 — `NORM_ADJUDICATION.md`.** 27/28 confirmed
   > check-domain false alarms; instancenorm 3/3 and rmsnorm 2/2 verified;
   > layernorm corrected to "1 of ≥4" because one flagged record
   > (`f322abe4`, (512,333)) is a **second real padded-lane reference bug**
   > (layernorm's unmasked variance lanes) — flagged for its own
   > investigation, not fixed.
2. **softmax** (12): inside the RANGE_LIMIT fit's negative class; the claim
   is already scoped softmax-only, so stakes are low, but the fit's
   interpretation ("failed ⟺ extreme-range input") should not be reused
   before these are adjudicated.
3. **matmul (65,65) and (64,256)@(256,32)** — **demoted from the previous
   "first look" recommendation.** They remain the most bug-like on their
   face (ordinary inputs failing an algebraic identity; plausibly the
   documented pre-loosening `scalar_associativity` tolerance FP,
   `matmul_properties.py:78`), but **no reported claim depends on them**, and
   this sweep's criterion is claim dependency, not volatility.

The impact round's earlier suggestion (matmul (65,65) / softmax (512,333)
first, on shape-class grounds) is superseded by this ordering.

## 5. Reproduce

```bash
.venv/bin/python verification_runs/attention_mask_fix_2026-08-27/probes/flagged42_extract.py
grep -n "120 proposals\|3 of 3\|48 softmax" benchmarks/BUG_CLASS_THEORY.md
grep -n "proposals-to-hit" BENCHMARK_RESULTS.md adversarial_results/CFA_NONHIT_ROOTCAUSE.md
```

Limits: the "valid = total − flagged" arithmetic assumes no *other*
reference-failed records for these operators — checked: the review sweep
(`data/review_before_relabel.log`) shows every reference-fail for these five
operators classifies invariant-kind, and the reconstruction reproduces the
§4 claim's denominators exactly (18−15=3, 12−11=1, 4−2=2). Patch values for
the softmax range check were read from the four representative proposals
shown in the sweep log; all carry ±1e3–1e4 patches.
