# OOB harness fix: shipped, all four regression criteria met, and the diagnosis's falsifiable prediction held — rmsnorm's collapse was the out-of-bounds content, and it is gone

**Shipped and regressed 2026-08-28** on a Colab T4 (session `kccfix`,
provisioned and stopped; `torch 2.11.0+cu128`, `triton 3.6.0`). Arms in
`arms/`, driver `oobfix.sh`, scoring `analysis/validate_fix.py` (output
banked with follow-ups in `analysis/out_validate_fix.txt`). This executes
the fix specified in `../oob_adjudication_2026-08-28/FINDINGS.md` §5 against
its four regression criteria.

## What shipped

1. **`verification/specs/layernorm.py`, `rmsnorm.py`** — `_non_power_of_two`
   is now width-adaptive (`w = 333` when the base width allows it, else the
   largest non-power-of-two width ≤ the base width; 127 at the corpus's
   width-128 shapes), and `get_adversarial_inputs` slices the captured
   companions to each variant's own width, so every returned tuple satisfies
   the spec's stated contract. **Draw-then-slice, deliberately**: the old
   `(rows, 333)` draw is kept and sliced, so the per-check-reseeded torch
   stream is consumed exactly as before and variants generated after
   `non_power_of_two` (rmsnorm's `constant_rows`, `large_variance`) receive
   bit-identical draws. At base widths ≥ 333 the variant is bitwise
   identical to its pre-fix behaviour.
2. **`TritonBench/reference/layernorm.py`, `rmsnorm.py`** — wrapper shape
   asserts: a companion shorter than `n_cols` now raises `ValueError` before
   any launch (routing into the `reference_failure_kind` machinery) instead
   of silently reading past its allocation.
3. **`tests/verification/specs/test_spec_contracts.py`** — the systemic
   contract test: every spec's `get_adversarial_inputs` executed at every
   entry of its own `valid_shapes` AND at the autokernel registry's corpus
   shapes, checked by three rules (R1 per-feature companions track the
   adversarial primary's width; R2 mask-like companions track its shape,
   applied where no executable math definition exists; R3 the `math_refs`
   float64 definition must run on every adversarial tuple). Includes a
   **negative control** that reconstructs the exact pre-fix tuple and
   asserts the contract check rejects it, unit pins for the width table,
   and a draw-for-draw stream-preservation pin. Coverage floors are pinned
   (≥100 (spec, shape) cases, ≥40 R1 and ≥200 R3 applications) so silent
   erosion fails the test. **309 passed locally**; full verification suite
   602 passed with only the pre-existing, unrelated
   `adversarial_search/test_worker_parsing` failure (documented in the two
   prior rounds).

## The four regression criteria

| criterion | result |
|---|---|
| **C1** verdicts byte-identical to pre-fix arms | **PASS.** A_fix vs banked `A_no_detector` and G_fix vs banked `G_gram`: every check outcome identical; 40/40 catch, 0/200 FP on both arms. |
| **C2** collapse resolved + Gram screen evaluates the classes, silent | **PASS, decisively.** layernorm: 16/16 distinct record fingerprints (was 16 → 1 sulp-fallback pre-fix); rmsnorm: **15/15 distinct (was 15 records → 1)**. `gram_n_valid = 20` on every one of the 31 records (was 0 — fail-open); zero Gram fires; worst \|log10 r\| = **4.3e-5** (ratio 1.00010) layernorm, 2.9e-5 rmsnorm — deep inside the in-scope band, s/ulp 7.2k–17.9k. |
| **C3** contract test passes at every valid shape | **PASS** (309 tests, incl. the (2048, 128) latent-trigger shape and the registry shapes; negative control confirms the pre-fix construction is rejected). |
| **C4** exactly zero catch-attribution changes | **PASS.** All 40 mutant detail strings identical to the pre-fix G arm. |

**The falsifiable diagnosis check.** The adjudication predicted rmsnorm's
bit-identical banked records were OOB-content-driven (stable adjacent memory
dominating the response), so the fix must dissolve the collapse. It did:
15/15 distinct post-fix, varying through the sliced captured gamma exactly
as the varying-companion mechanism requires. Had the records stayed
bit-identical, the diagnosis would have been wrong. It held.

**Wrapper asserts, exercised on GPU** (`arms/wrapper_assert.log`): matched
companions run on both wrappers; all three mismatch cases (short gamma,
short beta, rmsnorm short gamma) raise `ValueError` — `WRAPPER-ASSERTS-OK`.

## The stronger-than-C1 check, and its one adjudicated exception

Because the fix is draw-then-slice, every scope record **outside** the two
fixed classes should be *bit-identical* to the pre-fix G arm — a much
stronger property than verdict identity. Measured: all 31 fixed-class
records changed (expected), and 10 records outside them differed — **all
ten on the `frobenius_norm/wrong_norm` entry**, the `tl.atomic_add`
reference whose run-to-run nondeterminism is already documented (the
determinism-flake note in RESULTS_SUMMARY). Cross-check: the same records
differ 4/10 between the two *pre-fix* rounds (scope D 2026-08-26 vs gram G
2026-08-27, same seeds, no relevant code change), i.e. the same instability
without any fix in play; the matmul control (154 shared records) is
0-different across all three arms. The bit-identity guarantee therefore
holds on every deterministic-kernel record; nothing was stream-shifted.

## Bonus consistency note

Post-fix, layernorm's variant runs at width 127 with BLOCK = 128 — one
unmasked pad lane, so the known (still-unfixed) variance bug contributes
only `mean²/127` ≈ 6e-5 relative variance error at randn inputs, consistent
with the measured ratio 1.00010. The variant now *also* exercises the
non-pow2 tile boundary at a width where that separate bug is negligible —
previously it exercised it at width 333 where the pad term is 179 lanes.
When the layernorm variance bug gets its own fix round, its regression
should not expect movement from this variant.

## Limits

- The regression is against the gram-round arms (the adjudication's
  reference point). The older banked rounds (n_samples, scope, check_timing)
  retain their pre-fix records with the annotations already in place; they
  are historical artifacts and were not re-run.
- The contract test's R1/R2 rules are heuristics scoped by companion
  structure; specs whose companions are scalars or 2-D operands are covered
  by R3 only where a math definition exists (coverage floors pinned, resting
  on the 27-op `math_refs` registry).
- `masked_cumsum`'s R2 coverage is vacuous today (its variants are
  shape-preserving); the rule exists for the first spec that adds a
  shape-changing variant with a mask companion.
- The frobenius nondeterminism adjudication above rests on the documented
  atomic-add mechanism plus the cross-round differential; the kernel was not
  instrumented further here.

## Reproduce

```bash
# local:
.venv/bin/python -m pytest tests/verification/specs/test_spec_contracts.py -q
.venv/bin/python -m pytest tests/verification -q

# GPU (T4):
export HOME=~/.colab-home && colab new --gpu T4 -s kccfix
tar --exclude='__pycache__' --exclude='.venv' -czf /tmp/kcc7.tgz \
    verification benchmarks scripts tests TritonBench
# upload kcc7.tgz, probe_redundancy.py, probes/wrapper_assert_probe.py, oobfix.sh
# nohup bash /content/oobfix.sh; poll /content/oobfix/DONE; download /content/probe/*.json
colab stop -s kccfix

python3 analysis/validate_fix.py arms/A_fix.json.gz arms/G_fix.json.gz
```
