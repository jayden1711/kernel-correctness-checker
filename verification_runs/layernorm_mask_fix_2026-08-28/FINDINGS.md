# Layernorm padded-lane variance fix: shipped, and the corpus diff is exactly the predicted single attribution change — nothing else moved

**Shipped and regressed 2026-08-28** on a Colab T4 (session `lnfix`,
provisioned and stopped; torch 2.11.0+cu128, triton 3.6.0). This executes
the fix flagged in `../attention_mask_fix_2026-08-27/NORM_ADJUDICATION.md`
§2 and blast-radius-scoped in `../layernorm_mask_bug_2026-08-27/FINDINGS.md`
§4, closing the padding/masking bug family (member 3 of 3, after
flash_attention and scaled_dot_product_attention on 2026-08-27).

## What shipped

**`TritonBench/reference/layernorm.py`** — one line, the
groupnorm/instancenorm pattern:

```
diff = tl.where(mask, row - mean, 0.0)     # was: diff = row - mean
```

Padded lanes (loaded as 0.0 when `BLOCK_SIZE = next_pow2(n_cols) > n_cols`)
no longer contribute `mean²` each to the variance sum. Live only at
non-power-of-two widths; bitwise-invisible everywhere else (measured, below).

## Regression, against the four criteria of layernorm_mask_bug §4

Arms `A_lnfix`/`G_lnfix` (same probe, corpus, and `KCC_ABLATION_SEED=1`
discipline as every prior round) scored against the **banked post-oob-fix
arms** `../oob_fix_2026-08-28/arms/{A_fix,G_fix}.json.gz` by
`analysis/validate_lnfix.py` (output banked in
`analysis/out_validate_lnfix.txt`). GPU-side probe
`probes/ln_gpu_probe.py`, log `data/ln_gpu_probe.log`.

| criterion | result |
|---|---|
| **(a)** corpus diff shows **exactly one** attribution change — not zero | **PASS, exactly as predicted.** Both arms: the single outcome flip is `layernorm/wrong_variance_estimate` `cross_shape` fail → pass; the single detail change is `[L3]cross_shape; [L3]adversarial_wrong_variance_trigger` → `[L3]adversarial_wrong_variance_trigger`. The mutant stays caught by its trigger. **40/40 catch, 0/200 FP** on both arms. (Had the diff been zero, the banked catch would not have been bug-manufactured and the 08-27 adjudication would have been wrong — the flip is required evidence, and it appeared.) |
| **(b)** cross_shape sub-outcomes at (1000,333) for a correct candidate | **PASS.** All five shapes pass post-fix, (1000,333) included. Margin measured directly (10 seeds): mutant-vs-fixed-reference max_err **7.2–9.5e-7**, 105× inside atol 1e-4 (the 08-27 emulation predicted 4.8–7.2e-7 — same order, CPU/GPU reduction-order gap); vs the buggy kernel the same seeds fail at **0.021–0.043**, bracketing the banked 0.0249. Fixed reference vs float64 ideal math: 5.5–7.7e-7. |
| **(c)** pow2 bitwise identity | **PASS, on GPU.** Buggy kernel (reproduced verbatim in the probe) vs fixed reference: `torch.equal` at all four pow2 spec shapes + corpus (64,128), with **randn gamma/beta** (not the trivial ones/zeros). Fix liveness confirmed at (1000,333): max abs diff 7.0e-2. |
| **(d)** CHECK_ABLATION numbers | **Done.** Both ablation docs' 2026-08-27 derived-correction notes upgraded to verified-measurement notes pointing here (`cross_shape` 28→27 main table, 29→28 in FINDINGS, (1000,333) sub-row 10→9). The full ablation tables themselves remain the pre-fix run, stated in the note. |

## The oob-round prediction, verified

`../oob_fix_2026-08-28/FINDINGS.md` predicted this fix would produce **no
movement on the width-127 `adversarial_non_power_of_two` variant** (one pad
lane, `mean²/127 ≈ 6e-5` relative variance error, negligible vs the
factor-2 scope band). Measured: all 16 layernorm non-pow2 records keep
their verdicts, zero Gram fires, and the worst |log10 ratio| moves
4.31e-5 → 3.65e-5 — a shift of the predicted ~e-5 order, deeper into the
noise floor. Prediction held.

## The stronger value-level check

Outside the fix's two legitimate surfaces, every scope record's VALUE
fields (gram ratios, s/ulp, adaptive_tol) are **bit-identical** to the
banked pre-fix G arm. The 23 changed records decompose exactly:
**16 layernorm `adversarial_non_power_of_two`** (the reference output
changes at width 127 — expected) and **7 `frobenius_norm/wrong_norm`**
(the documented `tl.atomic_add` run-to-run flake; prior rounds measured
10 and 4 on the same records with no code change). Zero unexplained.

## Local suite

602 passed, 1 skipped, and only the pre-existing, unrelated
`adversarial_search/test_worker_parsing.py::TestWorkerRetry::test_all_retries_exhausted_raises`
failure (documented in the two prior rounds). The 309 spec-contract tests
pass — the fix touches no draw and no spec, so the stream-preservation pins
hold trivially.

## Status of the bug family

flash_attention (fixed 08-27) · scaled_dot_product_attention (fixed 08-27)
· **layernorm (fixed 08-28, this round)**. The 08-27 project-wide sweep
found no other member among the 64 kernels; the `math_refs.py` known-
deviation note now reads "no live deviation known".

## Limits

- The ablation tables (CHECK_ABLATION.md) were not re-run; the −1
  correction is verified at the corpus-arm level (same checks, same seeds),
  which is the level the prediction was stated at.
- The frobenius value-flake classification rests on the prior rounds'
  differential evidence, not new instrumentation.
- The GPU probe's buggy-arm is a verbatim reconstruction, not a git
  checkout of the old file (the two differ only in the one line).

## Reproduce

```bash
# local scoring:
.venv/bin/python analysis/validate_lnfix.py arms/A_lnfix.json.gz arms/G_lnfix.json.gz

# GPU (T4):
export HOME=~/.colab-home && colab new --gpu T4 -s lnfix
tar --exclude='__pycache__' --exclude='.venv' -czf /tmp/kcc8.tgz \
    verification benchmarks scripts tests TritonBench
# upload kcc8.tgz, ../oob_fix_2026-08-28/probe_redundancy.py,
#   probes/ln_gpu_probe.py, lnfix.sh; launch lnfix.sh detached;
# poll /content/lnfix/DONE; download /content/probe/*.json.gz + logs
colab stop -s lnfix
```
