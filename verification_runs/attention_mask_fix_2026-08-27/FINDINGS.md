# Masking fix SHIPPED — 40/200 regression clean with zero attribution drift; one impact-report prediction corrected; reference-failure categorization split shipped, and its first review pass surfaced 42 more records worth a look

**Shipped 2026-08-27.** GPU verification on a Colab T4 (session `maskfix`,
provisioned and stopped; same stack as all prior rounds). Probes in `probes/`,
raw data in `data/` (incl. the full post-fix regression arm
`POSTFIX_n40.json.gz` and a pre-relabel DB backup). Local analysis logs in
`data/*.log`.

---

## Verdict up front

| | result |
|---|---|
| The fix | One `tl.where` per kernel (`flash_attention.py`, `scaled_dot_product_attention.py`): padded key columns get `S = −inf`. `causal_flash_attention` untouched (immune). |
| Correctness gate | **27/27** (3 ops × 9 shapes incl. every previously-affected one): worst rel err vs true attention **9.1e-7**; weights-sum deviation ≤ 9.5e-7; N=1 exact. |
| **Full corpus regression (the canonical 40-mutant / 200-reference arm)** | **catch 40/40, FP 0/200 — and ZERO verdict diffs, ZERO catch-attribution diffs** vs the banked CURVE_n40 baseline: every mutant caught by the identical check list. |
| July N=130 proposals | Replayed on GPU against the fixed reference: all three **valid non-hit** exactly as the impact round predicted — reference passes its invariant (dev ≤ 4.8e-7), mutants caught, **no naive gap** (the recorded gap was a shared-bug artifact). Not a hit — the recorded flash search outcome stands. |
| The predicted `last_tile_dropped` ×~1.2 shift | **Did not materialise as predicted, and the prediction was wrong-in-form** — see §2. Both kernels' sensitivities on that variant are fp32-quantization-floor-pinned (buggy exactly 2 ulp at the 1e4 output scale — bitwise-identical tol across 7 of 10 seeds; fixed ~1 ulp), so a ratio between them is ulp-count noise (0.47–1.01). The classification invariance the taxonomy rests on is confirmed **20/20**. Correction note added to the impact report. |
| Live re-verification of "unaffected" quantities | 100%-catch/0%-FP: **re-measured, holds**. Exception taxonomy + scope screen: fp-floor class fires both ways (s_med/ulp 0.6–0.8 fixed vs 2–3 banked, ≪ 32). Gram law at 100×32: flash 0.895 → **0.977**, sdpa 0.863 → **0.933** median meas/pred, all \|z\| ≤ 1.08 — the −10% anomaly is gone. **Causal's measurements are bitwise identical pre/post fix** — the strongest no-op control. Search-efficiency table and 120/120 hit-invariant: unchanged (no verdict moved). |
| Part 2 | `reference_failure_kind` shipped (schema + coordinator + review script + 12 tests); 3 July records relabelled additively; **first review pass over all DBs found 45 reference-suspect verdicts — the 3 known ones plus 42 previously unexamined** (§4). |
| Tests | Search suite 226 passed (12 new); rest of tests/ 80 passed; the 1 pre-existing `test_worker_parsing` failure is untouched (documented in phase1 FINDINGS §7). |

---

## 1. Part 1 — the fix and its regression evidence

The change (`git diff TritonBench/reference/`): 13 lines, two kernels, one
semantic line each —

```
S = tl.where(kv_offsets[None, :] < N, S, float('-inf'))
```

placed after the scores are computed, before the running max. Every tile
contains ≥1 valid column (the loop bound is N), so `m_new` stays finite and
no NaN path opens; N=1 verifies exact.

**Stage A — correctness.** Fixed kernels vs true attention at
(1,64), (64,32), (65,64), (100,32), (128,64), (130,64), (192,64), (256,16),
(333,64): worst rel err 9.1e-7 (flash & sdpa), causal control 2.8e-7.
Weights-sum invariant ≤ 9.5e-7 everywhere — including the three spec-listed
shapes that previously produced 97% / 22% / 2% error.

**Stage D — the canonical regression.** `probe_redundancy.py` re-run with the
CURVE_n40 environment (n=40, recorded sensitivities, ablation seed 1) on the
fixed tarball: **catch 40/40, FP 0/200**, and a record-level diff against the
banked baseline shows **0 verdict changes and 0 catch-attribution changes**
across all 40 corpus entries — the per-mutant catching-check lists are
string-identical. The fix is verdict-neutral on the entire recorded corpus,
exactly as the impact assessment predicted.

**Stage E — the Gram-law recovery.** Post-fix 100×32 measurements (same
generator seeds as the attention_gram run):

| op | meas/pred median, buggy run | post-fix |
|---|--:|--:|
| `flash_attention` | 0.895 | **0.977** |
| `scaled_dot_product_attention` | 0.863 | **0.933** |
| `causal_flash_attention` | 0.977 | **0.977 (bitwise identical y)** |

All 9 post-fix points within \|z\| ≤ 1.08 of the pre-registered math-Jacobian
predictions. Causal being bit-identical while flash/sdpa move exactly onto
the predictions is the cleanest possible demonstration that the fix changed
precisely the thing the theory said was wrong and nothing else.

## 2. The one prediction that did not survive contact: the ×~1.2 tolerance shift

The impact report predicted the banked `last_tile_dropped` tolerances would
shift ×~1.2 under correction. Measured with buggy and fixed kernels on
**identical inputs and identical deltas** (stage B, 10 seeds): the ratio is
0.47–1.01, median 0.59 — not 1.2, and not smooth. The reason is visible in
the raw numbers: the buggy tol is **exactly 5.859e-3 = 2 ulp at the 1e4
output scale, bitwise identical across 7 of 10 seeds**, and the fixed tol is
~1 ulp. Both sensitivities are pinned to the fp32 quantization floor — which
is precisely this variant's documented taxonomy role — and a ratio between
two floor-pinned values is representable-spacing arithmetic, not a
measurement. The impact round's CPU emulation was itself floor-quantized
(0.8–1.5e-3 against ulp = 9.77e-4) and its "×1.2" read 1-vs-2-ulp counts as a
smooth shift. (Two seeds flip the dominant-key winner under perturbation and
land in a hard-select regime with tol ~3e4 — identically under both kernels.)

**What survives, and is what the taxonomy and the scope-round screen
validation actually rest on:** the fp-floor *classification* is invariant —
20/20 seeds classify identically under buggy and fixed kernels, s_med/ulp
0.6–0.8 (fixed) vs 2–3 (banked buggy), both far inside the ≤32 screen.
A correction note has been added to the impact report's §5.

## 3. Part 2 — the reference-failure split

- **`verification/adversarial_search/reference_failure.py`** — single source
  of truth. `"domain"` only when every failing check is on a **curated,
  closed** list ({nan_inf, dtype_preserved, output_shape, kernel_executed})
  or the execution errored; **anything else — including checks that do not
  exist yet — classifies `"invariant"` (REFERENCE-SUSPECT)**. The asymmetry
  is the anti-recurrence design: unknown failures err loud, never silent.
- **`ProposalVerdict.reference_failure_kind`** (schema is drift-tolerant both
  directions; old records read fine). The coordinator populates it, prefixes
  the failure summary with `REFERENCE-SUSPECT (...)`, and prints a warning
  pointing at the review script.
- **`scripts/review_reference_failures.py`** — the periodic check: scans
  history DBs, classifies every reference-failed verdict (pre-split records
  via their stored summaries, same classifier), groups by
  (operator, failed-checks), prints shapes, **exits non-zero when anything is
  reference-suspect** so it can gate CI or close out a search run.
- **12 unit tests** (`test_reference_failure.py`), including the verbatim
  July signature, the unknown-future-check-defaults-to-invariant guard, and a
  test that pins the curated domain list so growing it forces a visible diff.
- **Retroactive relabel**: the 3 July verdicts updated **additively**
  (original summary preserved under `original_failure_summary`; provenance
  note with the corrected-world classification "valid non-hit" and evidence
  pointers). Idempotent; DB backed up first
  (`data/search_history_pre_relabel.db.bak`).

## 4. What the first review pass surfaced — new, explicitly unadjudicated

Running the review script over all three DBs: **45 reference-suspect
verdicts. Three are the known July records. The other 42 were sitting in the
same "invalid input" bucket and have never been examined:**

| operator | n | invariants violated by the reference |
|---|--:|---|
| `instancenorm` | 15 | `unit_variance` (±`zero_mean`, `positive_scale_invariance`) |
| `softmax` | 12 | `tile_coverage_softmax_positivity` — *adjudicated 2026-08-28: all 12 check-domain false alarms (fp-underflow outside the check's derived validity domain, ideal math fails identically, margins ≥9.6×); `../l3_validity_2026-08-28/FINDINGS.md` §3* |
| `layernorm` | 11 | `unit_variance` / `precision_coercion` / `scale_invariance` combos |
| `matmul` | 2 | `scalar_associativity` — one at shape **(65,65)** — *adjudicated 2026-08-28: both check-domain false alarms of the since-loosened atol=1e-4 tolerance (any correct fp32 matmul fails it; the (65,65) is incidental — FP 10/10 at pow2 controls too); `../matmul_assoc_2026-08-28/FINDINGS.md`* |
| `rmsnorm` | 2 | `precision_coercion` / `unit_rms` |

These are *candidates*, not convictions: several match the documented
check-validity-domain issue (`run_random_baseline.py:45-56` — normalisation
invariants break when variance ≲ eps, exactly instancenorm/layernorm's
pattern on extreme proposals), and `precision_coercion` failing on extreme
inputs is plausibly the check leaving its own domain rather than the
reference being wrong. But the flash records looked like "just bad inputs"
too, for a month. **Each group needs the same treatment the flash records
got: emulate or re-run the reference on the recorded input and decide whether
the invariant genuinely fails on an in-contract input.** Deliberately NOT
adjudicated in this pass — that is new investigation, and it does not ride on
a shipping change. The matmul `(65,65)` and softmax `(512,333)` rows deserve
first look, given how this class of shape has behaved.

> **CLAIMS SWEEP 2026-08-27 — see `CLAIMS_SWEEP.md`.** Flag-only
> cross-reference of all 42 against every reported claim: corpus/theory/
> latency claims have **zero** dependency (the records exist only in the
> search DB); the one direct dependency is BUG_CLASS_THEORY §4's
> quick-hit claim, whose "valid proposal" denominators are exactly the
> complements of the flagged sets (18−15=3, 12−11=1, 4−2=2). **Adjudication
> priority is therefore revised: norm family first (28 records, resolves the
> only claim dependency in one mechanism-class pass), softmax second
> (inside the already-scoped RANGE_LIMIT fit), matmul demoted** — bug-like
> on its face but claim-independent. The first-look suggestion above is
> superseded.

## 5. Documents updated

- `attention_gram/ATTENTION_GRAM.md` §4 — FIXED note with the regression
  evidence.
- `adaptive_tol_theory_2026-08-25/GPU_NATIVE.md` addendum — fixed pointer.
- `attention_mask_bug_impact_2026-08-27/FINDINGS.md` §5 — fix-shipped note +
  the ×1.2-prediction correction (§2 here).
- The banked buggy-era `attn_native` `last_tile_dropped` tolerances remain
  as-recorded (historical artifacts); their post-fix counterparts and the
  quantization-floor explanation live here (stage B, `data/fix_suite.jsonl`).

## 6. Reproduce

```bash
# CPU-side
PY=.venv/bin/python
$PY -m pytest tests/verification/adversarial_search/ -q     # 226 pass (+1 pre-existing worker failure)
$PY scripts/review_reference_failures.py                    # 45 reference-suspect, exit 2
$PY verification_runs/attention_mask_fix_2026-08-27/probes/relabel_july.py   # idempotent

# GPU verification (T4)
export HOME=~/.colab-home
colab new --gpu T4 -s maskfix
tar czf /tmp/kccfix.tgz --exclude='__pycache__' verification TritonBench benchmarks/autokernel/files
colab upload -s maskfix /tmp/kccfix.tgz /content/kccfix.tgz          # + extract
colab upload -s maskfix verification_runs/attention_mask_fix_2026-08-27/probes/flash_attention_buggy.py /content/flash_attention_buggy.py
colab upload -s maskfix verification_runs/n_samples_curve_2026-08-25/probe_redundancy.py /content/probe_redundancy.py
colab exec  -s maskfix -f verification_runs/attention_mask_fix_2026-08-27/probes/fix_suite_gpu.py --timeout 1200
# regression arm: KCC_ARM=POSTFIX_n40 KCC_ABLATION_SEED=1 KCC_N_SAMPLES=40 KCC_RECORD_SENSITIVITIES=1 probe_redundancy.py
colab stop  -s maskfix

$PY - # diff POSTFIX_n40 vs banked CURVE_n40 -> data/regression_diff.log
```

Total GPU compute: under 15 minutes including two cold regression passes.

## 7. Limits

- The regression baseline (CURVE_n40) was recorded 2026-08-25 on the same
  GPU model and seed regime; the diff is per-verdict and per-attribution,
  not per-float — tolerances inside records legitimately differ at the
  affected variant (§2) and were not asserted equal.
- Stage C replays the July proposals' *deterministic* constructions exactly;
  the two randn-fill proposals use fresh draws (no recorded seeds exist), as
  in the impact round — their classification was seed-invariant 10/10 there
  and the GPU replay agrees on the drawn instance.
- The 42 newly surfaced reference-suspect records are flagged, not
  adjudicated (§4).
- The review script's domain list is curated and closed; if a genuinely new
  domain-failure check is added to the checker, it must be added there
  deliberately (a unit test pins the list to force that).
