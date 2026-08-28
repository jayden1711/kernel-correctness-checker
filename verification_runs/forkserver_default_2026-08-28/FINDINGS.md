# `use_forkserver` defaults ON: all three gates re-verified AT THE DEFAULT (140/140 kernel results forkserver with zero fallbacks, drift 0.9%/1.4%, timeout probes identical), effect −42%/−35% matching the A/B, corpus regression 40/40 0/200 — and the reseed guard now runs under pytest, so a regression can no longer ship green

**Flipped and GPU-verified 2026-08-28** (T4 session `kccflip`, stopped).
This adopts the decision `verification_runs/forkserver_ab/` was left
pending: that A/B measured 36–41% end-to-end with `summary.valid` true,
order drift <2% and `timeout_probe.identical` true — but it passed
`use_forkserver` explicitly per arm, so the *default path* had never been
exercised. Probe `probes/verify_default.py`, raw results and log in
`data/`.

## 1. The flip

Three sites, one decision:

| site | before | after |
|---|---|---|
| `executor.execute_proposal_batch` | `use_forkserver: bool = False` | `True` (docstring records the A/B + this re-verification) |
| `coordinator.SearchCoordinator` | `False`, "pending its own GPU before/after" | `True` (comment replaced with the dated decision + escape hatch) |
| `scripts/run_adversarial_search.py --forkserver` | `store_true`, default off | `BooleanOptionalAction`, default on; `--no-forkserver` restores spawn |

Unchanged on purpose: the single-kernel path stays spawn-only (it does not
seed; `check_batch_executor.py` §11 still asserts it never asks for
forkserver), platforms without forkserver still degrade to spawn with the
method recorded on every result, and `--no-batch` still implies spawn.

## 2. Gates re-verified at the new default, not assumed from the A/B

`verify_default.py` reruns the order-controlled replay protocol
(A1 spawn → **D with the kwarg OMITTED** → A2 spawn; recorded proposals,
no LLM; full A/B replay sizes, cfa 40 × 2 kernels + fa 12 × 5 kernels),
after first asserting the shipped signature default is True:

| operator | D methods | drift A1↔A2 | timeouts | forced-timeout probe | effect vs spawn | torch_import p50 |
|---|---|---|---|---|---|---|
| causal_flash_attention | **forkserver: 80/80** | **0.9%** | 0=0=0 | **IDENTICAL** | **0.58× (−42%)** | 4665 ms → **0.4 ms** |
| flash_attention | **forkserver: 60/60** | **1.4%** | 0=0=0 | **IDENTICAL** | **0.65× (−35%)** | 5712 ms → **0.4 ms** |

- **Gate 1** (the arm users actually get really forks, no silent
  fallback): every one of the 140 default-arm kernel results records
  `start_method == "forkserver"`.
- **Gate 2** (order drift < 2%): 0.9% and 1.4% — the spawn passes
  bracketing D agree, so warm caches are not the effect.
- **Gate 3** (timeout semantics): zero natural timeouts in every arm AND
  the forced-timeout probe (1s budget both arms must miss, spawn-explicit
  vs default-omitted) returns identical result counts and error sets.
- The effect sizes reproduce the A/B's 36–41% band on a different day and
  VM, now measured through the default path.

## 3. Correctness safety nets

- **Reseed guard passes at the new default**:
  `check_forkserver_executor.py` exit 0 (seed applied, applied before the
  first draw, derived from the proposal id; every positive assertion still
  flips under its paired mutation). `check_batch_executor.py` exit 0, and
  it now ALSO asserts the flipped defaults by signature (executor +
  coordinator), so a silent revert fails the guard.
- **The guards are no longer manual-only.** `tests/pytest.ini` collects
  `test_*.py`, so both `check_*.py` guards were invisible to CI — a reseed
  regression (identically-seeded tensors for every proposal: silent,
  severe, exactly what the guard exists for) would have shipped with a
  green suite. `tests/instrumentation/test_guard_scripts.py` now runs both
  under pytest **as subprocesses** — renaming was rejected because the
  scripts stub `sys.modules["torch"]` process-wide and in-process
  collection would corrupt `tests/verification/*` (the README's standing
  prohibition, which still holds and is amended with the dated
  subprocess exception). Suite: 604 passed + the 2 wrappers; the single
  pre-existing failure (`test_worker_parsing::test_all_retries_exhausted_raises`,
  a worker-vs-test drift about the exception type in the working tree)
  predates and is untouched by this change.
- **Corpus regression after the flip: 40/40 catch, 0/200 FP**
  (`results_gpu/kccflip_20260828_083346/`, the cold run of the
  cache-automation round — the flip cannot reach the corpus path, and the
  regression ran anyway per the adoption protocol).

## Limits

- Timing arms are single passes per operator at the A/B's replay sizes on
  one T4; the gates (per-record method stamps, drift, timeout identity)
  are the decision criteria, the −42%/−35% medians are corroboration of an
  effect the A/B already established with order control.
- The forced-timeout probe exercises the 1s path once per operator per
  method — same coverage as the A/B's probe, no deeper.
- `verify_default` asserts the *executor* default; the coordinator default
  is covered by the signature check in the batch guard rather than a live
  search run (an LLM-driven search would not be work-identical across
  arms, which is the same reason the A/B replayed recorded proposals).

## Reproduce

```bash
# guards + suite (local, no GPU):
.venv/bin/python -m pytest tests/instrumentation/test_guard_scripts.py -q
# gates (GPU box, DBs staged under /content/adversarial_results/):
python verification_runs/forkserver_default_2026-08-28/probes/verify_default.py \
    --root /content --out /content/forkserver_default.json
```
