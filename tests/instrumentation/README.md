# Instrumentation checks

Standalone verification for the benchmark-harness instrumentation. **These are
not pytest tests and must not become pytest tests** — *in-process*. Since
2026-08-28, `test_guard_scripts.py` here IS collected by pytest and runs
`check_forkserver_executor.py` and `check_batch_executor.py` **as
subprocesses**, asserting exit 0. That closes the hole where a reseed
regression (the forkserver guard's whole subject, now that `use_forkserver`
defaults ON) would ship with a green suite, while every isolation argument
below still holds: the stubs live and die in the child interpreter. The other
check scripts stay manual-only on purpose (one needs a real torch and is
expected to fail locally; the rest guard instrumentation whose breakage is
visible in its own reports).

## Running them

```bash
python3 tests/instrumentation/check_item2_instrumentation.py
python3 tests/instrumentation/check_ablation_report.py
python3 tests/instrumentation/check_autokernel_faithful_construction.py
python3 tests/instrumentation/check_batch_executor.py
```

Plain `python3` — no venv, no numpy, no torch, no pytest. Exit code 0 = pass,
non-zero = failure with the failing assertions printed.

Run them all:

```bash
for f in tests/instrumentation/check_*.py; do python3 "$f" >/dev/null || echo "FAIL $f"; done
```

### One exception: `check_kernel_executed_probe.py` needs a real torch

Every other script here runs with stubs on the dev machine. That one cannot:
the defect it guards is **numerical** — whether two float32 outputs are bitwise
equal is the entire question — so a shape-recording stub has nothing to say
about it. It needs a real `torch`, though **not a GPU**; the Colab VM's CPU is
enough. On the dev machine it will fail at `import torch`, which is expected,
not a regression. Run it in the same Colab session as any GPU work:

```bash
PYTHONPATH=/content python3 /content/tests/instrumentation/check_kernel_executed_probe.py
```

The loop above will report it as FAIL locally. That is the one entry to expect;
everything else must be green on the dev machine.

## Why they are not under pytest

1. **They stub `sys.modules["torch"]` and `sys.modules["numpy"]` process-wide.**
   `tests/conftest.py` imports the real `torch` at module scope and every
   `tests/verification/*` test depends on it. If pytest collected these in the
   same process, the stubs would leak and corrupt the rest of the suite.
2. `pytest.ini` sets `python_files = test_*.py`. These are named `check_*.py`
   precisely so collection skips them. **Do not rename them to `test_*.py`.**
3. The stub approach is also the only practical one here: the `.venv` lives on
   Google Drive File Stream and importing the real `torch` from it stalls for
   10+ minutes on network I/O. See `SESSION_HANDOFF.md` §0.

Stubbing is not a weaker test for this purpose — the defects these guard against
are **arity, shape and dtype** bugs, not numerical ones. A stub that records
shape/dtype catches them more directly than a real CPU run would, and works on a
machine with no GPU.

## What each covers

| File | Assertions | Guards |
|---|---|---|
| `check_item2_instrumentation.py` | 44 | `_try`'s four-valued outcome mapping (pass/fail/error/skip); subcheck passthrough for compound checks; `_summarize`'s joined detail string byte-identical to the pre-instrumentation implementation; `harness.summarize()` output byte-identical with and without the new `check_records` key; `harness._call` accepting both 3- and 4-tuple systems |
| `check_ablation_report.py` | 13 | `benchmarks/analyze_check_ablation.py` against a fixture with hand-derived counts: error-not-counted-as-catch, identical-catch-set redundancy detection, never-ran roster entry, operator with no properties, and a deliberate crash-as-catch that the consistency check must flag |
| `check_adversarial_search_fixes.py` | 60 | The three §2.1 adversarial-search fixes: `OPERATOR_CONTEXT` covers all 21 wired operators with tensor keys and a stated rank; `BUG_PATTERN_HINTS` covers every real mutant id; the `_resolve_paths` startup assertion **actually aborts** on a deliberately-unhinted mutant that points at a real file (so it also proves the check sits after the existence filter); and `_diagnose_reference_failure` replayed against the **122 real reference failures** in `search_history.db`, asserting a magnitude change is recommended only for the precision bucket |
| `check_autokernel_faithful_construction.py` | 12 families x (8 sweep + edge) shapes x 3 dtypes, **plus 3 negative controls run every time** | `autokernel_faithful` argument construction: correct arity per family, no dtype leaks, 8 configs per family, stage-3 probe coverage. **Does not validate numerics** — that needs a GPU. |
| `check_batch_executor.py` | 48 assertions, **6 mutations verified to trip the suite** | Item 2, one subprocess per PROPOSAL instead of one per kernel. Parent-side drain loop (ordering, per-kernel deadlines, exit grace, `on_result` streaming), the fallback that re-runs unreported kernels through the unchanged single-kernel path — **and the control that it does NOT fire on a clean batch** — the child's materialise-once/clone-per-kernel contract, proposal-derived seeding, and the poisoned-CUDA-context guard. Also replays kernel grouping against the real 160-row CFA history and round-trips the four new `executions` columns through a copy of that DB. |
| `check_kernel_executed_probe.py` (**needs real torch**) | 76 recorded cases x 4 controls, 8 seeds | The `check_kernel_executed` probe-ladder fix (`CHECK_ABLATION_FINDINGS.md` §3.0), replayed against the **real recorded tensor descriptors** from the adversarial-search history: 25 recorded false positives must now pass, 51 recorded passes must still pass, a genuine ghost kernel must still be caught, and each rung is measured **in isolation** (leave-one-IN). Control 0 runs the *old* check and requires it to fail all 25 — without it, "25/25 now pass" would be unfalsifiable. |

## The two guarantees worth understanding

**`check_item2_instrumentation.py` exists to prove a negative.** Adding per-check
attribution had to leave every pre-existing harness output bit-for-bit unchanged,
so the instrumented re-run stays comparable to previous runs. `_try` therefore
keeps the *original* (wrong) bool-coercion semantics for the verdict — an
exception and a skip both still yield `passed=False` — and fixes the semantics
only in the record's `outcome` field. If someone "cleans up" that coercion, these
assertions fail, and that is the intended behaviour: the fix is tracked
separately (`verification/checker.py:229`) precisely so it does not land silently
alongside an instrumentation change.

**`check_autokernel_faithful_construction.py` re-verifies itself on every run.**
It does not just run the happy path. After checking the shipped module, it
mutates a copy of that module's source three ways — shortened sweep, layernorm
arity bug, dtype leak — and *requires the checks to fail on each*. A control that
does not trip is a hard failure, and so is an anchor string that no longer
matches the source (a refactor would otherwise silently disarm the self-check
while leaving the run green). The last two mutations are exactly the bug classes
item #1 found in the old AutoKernel baseline.

This structure exists because the earlier version passed while validating a stale
copy under `/tmp` — a green run proved nothing, and only a hand-run negative
control exposed it. The script also prints `module under test: <path>` every run;
if that is ever not the repo's own
`benchmarks/autokernel/files/autokernel_faithful.py`, the run is worthless.

**`check_batch_executor.py` was itself validated by breaking the executor.**
Six mutations were applied to `executor.py` in turn — drop the per-kernel clone,
drop the poisoned-context guard, drop the seeding, drop the fallback, join before
draining, truncate the setup-failure fanout — and each was confirmed to fail the
suite. The join-before-drain case is worth knowing about: the first version of
that check compared source-text positions, which a `p.join(timeout=0.01)` walked
straight past. It is now *also* a functional reproduction — a bounded queue makes
the child block in `put()` exactly as a real `mp.Queue` does against the OS pipe
buffer, and a join-first parent then hangs and fails on a watchdog. A control
that only reads source text can be defeated by a refactor that changes nothing
semantically; prefer one that reproduces the failure.

**If you modify the instrumentation, re-run these first, and prefer verifying a
guard by breaking the thing it guards** rather than trusting a green run.
