"""
A1 — per-check execution detail is persisted (SESSION_HANDOFF.md §4, item A1).

The executor computes full per-check pass/fail/details for the reference and
every mutant on every proposal, then used to discard it. This verifies the new
`executions` table actually captures it, that opening an EXISTING database
migrates cleanly, and that no pre-existing output changed.

--------------------------------------------------------------------------
THE `check_*.py` FILENAME IS LOAD-BEARING. DO NOT RENAME THIS FILE.
--------------------------------------------------------------------------
tests/pytest.ini sets `python_files = test_*.py`, so `check_*.py` is never
collected by pytest. These scripts stub heavy imports at module scope and would
corrupt tests/verification/* if collected into the same process. See the README
in this directory.

Run:  python3 tests/instrumentation/check_execution_persistence.py
Exit 0 = pass.
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from verification.adversarial_search.history.store import SearchHistoryStore   # noqa: E402
from verification.adversarial_search.schemas import (                          # noqa: E402
    ExecutionError, KernelExecutionResult, ProposalVerdict, SearchResult)

REAL_DB = REPO / "adversarial_results" / "search_history.db"
fails = []


def ck(label, cond, ctx=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   [{ctx}]" if not cond else ""))
    if not cond:
        fails.append(label)


def _mk_exec(kid, passed_checker=False, passed_naive=True, with_error=False):
    checks = [
        {"check_name": "nan_inf", "passed": True, "layer": 1, "details": "finite"},
        {"check_name": "perturbation_tolerance", "passed": passed_checker,
         "layer": 2, "details": "max_err=0.512340, adaptive_tol=0.000100"},
        {"check_name": "rows_sum_to_one", "passed": True, "layer": 3, "details": None},
    ]
    err = ExecutionError(error_type="TimeoutError", message="Timed out after 30s",
                         layer=None, check_name=None, max_err=None,
                         traceback_snippet="line 1\nline 2") if with_error else None
    return KernelExecutionResult(
        proposal_id="prop-0001", kernel_id=kid, passed_checker=passed_checker,
        passed_naive=passed_naive, error=err, check_results=([] if with_error else checks),
        wall_time_ms=12.5)


tmpdir = tempfile.mkdtemp(prefix="a1_")
tmp_db = os.path.join(tmpdir, "search_history.db")

# ── 1. migration safety against the REAL database (on a copy) ────────────────
print("\n[1] migration safety: opening an existing DB gains the table, touches nothing")
BASELINE = {"runs": 12, "proposals": 262, "verdicts": 260, "memory_items": 11}
if not REAL_DB.exists():
    ck("real search_history.db present", False, str(REAL_DB))
else:
    shutil.copy(REAL_DB, tmp_db)
    pre = {}
    con = sqlite3.connect(tmp_db)
    for t in BASELINE:
        pre[t] = (con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0],
                  [d[1] for d in con.execute(f"PRAGMA table_info({t})")])
    had_exec = bool(con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='executions'"
    ).fetchone()[0])
    con.close()
    ck("copy of the real DB has NO executions table beforehand", not had_exec)
    ck("baseline row counts match the recorded state",
       {t: pre[t][0] for t in BASELINE} == BASELINE, {t: pre[t][0] for t in BASELINE})

    store = SearchHistoryStore(tmp_db)          # applies _SCHEMA via executescript
    con = sqlite3.connect(tmp_db)
    now_has = bool(con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='executions'"
    ).fetchone()[0])
    ck("opening with the new store CREATES the executions table", now_has)
    drift = []
    for t in BASELINE:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        cols = [d[1] for d in con.execute(f"PRAGMA table_info({t})")]
        if (n, cols) != pre[t]:
            drift.append((t, pre[t], (n, cols)))
    ck("pre-existing tables unchanged (rows AND columns)", not drift, str(drift)[:160])
    con.close()

# ── 2. round-trip fidelity ───────────────────────────────────────────────────
print("\n[2] round-trip: check_results and error survive field-for-field")
run_id = store.create_run(run_id="run-a1", operator="softmax", strategy="beam",
                          model="test", n_workers=1, max_iter=1) or "run-a1"
src = _mk_exec("wrong_reduction", passed_checker=False, passed_naive=True)
store.save_execution("run-a1", src)
got = store.get_executions(proposal_id="prop-0001")
ck("one row written", len(got) == 1, len(got))
if got:
    r = got[0]
    ck("check_results round-trip exactly", r["check_results"] == src.check_results)
    ck("layer preserved on every check",
       [c["layer"] for c in r["check_results"]] == [1, 2, 3])
    ck("details preserved (incl. the numeric detail string)",
       any("max_err=0.512340" in (c["details"] or "") for c in r["check_results"]))
    ck("denormalised n_failed is correct", r["n_failed"] == 1, r["n_failed"])
    ck("denormalised booleans round-trip as bools",
       r["passed_checker"] is False and r["passed_naive"] is True)
    ck("operator resolved from the run", r["operator"] == "softmax", r["operator"])

store.save_execution("run-a1", _mk_exec("crashed", with_error=True))
err_rows = [e for e in store.get_executions(proposal_id="prop-0001")
            if e["kernel_id"] == "crashed"]
ck("ExecutionError persisted", len(err_rows) == 1 and err_rows[0]["error"] is not None)
if err_rows and err_rows[0]["error"]:
    ck("error_type denormalised for querying", err_rows[0]["error_type"] == "TimeoutError")
    ck("traceback snippet preserved",
       "line 2" in err_rows[0]["error"]["traceback_snippet"])

# ── 3. existing outputs unchanged ────────────────────────────────────────────
print("\n[3] pre-existing outputs are byte-identical")
v = ProposalVerdict(proposal_id="p", is_hit=False, hit_mutants=[],
                    missed_mutants=["m"], reference_passed=True,
                    gap_confirmed=False, failure_summary="s")
EXPECTED_VERDICT_KEYS = {"proposal_id", "is_hit", "hit_mutants", "missed_mutants",
                         "reference_passed", "gap_confirmed", "failure_summary",
                         "beam_score", "not_caught", "caught_no_gap", "mutant_records"}
ck("ProposalVerdict.to_dict() shape unchanged by A1",
   set(v.to_dict()) == EXPECTED_VERDICT_KEYS, sorted(set(v.to_dict()) ^ EXPECTED_VERDICT_KEYS))
sr = SearchResult(run_id="r", operator="softmax", strategy="beam", total_proposals=1,
                  total_iterations=1, n_workers=1, winning_proposal=None,
                  winning_verdict=None, all_verdicts=[v], wall_time_s=1.0, model="m")
ck("SearchResult.to_json() carries no execution records (unchanged)",
   "check_results" not in sr.to_json() and "executions" not in sr.to_json())

# ── 4. NEGATIVE CONTROL: the gap is actually closed ──────────────────────────
print("\n[4] NEGATIVE CONTROL: the CFA question is unanswerable before, answerable after")
CFA_SQL = ("SELECT kernel_id, passed_checker, check_results_json FROM executions "
           "WHERE proposal_id=? AND kernel_id=?")
probe_db = os.path.join(tmpdir, "probe.db")
probe = SearchHistoryStore(probe_db)
probe.create_run(run_id="run-probe", operator="causal_flash_attention",
                 strategy="diverse", model="test", n_workers=1, max_iter=1)
con = sqlite3.connect(probe_db)
before = con.execute(CFA_SQL, ("prop-cfa", "wrong_causal_mask")).fetchall()
ck("BEFORE: 'was it caught, and by which check?' has no answer", before == [], before)

probe.save_execution("run-probe", KernelExecutionResult(
    proposal_id="prop-cfa", kernel_id="wrong_causal_mask", passed_checker=False,
    passed_naive=False, error=None, wall_time_ms=9.0,
    check_results=[{"check_name": "convex_hull_bound", "passed": False,
                    "layer": 3, "details": "violated at row 7"}]))
after = con.execute(CFA_SQL, ("prop-cfa", "wrong_causal_mask")).fetchall()
ck("AFTER: the question is answerable from SQL alone", len(after) == 1)
if after:
    caught = not bool(after[0][1])
    which = [c["check_name"] for c in json.loads(after[0][2]) if not c["passed"]]
    ck("  ...and it says the checker DID catch it", caught)
    ck("  ...naming the specific check", which == ["convex_hull_bound"], which)
con.close()

# ── 5. crash-partial persistence ─────────────────────────────────────────────
print("\n[5] a reference row survives an abort before any mutant runs")
partial = SearchHistoryStore(os.path.join(tmpdir, "partial.db"))
partial.create_run(run_id="run-p", operator="softmax", strategy="beam",
                   model="test", n_workers=1, max_iter=1)
partial.save_execution("run-p", _mk_exec("reference", passed_checker=True))
# no mutant executions follow -- simulating a crash/timeout mid-loop
rows = partial.get_executions(run_id="run-p")
ck("reference execution persisted on its own", len(rows) == 1 and rows[0]["kernel_id"] == "reference")
ck("  ...with its check detail intact", len(rows[0]["check_results"]) == 3)

# ── 6. spawn-cost instrumentation (#7b) ──────────────────────────────────────
print("\n[6] NEGATIVE CONTROL: total_wall_time_ms captures spawn cost, and None != 0.0")
# WHY THIS EXISTS: on the 2026-08-20 CFA run, in-kernel time was 0.03s median
# while the spawn-to-result interval was 10.25s -- subprocess spawn plus
# `import torch`/triton was ~71% of each worker's wall time and NOTHING
# measured it. `wall_time_ms` is recorded inside the subprocess, which by
# construction cannot see its own startup cost. If total_wall_time_ms ever
# silently stops being populated, the single largest cost in the search goes
# invisible again while every existing assertion still passes.
spawn_db = os.path.join(tmpdir, "spawn.db")
SPAWN_MS = 10250.0     # the real measured median, not a made-up number
with SearchHistoryStore(spawn_db) as st6:
    st6.create_run(run_id="run-spawn", operator="causal_flash_attention",
                   strategy="beam", model="m", n_workers=4, max_iter=20)
    r = _mk_exec("reference")
    r.total_wall_time_ms = SPAWN_MS          # parent-stamped
    st6.save_execution("run-spawn", r)
    r2 = _mk_exec("mutant")                   # left as None: never measured
    st6.save_execution("run-spawn", r2)

rows = sqlite3.connect(spawn_db).execute(
    "SELECT kernel_id, wall_time_ms, total_wall_time_ms FROM executions "
    "ORDER BY kernel_id").fetchall()
by_kid = {k: (w, t) for k, w, t in rows}
ck("total_wall_time_ms round-trips through the DB",
   by_kid["reference"][1] == SPAWN_MS, f"got {by_kid['reference'][1]}")
ck("in-subprocess wall_time_ms is untouched by the new field",
   by_kid["reference"][0] == 12.5, f"got {by_kid['reference'][0]}")
ck("spawn overhead is recoverable (total - wall)",
   abs((by_kid["reference"][1] - by_kid["reference"][0]) - (SPAWN_MS - 12.5)) < 1e-6)
# The distinction that matters: an unmeasured spawn must read NULL, not 0.0.
# 0.0 would claim a free spawn and silently drag any future average toward zero
# -- the same "never ran vs ran instantly" trap as duration_ms in _try.
ck("NEVER-MEASURED spawn persists as NULL, not 0.0",
   by_kid["mutant"][1] is None, f"got {by_kid['mutant'][1]!r}")
# Control: if the field stopped being written, the reference row would read
# NULL too and this comparison would collapse.
ck("control: measured and unmeasured rows are distinguishable",
   by_kid["reference"][1] is not None and by_kid["mutant"][1] is None)

shutil.rmtree(tmpdir, ignore_errors=True)
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
