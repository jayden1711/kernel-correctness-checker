"""
Adversarial-search fixes from SESSION_HANDOFF.md section 2.1.

Covers all three fixes and, per this project's practice, verifies each by
BREAKING what it guards rather than by trusting a green run:

  1. OPERATOR_CONTEXT completeness -- all 21 wired operators, none falling
     back to the bare `Operator: <name>` label.
  2. validate_bug_pattern_hints + the startup assertion in _resolve_paths --
     NEGATIVE CONTROL: a deliberately-unhinted mutant id is injected into
     _MUTANT_MAP and the run must abort.
  3. _diagnose_reference_failure -- replayed against the 122 REAL reference
     failures recorded in adversarial_results/search_history.db, not synthetic
     cases, asserting that "magnitude" advice is now given only for the
     precision bucket.

--------------------------------------------------------------------------
THE `check_*.py` FILENAME IS LOAD-BEARING. DO NOT RENAME THIS FILE.
--------------------------------------------------------------------------
This file replaces sys.modules["torch"] with a stub at module scope.
tests/pytest.ini sets `python_files = test_*.py`, so a file named `check_*.py`
is never collected by pytest -- which is the entire point. Renaming it would
let pytest collect it into the same process as the real suite, where
tests/conftest.py imports the real torch and every tests/verification/* test
depends on it. See the README in this directory.

Run it directly:
    python3 tests/instrumentation/check_adversarial_search_fixes.py

Exit code 0 = all checks passed. Non-zero = failures, listed on stdout.
"""
import json
import os
import re
import sqlite3
import sys
import types
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("CHECKER_ROOT", str(REPO))

# ── stubs: only what the import graph needs, nothing more ────────────────────
for name in ("torch", "numpy", "litellm", "dotenv"):
    if name not in sys.modules:
        m = types.ModuleType(name)
        if name == "torch":
            m.cuda = types.SimpleNamespace(is_available=lambda: False)
            for _dt in ("float32", "float16", "bfloat16", "int32", "int64"):
                setattr(m, _dt, _dt)   # materializer builds DTYPE_MAP at import
        if name == "dotenv":
            m.load_dotenv = lambda *a, **k: None
        sys.modules[name] = m

from verification.adversarial_search.prompts.base import (      # noqa: E402
    OPERATOR_CONTEXT, BUG_PATTERN_HINTS, format_first_turn,
    validate_bug_pattern_hints,
)
from verification.adversarial_search.schemas import REQUIRED_TENSOR_KEYS  # noqa: E402

fails = []


def ck(label, cond, ctx=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   [{ctx}]" if not cond else ""))
    if not cond:
        fails.append(label)


# ── 1. OPERATOR_CONTEXT completeness ─────────────────────────────────────────
print("\n[1] OPERATOR_CONTEXT covers every wired operator")
wired = sorted(REQUIRED_TENSOR_KEYS)
ck(f"all {len(wired)} wired operators have context",
   all(o in OPERATOR_CONTEXT for o in wired),
   f"missing={[o for o in wired if o not in OPERATOR_CONTEXT]}")

# The fallback is OPERATOR_CONTEXT.get(op, f"Operator: {op}") -- i.e. the
# context being EXACTLY that bare string. Several real contexts legitimately
# start with "Operator: <name> ...", so this must be equality, not startswith.
bare_fallback = [o for o in wired if OPERATOR_CONTEXT.get(o) == f"Operator: {o}"]
ck("no wired operator resolves to the bare fallback string", not bare_fallback,
   f"{bare_fallback}")
thin = [o for o in wired if len(OPERATOR_CONTEXT[o]) < 120]
ck("every context is substantive (>120 chars), not a stub", not thin, f"{thin}")

# every entry must state its tensor keys and a rank, since rank confusion is
# the failure this fix exists to prevent
for op in wired:
    ctx = OPERATOR_CONTEXT[op]
    ck(f"{op}: context names its tensor keys", "Required tensors:" in ctx)
# Accepts an explicit RANK statement or any concrete dimensioned shape --
# matmul's pre-existing entry uses "(M, K)"/"(K, N)" rather than the word rank.
missing_rank = [o for o in wired
                if not re.search(r"RANK|2D|\([A-Za-z_]+, [A-Za-z_]+", OPERATOR_CONTEXT[o])]
ck("every context states a rank/shape convention", not missing_rank, f"{missing_rank}")

# NEGATIVE CONTROL: the fallback must still exist for a genuinely unwired op
ck("NEGATIVE CONTROL: unknown operator still falls back to the bare label",
   format_first_turn("not_a_real_operator").startswith(
       "Operator context:\nOperator: not_a_real_operator"))


# ── 2. hint validation + the startup assertion ───────────────────────────────
print("\n[2] BUG_PATTERN_HINTS validation and the _resolve_paths assertion")
import scripts.run_adversarial_search as runner  # noqa: E402

all_mutants = {m for mp in runner._MUTANT_MAP.values() for m in mp}
missing = validate_bug_pattern_hints(all_mutants)
ck(f"all {len(all_mutants)} real mutant ids have a hint", not missing, f"missing={missing}")
ck("wrong_causal_mask specifically now has a hint",
   "wrong_causal_mask" in BUG_PATTERN_HINTS)
ck("validator flags an unknown id",
   validate_bug_pattern_hints({"definitely_not_a_real_mutant"}) == ["definitely_not_a_real_mutant"])

# NEGATIVE CONTROL: inject an unhinted mutant that points at a REAL file, so it
# survives the existence filter and must be caught by the hint assertion. This
# also proves the assertion sits AFTER that filter, not before it.
real_file = "TritonBench/cheating/softmax/first_tile.py"
assert (REPO / real_file).exists(), "fixture file missing; update this control"
runner._MUTANT_MAP["softmax"]["unhinted_control_mutant"] = real_file
try:
    runner._resolve_paths("softmax")
    ck("NEGATIVE CONTROL: unhinted mutant aborts the run", False,
       "_resolve_paths returned normally -- the assertion did NOT fire")
except ValueError as e:
    msg = str(e)
    ck("NEGATIVE CONTROL: unhinted mutant aborts the run", True)
    ck("  ...error names the offending mutant", "unhinted_control_mutant" in msg, msg[:120])
    ck("  ...error says where to fix it", "BUG_PATTERN_HINTS" in msg, msg[:120])
except Exception as e:
    ck("NEGATIVE CONTROL: unhinted mutant aborts the run", False,
       f"raised {type(e).__name__} instead of ValueError: {e}")
finally:
    del runner._MUTANT_MAP["softmax"]["unhinted_control_mutant"]

# and the clean path must still work once the control is removed
try:
    runner._resolve_paths("softmax")
    ck("clean softmax resolve still succeeds after control removed", True)
except Exception as e:
    ck("clean softmax resolve still succeeds after control removed", False,
       f"{type(e).__name__}: {e}")


# ── 3. four-branch dispatch, replayed on REAL history ────────────────────────
print("\n[3] _diagnose_reference_failure replayed against recorded history")
from verification.adversarial_search.executor import _diagnose_reference_failure  # noqa: E402

db = REPO / "adversarial_results" / "search_history.db"
if not db.exists():
    ck("search_history.db present for replay", False, str(db))
else:
    con = sqlite3.connect(str(db))
    buckets, magnitude_in = Counter(), Counter()
    for (vj,) in con.execute("SELECT verdict_json FROM verdicts"):
        v = json.loads(vj)
        if v["reference_passed"]:
            continue
        m = re.search(r"Reference failed: \[([^\]]*)\]", v.get("failure_summary", ""))
        names = [x.strip().strip("'") for x in m.group(1).split(",") if x.strip()] if m else []
        # reconstruct the check_results shape the classifier consumes
        crs = [{"check_name": n, "passed": False, "details": ""} for n in names]
        label, advice = _diagnose_reference_failure(crs)
        buckets[label] += 1
        # Distinguish PRESCRIBING a magnitude change from explicitly ruling it
        # out. The non-precision branches deliberately say "magnitude is not
        # the problem here" -- that is corrective, not a repeat of the bug,
        # and it matters because the worker has prior turns in context where
        # "reduce magnitude" was the standing advice. Only a RECOMMENDATION
        # to change magnitude belongs solely in the precision branch.
        if re.search(r"(reduce|lower|shrink|scale down)[^.]{0,40}magnitude", advice, re.I):
            magnitude_in[label] += 1

    print(f"      replayed {sum(buckets.values())} recorded reference failures")
    for k, n in buckets.most_common():
        print(f"        {n:>3}x  {k}")

    ck("kernel_raised bucket = 44 (matches the audit)", buckets["kernel_raised"] == 44,
       str(buckets["kernel_raised"]))
    ck("degenerate_input bucket = 41", buckets["degenerate_input"] == 41,
       str(buckets["degenerate_input"]))
    ck("property_violated bucket = 24", buckets["property_violated"] == 24,
       str(buckets["property_violated"]))
    ck("precision bucket = 9", buckets["precision"] == 9, str(buckets["precision"]))
    ck("executor_crash bucket = 4", buckets["executor_crash"] == 4,
       str(buckets["executor_crash"]))

    # THE point of the fix: magnitude advice only where magnitude is the cause
    ck("a magnitude CHANGE is recommended ONLY in the precision bucket",
       set(magnitude_in) <= {"precision"}, f"also in {sorted(set(magnitude_in) - {'precision'})}")
    for _lbl in ("kernel_raised", "degenerate_input", "executor_crash"):
        _, _adv = _diagnose_reference_failure(
            [{"check_name": {"kernel_raised": "nan_inf", "degenerate_input": "kernel_executed",
                             "executor_crash": "x"}[_lbl], "passed": False, "details": ""},
             {"check_name": "dtype_preserved", "passed": False, "details": ""}]
            if _lbl == "kernel_raised" else
            ([{"check_name": "kernel_executed", "passed": False, "details": ""}]
             if _lbl == "degenerate_input" else []))
        ck(f"{_lbl}: explicitly rules magnitude OUT (corrective, not silent)",
           re.search(r"magnitude", _adv, re.I) is not None)
    ck("...and it does appear there", magnitude_in.get("precision", 0) == 9,
       str(magnitude_in.get("precision", 0)))

    # NEGATIVE CONTROL: empty check_results must NOT silently produce no hint
    label, advice = _diagnose_reference_failure([])
    ck("NEGATIVE CONTROL: empty check_results yields a real message",
       label == "executor_crash" and len(advice) > 40, f"{label}: {advice[:60]}")
    ck("  ...and does not RECOMMEND a magnitude change",
       not re.search(r"(reduce|lower|shrink|scale down)[^.]{0,40}magnitude", advice, re.I))

# ── 4. §2.2: verdict bucket split + per-mutant persistence ───────────────────
print("\n[4] verdict split: not_caught vs caught_no_gap")
from verification.adversarial_search.coordinator import SearchCoordinator  # noqa: E402
from verification.adversarial_search.schemas import (                      # noqa: E402
    ProposalVerdict, KernelExecutionResult)


def _mk(kid, passed_checker, passed_naive):
    return KernelExecutionResult(proposal_id="p", kernel_id=kid,
                                 passed_checker=passed_checker,
                                 passed_naive=passed_naive, error=None,
                                 check_results=[], wall_time_ms=0.0)


ref_ok = _mk("reference", True, True)
_prop = types.SimpleNamespace(proposal_id="test-proposal-0000")
# _evaluate_verdict touches no instance state, so it is called unbound.
v = SearchCoordinator._evaluate_verdict(None, _prop, ref_ok, [
    _mk("gap",       False, True),   # checker caught, allclose did not -> HIT
    _mk("no_gap",    False, False),  # checker caught, allclose caught too
    _mk("missed_a",  True,  True),   # checker did not catch
    _mk("missed_b",  True,  False),  # checker did not catch
])
ck("caught-with-gap -> hit_mutants", v.hit_mutants == ["gap"], v.hit_mutants)
ck("caught-but-no-gap -> caught_no_gap", v.caught_no_gap == ["no_gap"], v.caught_no_gap)
ck("not caught -> not_caught", v.not_caught == ["missed_a", "missed_b"], v.not_caught)
ck("BACKWARD COMPAT: missed_mutants == caught_no_gap + not_caught",
   v.missed_mutants == v.caught_no_gap + v.not_caught, v.missed_mutants)
ck("is_hit still true", v.is_hit is True)

ck("mutant_records carry both booleans for every mutant",
   len(v.mutant_records) == 4 and all(
       set(r) == {"kernel_id", "passed_checker", "passed_naive", "outcome"}
       for r in v.mutant_records))
outcomes = {r["kernel_id"]: r["outcome"] for r in v.mutant_records}
ck("record outcomes are correct",
   outcomes == {"gap": "caught_with_gap", "no_gap": "caught_no_gap",
                "missed_a": "not_caught", "missed_b": "not_caught"}, outcomes)

# the summary line that previously called a CAUGHT mutant "missed"
ck("failure_summary distinguishes caught-no-gap from not-caught",
   "Caught but no allclose gap" in v.failure_summary
   and "Not caught" in v.failure_summary, v.failure_summary)
ck("failure_summary no longer labels a caught mutant 'Missed'",
   "Missed:" not in v.failure_summary, v.failure_summary)

# NEGATIVE CONTROL: the CFA situation -- checker caught it, allclose did too.
# Before this fix that was reported identically to "checker missed it", which
# is what made the 120-proposal non-hit undiagnosable.
v2 = SearchCoordinator._evaluate_verdict(None, _prop, ref_ok,
                                         [_mk("wrong_causal_mask", False, False)])
ck("NEGATIVE CONTROL: caught-no-gap is NOT reported as not_caught",
   v2.caught_no_gap == ["wrong_causal_mask"] and v2.not_caught == [],
   f"no_gap={v2.caught_no_gap} not_caught={v2.not_caught}")
ck("  ...and is still absent from hit_mutants (no gap = no hit)",
   v2.hit_mutants == [] and v2.is_hit is False)
ck("  ...but is now distinguishable from a genuine miss",
   v2.caught_no_gap != v2.not_caught)

# round-trip
ck("to_dict/from_dict round-trips the new fields",
   ProposalVerdict.from_dict(v.to_dict()).mutant_records == v.mutant_records)

# REAL DATA: every stored verdict predates these fields and must still load
if db.exists():
    con2 = sqlite3.connect(str(db))
    rows = [json.loads(r[0]) for r in con2.execute("SELECT verdict_json FROM verdicts")]
    ck(f"all {len(rows)} pre-existing stored verdicts still load",
       all("not_caught" not in r for r in rows) and
       all(ProposalVerdict.from_dict(r).not_caught == [] for r in rows))
    ck("...and their missed_mutants is preserved verbatim",
       all(ProposalVerdict.from_dict(r).missed_mutants == r["missed_mutants"]
           for r in rows))
ck("from_dict drops unknown keys instead of raising",
   ProposalVerdict.from_dict({**v.to_dict(), "field_from_the_future": 1}).is_hit is True)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
