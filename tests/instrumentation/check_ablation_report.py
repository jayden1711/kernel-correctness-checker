"""
Ablation report generator: benchmarks/analyze_check_ablation.py.

Runs the generator against a fixture whose counts were derived by hand,
including a deliberate crash-as-catch that the consistency check must flag.

--------------------------------------------------------------------------
THE `check_*.py` FILENAME IS LOAD-BEARING. DO NOT RENAME THIS FILE.
--------------------------------------------------------------------------
This file replaces sys.modules["torch"] and sys.modules["numpy"] with stubs at
module scope. tests/pytest.ini sets `python_files = test_*.py`, so a file named
`check_*.py` is never collected by pytest -- which is the entire point.

Renaming this to the conventional `test_*.py` would let pytest collect it into
the same process as the real suite. tests/conftest.py imports the real torch at
module scope and every tests/verification/* test depends on it, so the stubs
would leak and corrupt those tests. The failure would look like unrelated tests
breaking, not like a naming problem.

Run it directly instead:
    python3 tests/instrumentation/check_ablation_report.py

Plain python3 -- no venv, no numpy, no torch, no pytest. See the README in this
directory for the full rationale.

Exit code 0 = all assertions passed. Non-zero = failures, listed on stdout.
"""
import json, sys, os, tempfile, importlib.util
import os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = str(REPO / "benchmarks")

def R(name, outcome, subs=None):
    return {"name": name, "outcome": outcome, "detail": None, "subchecks": subs}

XS_PASS = [{"name": "shape=(512,512)", "outcome": "pass", "detail": None},
           {"name": "shape=(1,512)",   "outcome": "pass", "detail": None}]
XS_FAIL = [{"name": "shape=(512,512)", "outcome": "pass", "detail": None},
           {"name": "shape=(1,512)",   "outcome": "fail", "detail": "max_err=0.5"}]

def M(op, mut, caught, recs):
    return {"op": op, "mutant": mut, "caught": caught, "detail": None, "check_records": recs}

numeric_mutants = [
    # perturbation_tolerance and weight_magnitude catch the SAME two mutants -> identical/redundant
    #
    # NEGATIVE CONTROL (int-slot): tile_coverage_softmax_positivity carries an
    # INT in `subchecks`, not a list. This is not hypothetical -- it is the
    # exact shape a real corpus run produced (softmax/first_tile, subchecks=64,
    # the column count from tile_coverage.py's partial-coverage branch), and it
    # crashed the whole report with `TypeError: 'int' object is not iterable`.
    # Only 2 of 322 records were malformed, yet attribution for all 94 checks
    # was lost. The original fixture modelled `subchecks` as list-or-None only,
    # which is why 13 assertions passed against a reader that could not survive
    # real data. Keep this record: it fails loudly if _expand's isinstance
    # guard is ever removed. Attached to an existing mutant on purpose, so the
    # per-check counts asserted below are undisturbed.
    M("softmax","first_tile",True,  [R("output_shape","pass"),R("perturbation_tolerance","fail"),R("cross_shape","pass",XS_PASS),R("weight_magnitude","fail",[{"name":"large_uniform","outcome":"fail","detail":None}]),R("tile_coverage_softmax_positivity","fail",64)]),
    M("softmax","wrong_reduction",True,[R("output_shape","pass"),R("perturbation_tolerance","fail"),R("cross_shape","pass",XS_PASS),R("weight_magnitude","fail",[{"name":"large_uniform","outcome":"fail","detail":None}])]),
    M("matmul","partial_k",True,    [R("output_shape","pass"),R("perturbation_tolerance","pass"),R("cross_shape","fail",XS_FAIL),R("weight_magnitude","pass",[{"name":"large_uniform","outcome":"pass","detail":None}])]),
    M("gelu","sigmoid_approx",False,[R("output_shape","pass"),R("perturbation_tolerance","pass"),R("cross_shape","pass",XS_PASS),R("weight_magnitude","pass",[{"name":"large_uniform","outcome":"pass","detail":None}])]),
    # scored caught by the harness, but its ONLY non-pass record is an ERROR --
    # i.e. the "catch" was a crash. Consistency check must flag this.
    M("l1norm","partial_reduction",True,[R("output_shape","error"),R("perturbation_tolerance","pass"),R("cross_shape","pass",XS_PASS),R("weight_magnitude","pass",[])]),
]
raw = {
 "your_checker (numeric only)": {
   "mutant_results": numeric_mutants,
   "ref_results": [{"op":"matmul","false_positive":True,"detail":None,
                    "check_records":[R("cross_shape","fail",XS_FAIL)]}],
   "latencies":[0.01]},
 "your_checker (algebraic only)": {
   "mutant_results":[
     M("softmax","first_tile",True,[R("rows_sum_to_one","fail"),R("shift_invariance","pass")]),
     M("softmax","wrong_reduction",False,[R("rows_sum_to_one","pass"),R("shift_invariance","pass")]),
     M("batchnorm","wrong_stats",False,[]),   # no properties at all
   ],
   "ref_results":[], "latencies":[0.01]},
}
TMP = tempfile.mkdtemp(prefix="abl_")
p = os.path.join(TMP, "results_raw.json")
json.dump(raw, open(p,"w"))

spec = importlib.util.spec_from_file_location("aca", os.path.join(BENCH,"analyze_check_ablation.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.OUT_PATH = os.path.join(TMP, "CHECK_ABLATION.md")
sys.argv = ["x", p]

# main() is called inside a guard so the int-slot negative control reports as a
# named failure rather than an uncaught traceback. A TypeError here means
# _expand stopped validating `subchecks` and the reader is once again unable to
# process real corpus data.
_crash = None
try:
    m.main()
except Exception as e:
    _crash = f"{type(e).__name__}: {e}"

fails=[]
def ck(label, cond, ctx=""):
    print(("  PASS  " if cond else "  FAIL  ")+label+(f"   [{ctx}]" if not cond else ""))
    if not cond: fails.append(label)

print("\n--- int-slot negative control ---")
ck("report generated despite a non-list `subchecks` (int) in the fixture",
   _crash is None, _crash or "")
if _crash:
    print("\n"+f"REGRESSION: _expand no longer guards `subchecks` -- {_crash}")
    print(f"{len(fails)} FAILURES: {fails}")
    sys.exit(1)

out = open(m.OUT_PATH).read()
ck("int-slot check still counted as its own row (parent not dropped)",
   "| `tile_coverage_softmax_positivity` | 1 | 1 | 100% | 0 | 0 | 0 |" in out)
ck("no phantom sub-rows synthesised from the int slot",
   "tile_coverage_softmax_positivity[" not in out)

print("\n--- hand-derived expectations ---")
ck("perturbation_tolerance: ran 5, caught 2, rate 40%",
   "| `perturbation_tolerance` | 5 | 2 | 40% | 0 | 0 | 0 |" in out)
ck("weight_magnitude: ran 5, caught 2", "| `weight_magnitude` | 5 | 2 | 40% |" in out)
ck("cross_shape: ran 5, caught 1, 1 FP on reference",
   "| `cross_shape` | 5 | 1 | 20% | 0 | 0 | 1 |" in out)
ck("output_shape: ran 4 (error excluded), caught 0, errors 1",
   "| `output_shape` | 4 | 0 | 0% | 1 | 0 | 0 |" in out)
ck("subcheck cross_shape[shape=(1,512)] attributed separately",
   "`cross_shape[shape=(1,512)]`" in out)
ck("ERROR not counted as a catch (output_shape caught=0)",
   "| `output_shape` | 4 | 0 |" in out)
ck("backward_pass reported as never ran (roster, absent from data)",
   "never ran" in out and "backward_pass" in out)
ck("output_shape flagged 'ran but never caught'",
   "Ran but never caught anything" in out)
ck("redundancy: perturbation_tolerance vs weight_magnitude = identical",
   "identical" in out)
ck("consistency MISMATCH flagged for the crash-as-catch mutant",
   "MISMATCH" in out and "l1norm/partial_reduction" in out)
ck("layer3: batchnorm listed as having no properties",
   "no algebraic properties" in out and "`batchnorm`" in out)
ck("layer3 rollup present", "rolled up by property name" in out)
ck("layer3: rows_sum_to_one caught 1", "| softmax | `rows_sum_to_one` | 2 | 1 | 0 |" in out)

print("\n"+("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}"))
if fails:
    print("\n--- generated report ---\n"+out)
sys.exit(1 if fails else 0)
