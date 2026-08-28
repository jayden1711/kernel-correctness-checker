"""
Item #2 instrumentation: per-check attribution records.

Verifies the four-valued outcome mapping, subcheck passthrough, and -- the
critical guarantee -- that adding attribution left every pre-existing harness
output bit-for-bit unchanged.

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
    python3 tests/instrumentation/check_item2_instrumentation.py

Plain python3 -- no venv, no numpy, no torch, no pytest. See the README in this
directory for the full rationale.

Exit code 0 = all assertions passed. Non-zero = failures, listed on stdout.
"""
import sys, types, json, statistics, copy
import os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

FILES = str(REPO / "benchmarks" / "autokernel" / "files")

# ---- stub every heavy import so checker_adapter/harness are importable ----
def mod(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m

_np = mod("numpy"); _np.mean = lambda x: statistics.fmean(x) if x else 0.0
_np.random = types.SimpleNamespace(default_rng=lambda s=0: None)
mod("torch")
mod("verification"); mod("verification.layer1_structural"); mod("verification.layer2_numeric_oracle")
mod("verification.checker", KernelChecker=object, _check_cross_shape=None, _check_exact_match=None)
mod("verification.layer1_structural.ast_analysis", check_ghost_optimization=None,
    check_partial_computation=None, check_timing_manipulation=None)
mod("verification.layer1_structural.runtime_guards", check_determinism=None,
    check_dtype_preserved=None, check_kernel_executed=None, check_nan_inf=None)
mod("verification.layer1_structural.tile_coverage", check_all_tiles_visited=None,
    check_all_tiles_visited_generic=None)
mod("verification.layer2_numeric_oracle.perturbation", check_perturbation_tolerance=None)
mod("verification.layer2_numeric_oracle.shape_generalization", check_backward_pass=None,
    check_output_shape=None, check_weight_magnitude=None)
mod("baselines", allclose_gate=None, autokernel_gate=None)

sys.path.insert(0, FILES)
import checker_adapter as ca
import harness as H

fails = []
def check(label, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   {extra}" if extra and not cond else ""))
    if not cond: fails.append(label)

# ================= 1. _try four-valued outcomes =================
print("\n[1] _try outcome mapping (passed/detail keep LEGACY semantics)")
cases = [
    ("returns True",              lambda: True,                    True,  "pass",  None),
    ("returns False",             lambda: False,                   False, "fail",  None),
    ("returns None (SKIP)",       lambda: None,                    False, "skip",  None),
    ("(True,'ok')",               lambda: (True, "ok"),            True,  "pass",  "ok"),
    ("(False,'bad')",             lambda: (False, "bad"),          False, "fail",  "bad"),
    ("(None,'skipped ...')",      lambda: (None, "skipped x"),     False, "skip",  "skipped x"),
    ("raises",                    lambda: (_ for _ in ()).throw(ValueError("boom")), False, "error", "ValueError: boom"),
]
for label, fn, exp_passed, exp_outcome, exp_detail in cases:
    name, passed, detail, rec = ca._try(label, fn)
    check(f"{label}: passed={exp_passed}", passed == exp_passed, f"got {passed}")
    check(f"{label}: outcome={exp_outcome}", rec["outcome"] == exp_outcome, f"got {rec['outcome']}")
    check(f"{label}: detail preserved", detail == exp_detail, f"got {detail!r}")

# the two legacy-semantics guarantees, stated explicitly
_, p_skip, _, r_skip = ca._try("s", lambda: (None, "skipped"))
check("SKIP still yields legacy passed=False (verdict unchanged)", p_skip is False)
check("SKIP labelled 'skip', NOT 'fail' (not counted as a catch)", r_skip["outcome"] == "skip")
_, p_err, _, r_err = ca._try("e", lambda: (_ for _ in ()).throw(RuntimeError("x")))
check("ERROR still yields legacy passed=False (verdict unchanged)", p_err is False)
check("ERROR labelled 'error', NOT 'fail' (not counted as a catch)", r_err["outcome"] == "error")

# ---- per-check duration (#7a step 2) ----------------------------------------
# NEGATIVE CONTROL: a check that sleeps a known interval must report a
# duration_ms in that range. A duration field that always reads 0.0 -- or that
# silently stops being populated -- would look perfectly healthy in the ablation
# table while making every check appear free, which is precisely the class of
# "instrument reports nothing and verifies nothing" failure logged in
# SESSION_HANDOFF.md §5 (instances 5, 6 and 8). Verified to fail when the timing
# is removed from _try.
import time as _time
_SLEEP_S = 0.05
_, _, _, r_slow = ca._try("slow", lambda: (_time.sleep(_SLEEP_S), True)[1])
check("duration_ms present on a timed check", "duration_ms" in r_slow)
check(f"duration_ms reflects real elapsed time (>= {_SLEEP_S*1000:.0f}ms)",
      r_slow["duration_ms"] is not None and r_slow["duration_ms"] >= _SLEEP_S * 1000 * 0.9,
      f"got {r_slow['duration_ms']}")
check("duration_ms is not absurdly large (sane upper bound)",
      r_slow["duration_ms"] < _SLEEP_S * 1000 * 20, f"got {r_slow['duration_ms']}")
_, _, _, r_fast = ca._try("fast", lambda: True)
check("a trivial check is measurably cheaper than the sleeping one",
      r_fast["duration_ms"] < r_slow["duration_ms"])
# An ERROR is still timed -- a check that raises after 3s has a very different
# cost profile from one that raises immediately.
_, _, _, r_errt = ca._try("et", lambda: (_ for _ in ()).throw(RuntimeError("x")))
check("errors carry a duration too (not left unmeasured)",
      r_errt["duration_ms"] is not None)
# _record() never executed a check: None, NOT 0.0. "never ran" and "ran
# instantly" must stay distinguishable, same reason outcome is four-valued.
_, _, _, r_norun = ca._record("nr", False, "d", "skip")
check("_record() (never ran) reports duration_ms None, not 0.0",
      r_norun["duration_ms"] is None, f"got {r_norun['duration_ms']}")

# ================= 2. subchecks passthrough =================
print("\n[2] compound-check subcheck passthrough")
subs = [{"name": "shape=(1,512)", "outcome": "fail", "detail": "max_err=0.5"}]
_, _, _, rec = ca._try("cross_shape", lambda: (False, "Cross-shape failures: a; b", subs))
check("3rd element captured as subchecks", rec["subchecks"] == subs, f"got {rec['subchecks']}")
_, _, _, rec2 = ca._try("plain", lambda: (True, "ok"))
check("2-tuple leaves subchecks None", rec2["subchecks"] is None)

# ================= 3. _summarize string is byte-identical =================
print("\n[3] _summarize joined detail string unchanged vs legacy")
def legacy_summarize(checks3):
    failed = [c for c in checks3 if not c[1]]
    if failed:
        return False, "; ".join(f"{n}: {d}" for n, _, d in failed)
    return True, None

for scenario in (
    [("a", lambda: (True, "ok")), ("b", lambda: (False, "bad")), ("c", lambda: (None, "skipped"))],
    [("a", lambda: (True, "ok"))],
    [("x", lambda: (False, "Cross-shape failures: shape=(1,2): e; shape=(3,4): f"))],
):
    new_checks = [ca._try(n, f) for n, f in scenario]
    passed, detail, records = ca._summarize(new_checks)
    lp, ld = legacy_summarize([(c[0], c[1], c[2]) for c in new_checks])
    check(f"verdict identical ({len(scenario)} checks)", passed == lp, f"{passed} vs {lp}")
    check(f"detail string identical ({len(scenario)} checks)", detail == ld, f"{detail!r} vs {ld!r}")
    check(f"one record per check ({len(scenario)})", len(records) == len(scenario))

# detail containing the separator must not be re-split
_, d, _ = ca._summarize([ca._try("cross_shape", lambda: (False, "Cross-shape failures: a; b"))])
check("separator inside a detail is preserved verbatim", d == "cross_shape: Cross-shape failures: a; b", repr(d))

# ================= 4. summarize() regression: check_records is inert =========
print("\n[4] harness.summarize() unaffected by the new check_records key")
base = {"sys": {
    "mutant_results": [
        {"op": "softmax", "mutant": "first_tile", "caught": True,  "detail": "x"},
        {"op": "softmax", "mutant": "wrong_red",  "caught": False, "detail": None},
        {"op": "matmul",  "mutant": "partial_k",  "caught": True,  "detail": "y"},
    ],
    "ref_results": [
        {"op": "softmax", "false_positive": False, "detail": None},
        {"op": "matmul",  "false_positive": True,  "detail": "boom"},
    ],
    "latencies": [0.001, 0.002, 0.003],
}}
withrec = copy.deepcopy(base)
for r in withrec["sys"]["mutant_results"]:
    r["check_records"] = [{"name": "perturbation_tolerance", "outcome": "fail",
                           "detail": None, "subchecks": None}]
for r in withrec["sys"]["ref_results"]:
    r["check_records"] = [{"name": "cross_shape", "outcome": "pass",
                           "detail": None, "subchecks": None}]
a = json.dumps(H.summarize(base), sort_keys=True, default=str)
b = json.dumps(H.summarize(withrec), sort_keys=True, default=str)
check("summarize() output byte-identical with/without check_records", a == b)
s = H.summarize(withrec)["sys"]
check("catch_rate preserved", abs(s["catch_rate"] - 2/3) < 1e-9, s["catch_rate"])
check("false_positive_rate preserved", abs(s["false_positive_rate"] - 0.5) < 1e-9)
check("missed_mutants preserved", s["missed_mutants"] == ["softmax/wrong_red"], s["missed_mutants"])

# ================= 5. _call accepts 3- and 4-tuple systems =================
print("\n[5] harness._call backward compatibility")
p, dt, det, rec = H._call(lambda e, m, r: (True, 0.5, "d"), None, True, None)
check("legacy 3-tuple system -> records None", (p, dt, det, rec) == (True, 0.5, "d", None))
p, dt, det, rec = H._call(lambda e, m, r: (False, 0.1, "d", [{"name": "c"}]), None, True, None)
check("4-tuple system -> records carried", rec == [{"name": "c"}])

# ================= 6. _warm() must not perturb the RNG =================
print("\n[6] harness._warm() is RNG-neutral (catch rates cannot move)")
# This is the safety property of the cache-warming change: warming calls each
# system twice more, and those calls consume rng draws exactly as the timed
# calls do. If the bit-generator state is not restored, every subsequent input
# in the run differs and CATCH RATES MOVE -- a latency fix silently becoming a
# correctness change. Verified here against a recording fake; numpy's own
# state round-trip fidelity was verified separately against REAL numpy 2.5.1
# (exact for both getter and setter), which is the claim this fake cannot make.
class _FakeBitGen:
    def __init__(self): self.counter = 0
    @property
    def state(self): return {"counter": self.counter}
    @state.setter
    def state(self, s): self.counter = s["counter"]

class _FakeRng:
    def __init__(self): self.bit_generator = _FakeBitGen()
    def draw(self):
        self.bit_generator.counter += 1
        return self.bit_generator.counter

def _consuming_system(entry, is_mutant, rng):
    rng.draw()                      # mirrors input_fn(rng) consuming state
    return True, 0.0, None

def _drive(rng, n=6):
    """Call the system n times, recording the COUNTER after each call.

    Written as an explicit loop, not `system(...) or rng.counter`: the system
    returns a truthy tuple, so `or` short-circuits and records the tuple
    instead of the counter -- which makes every sequence trivially equal and
    the comparison below vacuous. That mistake was made here once and caught
    by the negative control further down, which is the whole reason it exists.
    """
    out = []
    for _ in range(n):
        _consuming_system(None, False, rng)
        out.append(rng.bit_generator.counter)
    return out

_rng_plain = _FakeRng()
_seq_plain = _drive(_rng_plain)
_rng_warm = _FakeRng()
H._warm(_consuming_system, None, _rng_warm)
_seq_warm = _drive(_rng_warm)
check("warmup leaves the draw sequence byte-identical", _seq_plain == _seq_warm,
      f"{_seq_plain} vs {_seq_warm}")
check("warmup consumed draws then rewound them (state back to start)",
      _seq_warm[0] == 1, f"first post-warmup draw was {_seq_warm[0]}, expected 1")

# NEGATIVE CONTROL: without the restore the sequence must move. If this ever
# passes, _warm's snapshot/restore has stopped doing anything and the guard
# above is vacuous.
_rng_bad = _FakeRng()
try:
    _consuming_system(None, True, _rng_bad); _consuming_system(None, False, _rng_bad)
except Exception:
    pass
_seq_bad = _drive(_rng_bad)
check("control: WITHOUT restore the sequence does move (guard is not vacuous)",
      _seq_bad != _seq_plain, f"{_seq_bad} vs {_seq_plain}")

# A system that raises during warmup must not abort the run -- it will raise
# identically when timed, where the harness already models it as `error`.
def _raising_system(entry, is_mutant, rng):
    raise RuntimeError("boom")
_rng_raise = _FakeRng()
_raised = False
try:
    H._warm(_raising_system, None, _rng_raise)
except Exception:
    _raised = True
check("warmup swallows system exceptions (run not aborted)", not _raised)
check("state still restored after a raising warmup",
      _rng_raise.bit_generator.counter == 0,
      f"counter={_rng_raise.bit_generator.counter}")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}"))
sys.exit(1 if fails else 0)
