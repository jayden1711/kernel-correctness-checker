"""
forkserver — the inherited-RNG hazard, and the seeding that neutralises it.

A `forkserver` child is forked from a torch-preloaded server, so it inherits that
server's random-number generator state. For the BATCHED path that is harmless,
because the child re-seeds from the proposal id before drawing anything. That is
the whole safety argument for using forkserver here, and it rests on two facts
that no existing test could see:

  1. the seed is applied at all, and
  2. it is applied BEFORE the first draw.

Lose either and the failure is silent and severe: every proposal in the run draws
the SAME tensors. Not "unseeded" -- identically seeded. Nothing raises, no
verdict looks malformed, and a search comparing one proposal against itself would
report perfect reproducibility while having collapsed its own input diversity.

--------------------------------------------------------------------------
WHY THIS IS NOT A SOURCE-TEXT CHECK
--------------------------------------------------------------------------
`check_batch_executor.py` already asserts `manual_seed` is ABSENT from the
single-kernel path, by searching the source. That direction is fine: absence is
what it needs, and absence is what a text search can establish.

The property here is the opposite and text cannot reach it. `manual_seed`
appearing in the source says nothing about whether it runs before the draw, or
whether the value derives from the proposal id. So this drives the real function
with a generator whose state is READ AT DRAW TIME, and observes what the draw
actually depended on.

--------------------------------------------------------------------------
EVERY POSITIVE ASSERTION HERE IS PAIRED WITH A MUTATION THAT MUST BREAK IT
--------------------------------------------------------------------------
"Two proposals drew differently" is unfalsifiable on its own -- a harness too
lossy to reproduce the bug prints exactly the same line (§5 instances 9 and 11).
So each claim is re-run against a mutated copy of the function that reintroduces
the defect, and the mutation is REQUIRED to flip the result. A control that does
not fire fails this script, and so does an anchor string that no longer matches
the source, so a refactor cannot quietly disarm it.

--------------------------------------------------------------------------
THE `check_*.py` FILENAME IS LOAD-BEARING. DO NOT RENAME THIS FILE.
--------------------------------------------------------------------------
tests/pytest.ini sets `python_files = test_*.py`, so `check_*.py` is never
collected. This script stubs `torch` at module scope and would corrupt
tests/verification/* if collected into the same process. See the README here.

Run:  python3 tests/instrumentation/check_forkserver_executor.py
Exit 0 = pass.  No GPU, no real torch, no network.
"""
import queue
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

fails = []


def ck(label, cond, ctx=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   [{ctx}]" if not cond else ""))
    if not cond:
        fails.append(label)


# ── a torch stub whose generator state is REAL ────────────────────────────────
#
# The point of the stub is that a draw is a function of the state AT THE MOMENT
# IT HAPPENS. A stub that merely recorded seed values could not tell "seeded then
# drew" from "drew then seeded", which is one of the two failure modes here.

class _Rng:
    def __init__(self):
        self.state = 0
        self.events = []          # ("seed", v) / ("draw", v), in order

    def manual_seed(self, s):
        self.state = int(s)
        self.events.append(("seed", int(s)))

    def draw(self):
        # Any state-dependent, state-advancing function will do; this is an LCG.
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % (2 ** 61 - 1)
        self.events.append(("draw", self.state))
        return self.state


_rng = _Rng()


class FakeTensor:
    """Carries the draw that produced it, so provenance survives into the test."""

    def __init__(self, value=0.0):
        self.value = value

    def clone(self):
        return FakeTensor(self.value)

    def float(self):
        return self


class _FakeCuda:
    def is_available(self):
        return False

    def synchronize(self):
        pass

    def manual_seed_all(self, seed):
        _rng.manual_seed(seed)


_torch = types.ModuleType("torch")
_torch.cuda = _FakeCuda()
_torch.manual_seed = _rng.manual_seed
_torch.zeros = lambda *a, **k: FakeTensor()
_torch.allclose = lambda a, b, **k: True
_torch.Tensor = FakeTensor
for _dt in ("float32", "float16", "bfloat16", "int32", "int64"):
    setattr(_torch, _dt, _dt)
sys.modules["torch"] = _torch

from verification.adversarial_search import executor as EX                     # noqa: E402
from verification.adversarial_search.schemas import (                          # noqa: E402
    InputProposal, KernelExecutionResult, TensorDescriptor)

SRC = (REPO / "verification/adversarial_search/executor.py").read_text()
REF_PATH = "/kernels/softmax/reference.py"

# What a child would inherit from the forkserver: one state, identical in every
# fork, and NOT derived from any proposal.
FORKSERVER_INHERITED_STATE = 99999999


def mk_proposal(pid):
    return InputProposal(
        proposal_id=pid, worker_id="w0", iteration=0, operator="softmax",
        tensors={"x": TensorDescriptor(
            shape=[64, 64], dtype="float32", fill="randn")},
        rationale="r", predicted_failure_mode="p",
    )


def mk_result(pid, kid):
    return KernelExecutionResult(
        proposal_id=pid, kernel_id=kid, passed_checker=True, passed_naive=True,
        error=None, check_results=[], wall_time_ms=1.0)


KS = [("reference", REF_PATH), ("mut_a", "/kernels/softmax/a.py")]


class _FakeChildCtx:
    def load_fn(self, path, op):
        return lambda *a, **k: FakeTensor()

    SPEC_MAP = {"softmax": (lambda: "SPEC")}
    KernelChecker = staticmethod(lambda spec: None)


def run_child(fn, pid):
    """Drive one child body and report the draw its inputs came from.

    `fn` is the function under test -- either the real
    `_run_batch_in_subprocess` or a deliberately broken copy of it.

    THE STUBS GO INTO `fn.__globals__`, NOT ONTO THE MODULE. For the real
    function those are the same dict, but a mutant compiled into a COPY of the
    module namespace would otherwise keep resolving `materialize_proposal` to
    the real one, raise inside the child's setup handler, and never draw at all.
    Both negative controls below did exactly that on their first run: they
    reported the mutant's draws as equal, which is what a fired control looks
    like, when in truth neither mutant had drawn anything and the comparison was
    `None == None`. §5 instance 5, reproduced live -- which is why every control
    here is paired with an assertion about WHAT it detected.
    """
    q = queue.Queue()
    drawn = []

    def fake_materialize(proposal, device="cuda"):
        # The draw happens HERE, reading whatever generator state exists now.
        v = _rng.draw()
        drawn.append(v)
        return {"x": FakeTensor(value=v)}

    def fake_eval(proposal_id, kernel_id, cand, ref, inputs, spec, checker):
        return mk_result(proposal_id, kernel_id)

    g = fn.__globals__
    patches = {
        "_ChildContext": lambda: _FakeChildCtx(),
        "materialize_proposal": fake_materialize,
        "_evaluate_kernel": fake_eval,
        "tensors_to_inputs": lambda op, tensors: tensors["x"],
    }
    saved = {k: g[k] for k in patches if k in g}
    g.update(patches)

    # Every child starts from the SAME inherited state -- that is what forking
    # off one forkserver means, and it is the condition the seeding must survive.
    _rng.state = FORKSERVER_INHERITED_STATE
    _rng.events = []
    err = []
    try:
        fn(mk_proposal(pid).to_dict(), KS, REF_PATH, "softmax", q, time.time())
    finally:
        g.update(saved)

    # A child that failed setup reports the cause on every kernel. Surface it:
    # a mutant that never reached the draw would otherwise be indistinguishable
    # from one that drew, and its control would pass on `None == None`.
    while not q.empty():
        payload = q.get()
        e = payload.get("error") if isinstance(payload, dict) else None
        if e:
            err.append(e.get("error_type"))

    return drawn[0] if drawn else None, list(_rng.events), err


def mutate(anchor, replacement, label):
    """Compile a copy of `_run_batch_in_subprocess` with `anchor` replaced.

    The anchor must still match the current source. A control whose anchor has
    silently stopped matching is a control that no longer tests anything, and
    this project has shipped one of those before (§5 instance 4).
    """
    start = SRC.index("def _run_batch_in_subprocess")
    end = SRC.index("def execute_proposal_batch")
    body = SRC[start:end]
    if anchor not in body:
        ck(f"CONTROL ANCHOR still matches the source ({label})", False, anchor[:60])
        return None
    ns = dict(EX.__dict__)
    exec(compile(body.replace(anchor, replacement), "<mutant>", "exec"), ns)
    return ns["_run_batch_in_subprocess"]


# ══ 1. The real function: the inherited state must not reach the draw ════════

print("\n── 1. seeding survives the fork ──")

draw_a, events_a, err_a = run_child(EX._run_batch_in_subprocess, "prop-AAAA")
draw_b, events_b, err_b = run_child(EX._run_batch_in_subprocess, "prop-BBBB")
draw_a2, _, _ = run_child(EX._run_batch_in_subprocess, "prop-AAAA")

ck("a draw actually happened and the child did not error out -- otherwise "
   "everything below compares None against None",
   draw_a is not None and draw_b is not None and not err_a and not err_b,
   f"{draw_a} {draw_b} errors={err_a + err_b}")
ck("two proposals forked from the SAME inherited state draw DIFFERENTLY",
   draw_a != draw_b, f"{draw_a} vs {draw_b}")
ck("the same proposal draws identically every time (resume-safe, replay-safe)",
   draw_a == draw_a2, f"{draw_a} vs {draw_a2}")

kinds = [k for k, _ in events_a]
ck("the seed is applied BEFORE the first draw -- a seed applied after it would "
   "leave the inherited state determining the tensors",
   "seed" in kinds and "draw" in kinds
   and kinds.index("seed") < kinds.index("draw"), str(kinds))

seeded = [v for k, v in events_a if k == "seed"]
ck("and the seed is derived from the PROPOSAL ID, not from a counter or the "
   "clock, so it is stable across resumes and worker interleaving",
   seeded and seeded[0] == EX._seed_for("prop-AAAA"),
   f"{seeded} vs {EX._seed_for('prop-AAAA')}")


# ══ 2. Negative controls: reintroduce the defect, require the result to flip ══

print("\n── 2. negative controls (each MUST fire) ──")

# (a) The seed is never applied -- the literal forkserver hazard.
no_seed = mutate("torch.manual_seed(seed)", "pass  # MUTANT: seed dropped",
                 "seed dropped")
if no_seed is not None:
    m_a, _, m_err = run_child(no_seed, "prop-AAAA")
    m_b, _, _ = run_child(no_seed, "prop-BBBB")
    ck("CONTROL PRECONDITION: the mutant actually REACHED the draw. Without "
       "this the next assertion passes on None == None, which is what it did "
       "on this script's first run",
       m_a is not None and not m_err, f"{m_a} errors={m_err}")
    ck("CONTROL: with the seed dropped, two DIFFERENT proposals draw the SAME "
       "tensors -- this is the silent failure forkserver would cause, and the "
       "test above is only meaningful because this one fires",
       m_a == m_b, f"{m_a} vs {m_b}")
    ck("CONTROL detected the RIGHT thing: the shared draw is the one the "
       "inherited state produces, not merely some other coincidence",
       m_a == (FORKSERVER_INHERITED_STATE * 6364136223846793005
               + 1442695040888963407) % (2 ** 61 - 1),
       str(m_a))

# (b) The seed is applied, but too late. Distinct from (a): the source still
# contains `manual_seed`, derived from the right id, so every text-based check
# passes while the tensors are still the inherited ones.
late_seed = mutate(
    "        seed = _seed_for(proposal.proposal_id)\n"
    "        torch.manual_seed(seed)",
    "        seed = _seed_for(proposal.proposal_id)\n"
    "        _LATE = seed  # MUTANT: seeding moved after materialization",
    "seed applied late")
if late_seed is not None:
    # Re-apply the seed after materialization, mimicking a reordering.
    src_start = SRC.index("def _run_batch_in_subprocess")
    src_end = SRC.index("def execute_proposal_batch")
    reordered = SRC[src_start:src_end].replace(
        "        seed = _seed_for(proposal.proposal_id)\n"
        "        torch.manual_seed(seed)",
        "        seed = _seed_for(proposal.proposal_id)").replace(
        "        reference_fn = ctx.load_fn(reference_src_path, operator)",
        "        torch.manual_seed(seed)  # MUTANT: too late\n"
        "        reference_fn = ctx.load_fn(reference_src_path, operator)")
    ns = dict(EX.__dict__)
    exec(compile(reordered, "<mutant-late>", "exec"), ns)
    l_a, l_events, l_err = run_child(ns["_run_batch_in_subprocess"], "prop-AAAA")
    l_b, _, _ = run_child(ns["_run_batch_in_subprocess"], "prop-BBBB")
    l_kinds = [k for k, _ in l_events]
    ck("CONTROL PRECONDITION: the reordered mutant reached the draw",
       l_a is not None and not l_err, f"{l_a} errors={l_err}")
    ck("CONTROL: a seed applied AFTER materialization still leaves both "
       "proposals on the inherited draw -- text-level checks cannot see this, "
       "which is why the ordering assertion above is a runtime one",
       l_a == l_b, f"{l_a} vs {l_b}")
    ck("CONTROL detected the RIGHT thing: the seed did run, just too late",
       "seed" in l_kinds and l_kinds.index("draw") < l_kinds.index("seed"),
       str(l_kinds))


# ══ 3. The preload list, which is what keeps the stamps honest ═══════════════

print("\n── 3. the preload list stays narrow ──")

ck("_FORKSERVER_PRELOAD is exactly ['torch']",
   EX._FORKSERVER_PRELOAD == ["torch"], str(EX._FORKSERVER_PRELOAD))
ck("this module is NOT preloaded, so each child re-imports it and its own "
   "import stamps stay this execution's -- torch_import_ms then collapses "
   "toward zero rather than reporting the forkserver's one-time cost forever",
   not any("executor" in m or "verification" in m
           for m in EX._FORKSERVER_PRELOAD), str(EX._FORKSERVER_PRELOAD))

# The guard that catches it if somebody widens the list anyway.
real_pid = EX._MODULE_IMPORT_PID
try:
    EX._MODULE_IMPORT_PID = -1                     # pretend: stamped elsewhere
    inherited = EX._startup_phases(time.time())
finally:
    EX._MODULE_IMPORT_PID = real_pid
own = EX._startup_phases(time.time())

ck("CONTROL: stamps taken in ANOTHER process are renamed, never reported as "
   "this execution's torch_import_ms",
   "torch_import_ms" not in inherited
   and "startup_stamps_inherited_ms" in inherited, str(sorted(inherited)))
ck("CONTROL: and no negative pre_module_ms is emitted -- the arithmetic that "
   "produces one is simply not reached",
   "pre_module_ms" not in inherited, str(inherited))
ck("CONTROL IS NOT VACUOUS: stamps taken in THIS process keep the original "
   "key names, so the spawn and forkserver arms stay directly comparable",
   "torch_import_ms" in own and "pre_module_ms" in own, str(sorted(own)))
# The sign here is an artifact of the harness, not a production expectation:
# this call passes a spawn time LATER than the module import, which is the
# reverse of a real child. It is asserted only to prove the pre_module_ms
# arithmetic was actually evaluated on this branch rather than skipped.
ck("and that branch really did compute pre_module_ms rather than defaulting it",
   own["pre_module_ms"] < 0, str(own["pre_module_ms"]))


# ══ 4. The delegation-ratio instrumentation (item 1d) ════════════════════════

print("\n── 4. the delegation detector reports its ratio on BOTH outcomes ──")

# `check_kernel_executed`'s delegation detector is a pure timing comparison, and
# for a REFERENCE kernel it times one function against itself. It was measured
# at p50 0.92 / p99 11.45 across 560 executions -- i.e. its 10x threshold sits at
# about the 98th percentile of its own noise, which is item 1d. That measurement
# is only possible because the ratio is now emitted on the PASS path too; before,
# it was computed and discarded unless it tripped, so the only observable samples
# were the ones above the threshold. Losing that again would silently make the
# defect unmeasurable while leaving it entirely present.

GUARD = (REPO / "verification/layer1_structural/runtime_guards.py").read_text()
ke = GUARD[GUARD.index("def check_kernel_executed"):]

ck("the ratio is computed once and reused, not recomputed per branch",
   ke.count("ratio = (t_ref / t_cand)") == 1, "expected a single assignment")
ck("it is emitted on the FAIL path (the delegating/ghost verdict)",
   ke.count("delegation_ratio=") >= 2
   and "Likely delegating to reference" in ke)
ck("and on the PASS path -- without this only above-threshold samples are "
   "observable and the false-positive rate cannot be measured at all",
   "verdict_detail += f\" [delegation_ratio=" in ke)
ck("the division is guarded, so a zero-time measurement reports inf rather "
   "than raising ZeroDivisionError inside a Layer-1 check",
   "if t_cand > 0 else float(\"inf\")" in ke)
ck("the THRESHOLD itself is unchanged -- this instrumentation must not move "
   "the verdict it is measuring",
   "t_cand < t_ref * 0.1" in ke)

# The token has to survive a round trip through the reader that consumes these
# strings, or the measurement is unparseable in exactly the runs that matter.
import re as _re                                                              # noqa: E402
_RATIO_RE = _re.compile(r"delegation_ratio=([0-9.eE+\-]+|inf)")
for sample, want in (
        ("Kernel executed and produced input-dependent output "
         "(perturbation: negation). [delegation_ratio=1.0431]", "1.0431"),
        ("Output is bit-identical to reference AND candidate is 11.3x faster. "
         "Likely delegating to reference. [delegation_ratio=11.2984]", "11.2984"),
        ("Kernel raised an exception: too many values to unpack", None)):
    got = _RATIO_RE.search(sample)
    ck(f"ratio parses out of {'a verdict' if want else 'a crash'} detail string",
       (got.group(1) if got else None) == want, str(got))


# ══ 5. Item 1d's fix: interleaved best-of-N timing ═══════════════════════════

print("\n── 5. the timing estimator must survive a scheduling stall ──")

ck("the two measurements are INTERLEAVED, not two sequential blocks",
   "for _ in range(_ROUNDS):" in ke
   and "t_cand = min(t_cand, _time_once(candidate_fn))" in ke
   and "t_ref = min(t_ref, _time_once(reference_fn))" in ke)
ck("the estimator is MIN across rounds -- contention only ever ADDS time, so "
   "the minimum is the sample a stall cannot inflate",
   't_cand = float("inf")' in ke and 't_ref = float("inf")' in ke)
ck("total kernel launches per side are unchanged (10), so this is a better "
   "sample of the same work rather than more work",
   "_ROUNDS, _CALLS = 5, 2" in ke)
ck("the THRESHOLD is untouched, so any verdict change is attributable to the "
   "estimator and not to a moved goalpost",
   "t_cand < t_ref * 0.1" in ke)

# FUNCTIONAL CONTROL. The claim is "a single stall can no longer decide the
# verdict". That is a claim about behaviour, so it is tested by injecting a
# stall and running BOTH estimators over the identical sequence of call costs.
# A source-text check cannot distinguish an estimator that resists stalls from
# one that merely looks like it does.

def _old_estimator(costs_c, costs_r):
    """10 candidate calls summed, then 10 reference calls summed."""
    return sum(costs_r) / sum(costs_c)

def _new_estimator(costs_c, costs_r, rounds=5, calls=2):
    """Interleaved rounds, min across rounds -- the shipped construction."""
    tc = tr = float("inf")
    for i in range(rounds):
        tc = min(tc, sum(costs_c[i * calls:(i + 1) * calls]))
        tr = min(tr, sum(costs_r[i * calls:(i + 1) * calls]))
    return tr / tc

NOMINAL = 1.0
STALL = 100.0
old_fires = new_fires = 0
for pos in range(10):                       # stall in each possible slot
    cc = [NOMINAL] * 10
    cr = [NOMINAL] * 10
    cr[pos] += STALL                        # the stall lands on the reference
    if _old_estimator(cc, cr) >= 10.0:
        old_fires += 1
    if _new_estimator(cc, cr) >= 10.0:
        new_fires += 1

ck("CONTROL: the OLD estimator false-positives on a single stall, in every "
   "slot it can land in -- without this the next assertion is unfalsifiable",
   old_fires == 10, f"old fired {old_fires}/10")
ck("the NEW estimator survives the same stall in every slot",
   new_fires == 0, f"new fired {new_fires}/10")

# ...and it must still CATCH a genuinely fast candidate, or the fix has simply
# disabled the check. A precomputed-output ghost is uniformly fast, not
# occasionally fast, so min-over-rounds sees it just as clearly.
ghost_c = [0.01] * 10
ghost_r = [1.0] * 10
ck("CONTROL IS NOT VACUOUS: a uniformly 100x-faster candidate is still caught "
   "by the new estimator, so this is noise rejection and not a disabled check",
   _new_estimator(ghost_c, ghost_r) >= 10.0,
   f"ratio {_new_estimator(ghost_c, ghost_r):.1f}")
ck("and a candidate that is merely EQUAL in speed is still not flagged",
   _new_estimator([1.0] * 10, [1.0] * 10) < 10.0)


# ── result ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
if fails:
    print(f"  {len(fails)} FAILED:")
    for f in fails:
        print(f"    - {f}")
    sys.exit(1)
print("  all checks passed")
sys.exit(0)
