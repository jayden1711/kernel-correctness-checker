"""
Item 2 — one subprocess per PROPOSAL instead of one per kernel.

Startup (interpreter + `import torch`/triton + CUDA init) was 71% of adversarial
search wall time -- 394.7s of the 556.2s causal_flash_attention run -- because it
was paid once per (proposal, kernel) pair. `execute_proposal_batch` pays it once
per proposal instead.

This verifies the parts of that change that can go silently wrong:

  * the PARENT's drain loop (deadlock, ordering, deadlines, exit grace)
  * the FALLBACK, which must fire when a batch cannot finish and must NOT fire
    when it can
  * the CHILD's one-materialisation-many-clones contract
  * the poisoned-CUDA-context guard, whose absence would fabricate "the next
    mutant crashed" for kernels that never ran

--------------------------------------------------------------------------
THE `check_*.py` FILENAME IS LOAD-BEARING. DO NOT RENAME THIS FILE.
--------------------------------------------------------------------------
tests/pytest.ini sets `python_files = test_*.py`, so `check_*.py` is never
collected by pytest. This script stubs `torch` at module scope and would corrupt
tests/verification/* if collected into the same process. See the README here.

Run:  python3 tests/instrumentation/check_batch_executor.py
Exit 0 = pass.

WHY IT CAN RUN WITHOUT A GPU, OR EVEN A REAL TORCH: every defect it guards is a
CONTROL-FLOW defect -- who gets re-run, in what order, how many times something
was materialised, whether a sentinel was honoured. None of them is numerical. A
stub that records calls answers those more directly than a real run would, and
the numerical behaviour of the per-kernel body is unchanged by this work: it is
the same `_evaluate_kernel` on both paths, which is itself asserted below.
"""
import os
import queue
import sys
import threading
import time
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

fails = []


def ck(label, cond, ctx=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   [{ctx}]" if not cond else ""))
    if not cond:
        fails.append(label)


# ── torch stub, installed BEFORE executor is imported ─────────────────────────

class FakeTensor:
    """Records its own provenance so a clone can be told from its original."""
    _next_id = [0]

    def __init__(self, value=0.0, origin=None):
        FakeTensor._next_id[0] += 1
        self.id = FakeTensor._next_id[0]
        self.value = value
        self.origin = origin if origin is not None else self.id

    def clone(self):
        return FakeTensor(self.value, origin=self.origin)

    def float(self):
        return self


class _FakeCuda:
    def __init__(self):
        self.available = False
        self.sync_raises_after = None      # kernel_id after which synchronize() blows up
        self.sync_calls = []
        self._armed = False

    def is_available(self):
        return self.available

    def synchronize(self):
        self.sync_calls.append(True)
        if self._armed:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    def manual_seed_all(self, seed):
        pass


_fake_cuda = _FakeCuda()
_torch = types.ModuleType("torch")
_torch.cuda = _fake_cuda
_torch.seeds = []
_torch.manual_seed = lambda s: _torch.seeds.append(s)
_torch.zeros = lambda *a, **k: FakeTensor()
_torch.allclose = lambda a, b, **k: True
_torch.Tensor = FakeTensor
for _dt in ("float32", "float16", "bfloat16", "int32", "int64"):
    setattr(_torch, _dt, _dt)
sys.modules["torch"] = _torch

from verification.adversarial_search import executor as EX                    # noqa: E402
from verification.adversarial_search.schemas import (                         # noqa: E402
    ExecutionError, InputProposal, KernelExecutionResult, TensorDescriptor)


# ── fixtures ──────────────────────────────────────────────────────────────────

REF_PATH = "/kernels/softmax/reference.py"


def mk_proposal(pid="prop-batch-0001", operator="softmax"):
    return InputProposal(
        proposal_id=pid,
        worker_id="w0",
        iteration=0,
        operator=operator,
        tensors={"x": TensorDescriptor(shape=[8, 16], dtype="float32", fill="randn")},
        rationale="r",
        predicted_failure_mode="partial_tile",
    )


def mk_result(pid, kernel_id, passed=True):
    return KernelExecutionResult(
        proposal_id=pid, kernel_id=kernel_id, passed_checker=passed,
        passed_naive=True, error=None,
        check_results=[{"check_name": "nan_inf", "passed": True,
                        "layer": 1, "details": "finite"}],
        wall_time_ms=1.0,
    )


def kernels_for(n_mutants):
    ks = [("reference", REF_PATH)]
    ks += [(f"mutant_{i}", f"/kernels/softmax/m{i}.py") for i in range(n_mutants)]
    return ks


# ── a fake mp context: real queue, thread-backed "process" ────────────────────

class FakeProcess:
    def __init__(self, target, args):
        self.target, self.args = target, args
        self.killed = False
        self._t = None

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        try:
            self.target(*self.args)
        except Exception:
            traceback.print_exc()

    def join(self, timeout=None):
        # A killed process is gone; join() returns at once. A thread cannot
        # actually be killed, so the flag stands in for it -- without this the
        # elapsed-time assertions would measure the FAKE child's sleep instead
        # of how fast the parent noticed its deadline, which is the only thing
        # under test here.
        if self._t is not None and not self.killed:
            self._t.join(timeout)

    def is_alive(self):
        return (not self.killed) and self._t is not None and self._t.is_alive()

    def kill(self):
        self.killed = True


class FakeCtx:
    def __init__(self):
        self.processes = []

    def Queue(self):
        return queue.Queue()

    def Process(self, target, args):
        p = FakeProcess(target, args)
        self.processes.append(p)
        return p


class _MpShim:
    """Stands in for the `multiprocessing` module.

    It RECORDS the start method the executor asks for. The requested name used
    to be discarded, which was fine while there was only one -- but a start
    method chosen wrongly produces results that look entirely normal, so the
    only way to test the choice is to observe it being made.

    `available` is the set of methods this platform claims to support, so the
    unavailable-forkserver fallback can be exercised without needing a platform
    that actually lacks it.
    """

    def __init__(self, available=("spawn", "fork", "forkserver")):
        self.ctx = FakeCtx()
        self.requested = []
        self.preloads = []
        self.available = list(available)

    def get_all_start_methods(self):
        return list(self.available)

    def get_context(self, name):
        self.requested.append(name)
        self.ctx.set_forkserver_preload = self.preloads.append
        return self.ctx


LAST_SHIM = {}          # the shim from the most recent run_parent, for start-method assertions


def run_parent(child_script, kernels, timeout_seconds=2, on_result=None,
               should_stop=None, proposal=None, use_forkserver=False,
               available=("spawn", "fork", "forkserver")):
    """Drive execute_proposal_batch with a scripted child. Returns (results, ctx)."""
    proposal = proposal or mk_proposal()
    shim = _MpShim(available=available)
    LAST_SHIM["shim"] = shim
    real_mp, real_single = EX.mp, EX.execute_proposal
    EX.mp = shim

    calls = []

    def fake_single(**kw):
        calls.append(kw["kernel_id"])
        return mk_result(kw["proposal"].proposal_id, kw["kernel_id"])

    EX.execute_proposal = fake_single

    # The scripted child replaces the real one by monkeypatching the module
    # attribute the parent spawns, so production code needs no test-only hook.
    real_child = EX._run_batch_in_subprocess

    def target(proposal_dict, ks, ref, op, q, spawn_t):
        child_script(q, ks, proposal_dict)

    EX._run_batch_in_subprocess = target

    try:
        results = EX.execute_proposal_batch(
            proposal=proposal, kernels=kernels, reference_src_path=REF_PATH,
            operator="softmax", timeout_seconds=timeout_seconds,
            on_result=on_result, should_stop=should_stop,
            use_forkserver=use_forkserver,
        )
    finally:
        EX.mp, EX.execute_proposal = real_mp, real_single
        EX._run_batch_in_subprocess = real_child
    return results, calls


# ══ 1. Parent: the happy path ════════════════════════════════════════════════

print("\n── 1. batched happy path: K kernels in, K results out, in order ──")

KS = kernels_for(3)


def clean_child(q, ks, pd):
    for kid, _ in ks:
        q.put(mk_result(pd["proposal_id"], kid).to_dict())
    q.put({EX._BATCH_DONE: True})


seen = []
res, fb = run_parent(clean_child, KS, on_result=seen.append)

ck("returns one result per requested kernel", len(res) == len(KS),
   f"{len(res)} vs {len(KS)}")
ck("preserves the requested order, reference first",
   [r.kernel_id for r in res] == [k for k, _ in KS],
   str([r.kernel_id for r in res]))
ck("on_result fired once per result, before the call returned",
   [r.kernel_id for r in seen] == [k for k, _ in KS],
   str([r.kernel_id for r in seen]))
ck("NO fallback fired on a clean batch -- this is the control that keeps every "
   "other fallback assertion below non-vacuous", fb == [], str(fb))


# ══ 2. Parent: the fallback, and it must be selective ════════════════════════

print("\n── 2. a child that dies partway: banked results kept, rest recovered ──")


def dies_after_two(q, ks, pd):
    for kid, _ in ks[:2]:
        q.put(mk_result(pd["proposal_id"], kid).to_dict())
    return                                   # no sentinel, thread just ends


seen = []
res, fb = run_parent(dies_after_two, KS, on_result=seen.append)

ck("still returns one result per kernel after a mid-batch death",
   len(res) == len(KS), str(len(res)))
ck("the kernels that DID report are kept, not re-run",
   fb == [k for k, _ in KS[2:]], str(fb))
ck("recovered kernels are marked as fallbacks in the data, not merely printed",
   all(r.exec_mode.startswith("single_fallback") for r in res[2:]),
   str([r.exec_mode for r in res[2:]]))
ck("the reference is never the one lost -- it runs first",
   res[0].kernel_id == "reference" and not res[0].exec_mode.startswith("single_fallback"))

print("\n── 2b. a poisoned CUDA context aborts the batch, not the proposal ──")


def poisons_after_one(q, ks, pd):
    q.put(mk_result(pd["proposal_id"], ks[0][0]).to_dict())
    q.put({EX._BATCH_ABORTED: "CUDA context unusable after 'reference'"})


res, fb = run_parent(poisons_after_one, KS)
ck("every kernel after the abort is re-run in a clean process",
   fb == [k for k, _ in KS[1:]], str(fb))
ck("the abort reason reaches the persisted record",
   all(r.exec_mode == "single_fallback:aborted" for r in res[1:]),
   str([r.exec_mode for r in res[1:]]))

print("\n── 2c. a hung kernel trips a PER-KERNEL deadline, not a batch-wide one ──")


def hangs_after_one(q, ks, pd):
    q.put(mk_result(pd["proposal_id"], ks[0][0]).to_dict())
    time.sleep(30)                            # daemon thread; never reports again


t0 = time.time()
res, fb = run_parent(hangs_after_one, KS, timeout_seconds=1)
elapsed = time.time() - t0
ck("the hung kernel and everything behind it are recovered",
   fb == [k for k, _ in KS[1:]], str(fb))
ck("a hang costs ONE kernel's timeout, not the whole batch's budget",
   elapsed < 1 + len(KS), f"{elapsed:.2f}s for timeout_seconds=1, {len(KS)} kernels")


# ══ 3. Parent: the deadlock guard ════════════════════════════════════════════

print("\n── 3. the drain loop must never join-before-drain ──")

src = (REPO / "verification/adversarial_search/executor.py").read_text()
batch_src = src[src.index("def execute_proposal_batch"):]
_join_at = batch_src.index("p.join(")
_drain_at = batch_src.index("queue.get(timeout=")
ck("execute_proposal_batch drains the queue BEFORE joining the child",
   _drain_at < _join_at,
   "a child blocked in queue.put() while the parent blocks in join() is a "
   "deadlock, and N+1 check_results payloads no longer fit the pipe buffer")

# The text check above is necessary but weak -- it cannot tell a harmless join
# from a blocking one. This reproduces the deadlock itself. A real mp.Queue is
# bounded by the OS pipe buffer, so a child streaming N+1 check_results payloads
# WILL block in put(); a bounded queue.Queue reproduces exactly that condition,
# where the unbounded default silently cannot.


class _BoundedCtx(FakeCtx):
    def Queue(self):
        return queue.Queue(maxsize=1)


def _run_bounded():
    shim = _MpShim()
    shim.ctx = _BoundedCtx()
    real_mp, real_child = EX.mp, EX._run_batch_in_subprocess
    EX.mp = shim
    EX._run_batch_in_subprocess = lambda pd, ks, ref, op, q, t: clean_child(q, ks, pd)
    box = {}
    try:
        box["res"] = EX.execute_proposal_batch(
            proposal=mk_proposal(), kernels=KS, reference_src_path=REF_PATH,
            operator="softmax", timeout_seconds=2)
    finally:
        EX.mp, EX._run_batch_in_subprocess = real_mp, real_child
    return box


_box = {}
_th = threading.Thread(target=lambda: _box.update(_run_bounded()), daemon=True)
_th.start()
_th.join(timeout=15)
ck("CONTROL: a child that BLOCKS in put() because the queue is full still "
   "completes -- the parent drains concurrently instead of deadlocking",
   not _th.is_alive() and len(_box.get("res", [])) == len(KS),
   "still running after 15s" if _th.is_alive() else str(len(_box.get("res", []))))


def never_signals(q, ks, pd):
    for kid, _ in ks:
        q.put(mk_result(pd["proposal_id"], kid).to_dict())
    time.sleep(30)                            # all results sent, no sentinel


t0 = time.time()
res, fb = run_parent(never_signals, KS, timeout_seconds=1)
ck("a missing sentinel does not hang: all results present, loop exits on count",
   len(res) == len(KS) and fb == [] and (time.time() - t0) < 5,
   f"{time.time() - t0:.2f}s, fallback={fb}")


def big_payloads(q, ks, pd):
    for kid, _ in ks:
        r = mk_result(pd["proposal_id"], kid)
        r.check_results = [{"check_name": f"c{i}", "passed": True, "layer": 2,
                            "details": "x" * 4096} for i in range(200)]
        q.put(r.to_dict())
    q.put({EX._BATCH_DONE: True})


t0 = time.time()
res, fb = run_parent(big_payloads, KS, timeout_seconds=3)
ck("oversized payloads for every kernel still drain without deadlock",
   len(res) == len(KS) and fb == [] and (time.time() - t0) < 5,
   f"{time.time() - t0:.2f}s, fallback={fb}")


# ══ 4. Parent: the stop event ════════════════════════════════════════════════

print("\n── 4. a confirmed hit stops the batch without spawning fallbacks ──")

stop = {"v": False}


def stops_midway(q, ks, pd):
    q.put(mk_result(pd["proposal_id"], ks[0][0]).to_dict())
    stop["v"] = True
    time.sleep(30)


res, fb = run_parent(stops_midway, KS, timeout_seconds=5,
                     should_stop=lambda: stop["v"])
ck("a stop does NOT trigger fallback re-runs -- the search is over", fb == [], str(fb))
ck("the contract still holds: one result per kernel", len(res) == len(KS))
ck("unrun kernels are marked SearchStopped, not silently passed",
   all(r.error is not None and r.error.error_type == "SearchStopped"
       for r in res[1:]),
   str([(r.kernel_id, r.error.error_type if r.error else None) for r in res[1:]]))


# ══ 5. Child: one materialisation, one clone per kernel ══════════════════════

print("\n── 5. the child materialises ONCE and clones per kernel ──")


class FakeCtxObj:
    def __init__(self):
        self.loads = []

    def load_fn(self, path, op):
        self.loads.append(path)
        return lambda *a, **k: FakeTensor()

    SPEC_MAP = {"softmax": (lambda: "SPEC")}
    KernelChecker = staticmethod(lambda spec: None)


def run_child(kernels, cuda=False, poison_after=None, materialize_raises=False,
              pid="prop-batch-0001"):
    q = queue.Queue()
    calls = {"materialize": 0, "inputs": [], "kernels": []}

    child_ctx = FakeCtxObj()
    real = (EX._ChildContext, EX.materialize_proposal, EX._evaluate_kernel)

    def fake_materialize(proposal, device="cuda"):
        calls["materialize"] += 1
        if materialize_raises:
            raise ValueError("Failed to materialize tensor 'x': bad shape")
        return {"x": FakeTensor(value=1.0)}

    def fake_eval(proposal_id, kernel_id, cand, ref, inputs, spec, checker):
        calls["kernels"].append(kernel_id)
        calls["inputs"].append(inputs)
        if poison_after is not None and kernel_id == poison_after:
            _fake_cuda._armed = True
        return mk_result(proposal_id, kernel_id)

    EX._ChildContext = lambda: child_ctx
    EX.materialize_proposal = fake_materialize
    EX._evaluate_kernel = fake_eval
    _fake_cuda.available = cuda
    _fake_cuda._armed = False
    _torch.seeds.clear()
    try:
        EX._run_batch_in_subprocess(
            mk_proposal(pid).to_dict(), kernels, REF_PATH, "softmax", q,
            time.time())
    finally:
        EX._ChildContext, EX.materialize_proposal, EX._evaluate_kernel = real
        _fake_cuda.available = False
        _fake_cuda._armed = False

    drained = []
    while True:
        try:
            drained.append(q.get_nowait())
        except queue.Empty:
            break
    return drained, calls, child_ctx


drained, calls, cctx = run_child(KS)

ck("materialize_proposal is called EXACTLY ONCE for the whole batch",
   calls["materialize"] == 1, str(calls["materialize"]))
ck("every kernel ran, reference first",
   calls["kernels"] == [k for k, _ in KS], str(calls["kernels"]))
ck("each kernel gets a DISTINCT tensor object -- an in-place write by one "
   "mutant cannot reach the next",
   len({id(t) for t in calls["inputs"]}) == len(KS),
   str([id(t) for t in calls["inputs"]]))
ck("but every clone descends from the SAME draw, which is the whole point of "
   "materialising once",
   len({t.origin for t in calls["inputs"]}) == 1,
   str([t.origin for t in calls["inputs"]]))
ck("the reference module is loaded once and shared; candidates load per kernel",
   cctx.loads.count(REF_PATH) == 2 and len(cctx.loads) == len(KS) + 1,
   str(cctx.loads))
ck("every result is streamed, plus a completion sentinel",
   len(drained) == len(KS) + 1 and drained[-1].get(EX._BATCH_DONE) is True,
   str(len(drained)))
ck("results carry exec_mode='batched' and a shared batch_spawn_ms",
   all(d["exec_mode"] == "batched" for d in drained[:-1])
   and len({d["batch_spawn_ms"] for d in drained[:-1]}) == 1,
   str([(d.get("exec_mode"), d.get("batch_spawn_ms")) for d in drained[:-1]]))
ck("startup_phases decompose the spawn cost, not just total it",
   set(drained[0]["startup_phases"]) >= {"torch_import_ms", "pre_module_ms",
                                         "spec_import_ms", "materialize_ms"},
   str(sorted(drained[0]["startup_phases"])))
ck("total_wall_time_ms stays NULL on batched rows -- a per-kernel spawn "
   "interval does not exist for a shared subprocess and must not be invented",
   all(d["total_wall_time_ms"] is None for d in drained[:-1]))


# ══ 6. Child: seeding ════════════════════════════════════════════════════════

print("\n── 6. the declared semantic change: seeded, reproducible draws ──")

_, _, _ = run_child(KS, pid="prop-A")
seeds_a = list(_torch.seeds)
_, _, _ = run_child(KS, pid="prop-A")
seeds_a2 = list(_torch.seeds)
_, _, _ = run_child(KS, pid="prop-B")
seeds_b = list(_torch.seeds)

ck("the same proposal id always seeds identically (resume-safe, re-run-safe)",
   seeds_a == seeds_a2 and len(seeds_a) == 1, f"{seeds_a} vs {seeds_a2}")
ck("different proposals do not collide", seeds_a != seeds_b, f"{seeds_a} {seeds_b}")
ck("the single-kernel path is left UNSEEDED, so the A/B arms really do differ "
   "and the change is measurable rather than confounding",
   "manual_seed" not in src[src.index("def _run_in_subprocess"):
                            src.index("def execute_proposal(")])


# ══ 7. Child: negative controls ══════════════════════════════════════════════

print("\n── 7. negative controls (each must FIRE) ──")

drained, calls, _ = run_child(KS, cuda=True, poison_after="reference")
kinds = [d for d in drained if EX._BATCH_ABORTED in d]
ck("CONTROL: a kernel that poisons the CUDA context aborts the batch",
   len(kinds) == 1, str(drained[-1]))
ck("CONTROL: no result is fabricated for the kernels that never ran -- this is "
   "the failure mode that would silently corrupt every verdict after it",
   len([d for d in drained if "kernel_id" in d]) == 1,
   str([d.get("kernel_id") for d in drained if "kernel_id" in d]))

drained, calls, _ = run_child(KS, cuda=True)
ck("CONTROL IS NOT VACUOUS: with nothing poisoned, no abort is emitted and "
   "every kernel runs", len([d for d in drained if EX._BATCH_ABORTED in d]) == 0
   and calls["kernels"] == [k for k, _ in KS])

drained, calls, _ = run_child(KS, materialize_raises=True)
ck("CONTROL: a setup failure reports the REAL cause for every kernel rather "
   "than letting the parent re-run each one into it at ~10s a time",
   len([d for d in drained if d.get("kernel_id")]) == len(KS)
   and all(d["error"]["error_type"] == "ValueError"
           for d in drained if d.get("kernel_id")),
   str([d.get("error", {}).get("error_type") for d in drained if d.get("kernel_id")]))
ck("CONTROL: setup failure still emits the sentinel, so the parent does not "
   "sit on a deadline waiting for a child that already gave up",
   drained[-1].get(EX._BATCH_DONE) is True)


# ══ 8. Both paths share one per-kernel body ══════════════════════════════════

print("\n── 8. the two paths cannot drift apart ──")

ck("_evaluate_kernel is the ONLY place the checker is run -- a fix on one path "
   "silently missing on the other would surface exactly when a fallback fires",
   src.count("checker.run(") == 1 and src.count("_evaluate_kernel(") == 3,
   f"checker.run={src.count('checker.run(')} "
   f"_evaluate_kernel={src.count('_evaluate_kernel(')}")
ck("the batch fallback calls the UNCHANGED single-kernel path",
   "result = execute_proposal(" in src)


# ══ 9. Grouping replayed against the real recorded run ═══════════════════════

print("\n── 9. grouping replayed against the real 160-row CFA history ──")

import sqlite3                                                                # noqa: E402
from collections import defaultdict                                           # noqa: E402

DB = REPO / "adversarial_results/cfa_rerun_2026-08-20/search_history.db"
if not DB.exists():
    ck("real CFA history DB present", False, str(DB))
else:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    by_prop = defaultdict(list)
    for pid, kid in con.execute("SELECT proposal_id, kernel_id FROM executions"):
        by_prop[pid].append(kid)
    con.close()

    ck("every recorded proposal has exactly one reference execution",
       all(k.count("reference") == 1 for k in by_prop.values()),
       str([p for p, k in by_prop.items() if k.count("reference") != 1][:3]))
    ck("no proposal recorded a duplicate kernel -- the batch keys results by "
       "kernel_id, so a duplicate would silently overwrite",
       all(len(k) == len(set(k)) for k in by_prop.values()))
    sizes = {len(k) for k in by_prop.values()}
    ck("the whole run is uniform 2-kernel proposals: 160 spawns become 80",
       sizes == {2} and len(by_prop) == 80,
       f"sizes={sizes} proposals={len(by_prop)}")


# ══ 10. The new columns survive a round trip through a REAL database ═════════

print("\n── 10. migration + persistence of the new timing columns ──")

import shutil                                                                 # noqa: E402
import tempfile                                                               # noqa: E402
from verification.adversarial_search.history.store import (                   # noqa: E402
    SearchHistoryStore, _LATE_EXECUTION_COLUMNS)

if not DB.exists():
    ck("real DB available to migrate", False)
else:
    tmp = Path(tempfile.mkdtemp()) / "copy.db"
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    out = sqlite3.connect(str(tmp))
    con.backup(out)
    out.close()
    con.close()

    before = sqlite3.connect(str(tmp))
    n_before = before.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
    cols_before = {r[1] for r in before.execute("PRAGMA table_info(executions)")}
    before.close()

    store = SearchHistoryStore(str(tmp))
    with store._conn as c:
        cols_after = {r[1] for r in c.execute("PRAGMA table_info(executions)")}
        n_after = c.execute("SELECT COUNT(*) FROM executions").fetchone()[0]

    ck("migration adds every late column to a pre-existing DB",
       all(name in cols_after for name, _ in _LATE_EXECUTION_COLUMNS),
       str(sorted(set(n for n, _ in _LATE_EXECUTION_COLUMNS) - cols_after)))
    ck("migration removes nothing", cols_before <= cols_after)
    ck("migration preserves every existing row", n_before == n_after == 160,
       f"{n_before} -> {n_after}")
    ck("pre-existing rows read back NULL, not 0.0 -- a fabricated zero would "
       "claim a free subprocess spawn",
       all(r[0] is None for r in store._conn.execute(
           "SELECT batch_spawn_ms FROM executions LIMIT 20")))

    run_id = list(store.list_runs())[0]["run_id"]
    r = mk_result("prop-roundtrip", "reference")
    r.exec_mode = "batched"
    r.batch_spawn_ms = 10250.0
    r.kernel_wall_time_ms = 31.5
    r.startup_phases = {"pre_module_ms": 300.0, "torch_import_ms": 4200.0,
                        "spec_import_ms": 900.0, "cuda_init_ms": 3100.0,
                        "materialize_ms": 12.0}
    r.start_method = "forkserver"
    store.save_execution(run_id, r)
    back = [e for e in store.get_executions(proposal_id="prop-roundtrip")]
    ck("a batched execution round-trips through the store", len(back) == 1)
    if back:
        got = back[0]
        ck("exec_mode, batch_spawn_ms and kernel_wall_time_ms all persist",
           got["exec_mode"] == "batched" and got["batch_spawn_ms"] == 10250.0
           and got["kernel_wall_time_ms"] == 31.5, str(got.get("exec_mode")))
        ck("startup_phases parse back to the decomposition, not a JSON blob",
           isinstance(got["startup_phases"], dict)
           and got["startup_phases"]["cuda_init_ms"] == 3100.0,
           str(got.get("startup_phases"))[:80])
        ck("start_method persists -- without it a forkserver run and a run that "
           "silently fell back to spawn are the same row on disk",
           got.get("start_method") == "forkserver", str(got.get("start_method")))
        ck("pre-existing rows read back start_method NULL, never 'spawn' -- "
           "'never recorded' and 'recorded as spawn' are different facts",
           all(r[0] is None for r in store._conn.execute(
               "SELECT start_method FROM executions "
               "WHERE proposal_id != 'prop-roundtrip' LIMIT 20")))
    store.close() if hasattr(store, "close") else None
    shutil.rmtree(tmp.parent, ignore_errors=True)


# ══ 11. The start method: chosen, recorded, and honest about falling back ════

print("\n── 11. start method: which one was ASKED for, and which was USED ──")

# A start method chosen wrongly produces results that look completely ordinary,
# so every assertion here is about observing the choice rather than its output.

res, _ = run_parent(clean_child, KS, use_forkserver=True)
shim = LAST_SHIM["shim"]
ck("batched + use_forkserver asks multiprocessing for 'forkserver'",
   shim.requested == ["forkserver"], str(shim.requested))
ck("the preload is EXACTLY ['torch'] -- widening it silently moves the module "
   "import stamps into the forkserver and makes torch_import_ms report the "
   "amortised cost on every child",
   shim.preloads == [["torch"]], str(shim.preloads))
ck("every batched result records start_method='forkserver'",
   all(r.start_method == "forkserver" for r in res),
   str([r.start_method for r in res]))

res, _ = run_parent(clean_child, KS, use_forkserver=False)
shim = LAST_SHIM["shim"]
ck("CONTROL IS NOT VACUOUS: the same call without the flag asks for 'spawn' "
   "and sets no preload",
   shim.requested == ["spawn"] and shim.preloads == [], str(shim.requested))
ck("and its results record start_method='spawn'",
   all(r.start_method == "spawn" for r in res),
   str([r.start_method for r in res]))

# THE SILENT-FALLBACK CASE. forkserver is unavailable on some platforms, and
# dropping to spawn without saying so would produce a run that reports
# "forkserver made no difference" having never once forked -- the same defect as
# a subprocess timing its own startup, which is how 71% of search wall time went
# unattributed (#7b).
res, _ = run_parent(clean_child, KS, use_forkserver=True, available=("spawn",))
shim = LAST_SHIM["shim"]
ck("where forkserver is unavailable the request falls back to spawn",
   shim.requested == ["spawn"], str(shim.requested))
ck("and the FALLBACK IS RECORDED as spawn, not as the forkserver that was "
   "asked for -- otherwise the run is indistinguishable from one that forked",
   all(r.start_method == "spawn" for r in res),
   str([r.start_method for r in res]))

# The single-kernel path is spawn-only BY DESIGN, not by omission: it does not
# seed, so under fork every child would inherit one generator state and draw
# identical tensors for every proposal. Asserted on the real function, with mp
# shimmed, because a source-text check could not tell a deliberate choice from a
# forgotten one.
_shim = _MpShim()
_real_mp = EX.mp
EX.mp = _shim
try:
    class _DeadProc:
        def __init__(self, *a, **k): pass
        def start(self): pass
        def join(self, timeout=None): pass
        def is_alive(self): return False
        def kill(self): pass
    _shim.ctx.Process = lambda target, args: _DeadProc()
    _single = EX.execute_proposal(
        proposal=mk_proposal(), kernel_id="reference",
        candidate_src_path=REF_PATH, reference_src_path=REF_PATH,
        operator="softmax", timeout_seconds=1)
finally:
    EX.mp = _real_mp

ck("execute_proposal NEVER asks for forkserver -- it seeds nothing, so a fork "
   "would give every proposal the SAME tensors rather than independent ones",
   _shim.requested == ["spawn"] and _shim.preloads == [], str(_shim.requested))
ck("even a crashed single execution records the start method it used",
   _single.start_method == "spawn", str(_single.start_method))

# The DEFAULT flipped ON 2026-08-28 after the GPU A/B
# (verification_runs/forkserver_ab/: 36-41% end-to-end, gates green) and the
# default-path re-verification. A silent revert would cost that win invisibly
# -- every call site passes the flag explicitly in tests, so only a signature
# check sees the default itself.
import inspect                                                                 # noqa: E402
from verification.adversarial_search.coordinator import SearchCoordinator      # noqa: E402
ck("execute_proposal_batch defaults use_forkserver=True (flipped 2026-08-28; "
   "a revert must be a decision with its own A/B, not a drive-by)",
   inspect.signature(EX.execute_proposal_batch)
       .parameters["use_forkserver"].default is True)
ck("SearchCoordinator defaults use_forkserver=True (same decision, same date)",
   inspect.signature(SearchCoordinator.__init__)
       .parameters["use_forkserver"].default is True)


# ── result ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
if fails:
    print(f"  {len(fails)} FAILED:")
    for f in fails:
        print(f"    - {f}")
    sys.exit(1)
print("  all checks passed")
sys.exit(0)
