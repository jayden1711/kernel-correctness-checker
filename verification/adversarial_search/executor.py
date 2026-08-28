"""
verification/adversarial_search/executor.py

Subprocess-isolated executor.

Two paths, sharing one per-kernel body:

  execute_proposal()        one subprocess per KERNEL. The original path, and
                            still the fallback whenever a batch cannot finish.
  execute_proposal_batch()  one subprocess per PROPOSAL -- reference and all
                            mutants together. Startup (interpreter, `import
                            torch`/triton, CUDA init) was 71% of search wall
                            time because it was paid once per kernel rather
                            than once per proposal; this drops the spawn count
                            from N+1 to 1.

Either way the process is created fresh and killed on timeout, so a crashing
Triton kernel still cannot kill the coordinator.
Always returns KernelExecutionResult — never raises.

HOW each process is created is a SEPARATE axis from how many of them there are,
and the two compose:

  spawn       a fresh interpreter, which re-pays `import torch` -- 5241ms, 85%
              of the ~6185ms startup. The default, and the only method the
              single-kernel path uses.
  forkserver  fork a torch-preloaded server, so `import torch` is paid once per
              search. Offered on the BATCHED path only, because that path
              re-seeds from the proposal id and so overwrites the generator
              state a fork inherits; the single-kernel path deliberately does
              not seed, and under fork "unseeded" would silently become
              "identical for every proposal". See `_mp_context` and
              `execute_proposal`.

The three-layer KernelChecker is the correctness arbiter.
Naive allclose is also recorded to document the checker gap.
"""

from __future__ import annotations
import time

# STAMPED BEFORE `import torch`, DELIBERATELY, AND IT MUST STAY THERE.
#
# With the "spawn" start method the child re-imports this module (via the
# parent's __main__) BEFORE the target function's first line runs, so a
# timestamp taken at function entry cannot see `import torch` at all -- which
# is the single largest component of the ~10.3s startup we are trying to
# attribute. These two module-level stamps are the only place that cost is
# observable from inside the child.
#
# time.time(), not time.perf_counter(): these are compared against a stamp
# taken in the PARENT process, and perf_counter has no cross-process meaning.
_MODULE_IMPORT_T0 = time.time()

import hashlib
import multiprocessing as mp
import os
import queue as _queue
import re
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from verification.adversarial_search.schemas import (
    ExecutionError,
    InputProposal,
    KernelExecutionResult,
)
from verification.adversarial_search.materializer import (
    materialize_proposal,
    tensors_to_inputs,
)

_MODULE_IMPORT_T1 = time.time()

# WHICH PROCESS STAMPED THE TWO TIMES ABOVE.
#
# The stamps only mean "this execution's startup" if this module was imported by
# THIS process. Under both start methods in use today it is:
#
#   spawn       the child re-imports the module via the parent's __main__.
#   forkserver  the preload list is exactly ["torch"] (see _FORKSERVER_PRELOAD),
#               so this module is NOT in the forkserver's sys.modules and each
#               forked child imports it on unpickling the target -- at which
#               point `import torch` is a sys.modules hit costing ~0 rather than
#               the 5241ms it costs cold. That is precisely the saving, and it
#               lands in `torch_import_ms` where it is directly comparable
#               against the spawn arm.
#
# It stops being true the moment somebody adds this module (or anything that
# imports it) to the preload list: the stamps would then be the FORKSERVER's,
# taken once at daemon start and inherited by every fork, so `torch_import_ms`
# would report ~5241ms on every execution -- reading as "forkserver changed
# nothing" -- and `pre_module_ms` would go NEGATIVE, the module having been
# imported before the parent asked for the process.
#
# `_startup_phases` compares this pid against its own and renames the keys if
# they differ, so that mistake shows up as differently-named data rather than as
# plausible-looking wrong numbers. Cheap insurance against a one-line change
# made months from now for an unrelated reason.
_MODULE_IMPORT_PID = os.getpid()

# Sentinels the batched child puts on the queue. A result payload is a dict of
# KernelExecutionResult fields, so these use keys no result can collide with.
_BATCH_DONE = "__batch_done__"
_BATCH_ABORTED = "__batch_aborted__"


# ── Child-side shared machinery ───────────────────────────────────────────────
#
# Both executor paths -- one subprocess per kernel, and one subprocess per
# proposal -- run the SAME per-kernel body. It lives here once, rather than
# being copied into each path, so the two cannot drift: a fix applied to the
# batched path and missed on the fallback path would be invisible until a
# fallback fired, which is exactly when it would matter most.


class _ChildContext:
    """Imports and lookup tables every child needs.

    Constructed inside the subprocess, never in the parent: the per-operator
    spec imports pull in the whole `verification.specs` tree and the parent has
    no use for it.
    """

    __slots__ = ("KernelChecker", "SPEC_MAP", "FUNC_NAMES")

    def __init__(self):
        import os
        import sys

        project_root = os.environ.get("CHECKER_ROOT", ".")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from verification.checker import KernelChecker
        from verification.specs.softmax import get_spec as softmax_spec
        from verification.specs.layernorm import get_spec as layernorm_spec
        from verification.specs.matmul import get_spec as matmul_spec
        from verification.specs.flash_attention import get_spec as flash_attention_spec
        from verification.specs.rmsnorm import get_spec as rmsnorm_spec
        from verification.specs.log_softmax import get_spec as log_softmax_spec
        from verification.specs.swish import get_spec as swish_spec
        from verification.specs.gelu import get_spec as gelu_spec
        from verification.specs.sum_reduction import get_spec as sum_reduction_spec
        from verification.specs.mean_reduction import get_spec as mean_reduction_spec
        from verification.specs.max_reduction import get_spec as max_reduction_spec
        from verification.specs.min_reduction import get_spec as min_reduction_spec
        from verification.specs.l1norm import get_spec as l1norm_spec
        from verification.specs.l2norm import get_spec as l2norm_spec
        from verification.specs.frobenius_norm import get_spec as frobenius_norm_spec
        from verification.specs.argmax import get_spec as argmax_spec
        from verification.specs.argmin import get_spec as argmin_spec
        from verification.specs.instancenorm import get_spec as instancenorm_spec
        from verification.specs.batchnorm import get_spec as batchnorm_spec
        from verification.specs.scaled_dot_product_attention import get_spec as sdpa_spec
        from verification.specs.causal_flash_attention import get_spec as causal_flash_attention_spec

        self.KernelChecker = KernelChecker

        self.SPEC_MAP = {
            "softmax":         softmax_spec,
            "layernorm":       layernorm_spec,
            "matmul":          matmul_spec,
            "flash_attention": flash_attention_spec,
            "rmsnorm":         rmsnorm_spec,
            "log_softmax":                    log_softmax_spec,
            "swish":                          swish_spec,
            "gelu":                           gelu_spec,
            "sum_reduction":                  sum_reduction_spec,
            "mean_reduction":                 mean_reduction_spec,
            "max_reduction":                  max_reduction_spec,
            "min_reduction":                  min_reduction_spec,
            "l1norm":                         l1norm_spec,
            "l2norm":                         l2norm_spec,
            "frobenius_norm":                 frobenius_norm_spec,
            "argmax":                         argmax_spec,
            "argmin":                         argmin_spec,
            "instancenorm":                   instancenorm_spec,
            "batchnorm":                      batchnorm_spec,
            "scaled_dot_product_attention":   sdpa_spec,
            "causal_flash_attention":         causal_flash_attention_spec,
        }

        self.FUNC_NAMES = {
            "softmax":         ["softmax"],
            "layernorm":       ["layernorm"],
            "matmul":          ["matmul"],
            "flash_attention": ["flash_attention"],
            "rmsnorm":         ["rmsnorm", "rms_norm"],
            "log_softmax":                    ["log_softmax"],
            "swish":                          ["swish"],
            "gelu":                           ["gelu"],
            "sum_reduction":                  ["sum_reduction"],
            "mean_reduction":                 ["mean_reduction"],
            "max_reduction":                  ["max_reduction"],
            "min_reduction":                  ["min_reduction"],
            "l1norm":                         ["l1norm"],
            "l2norm":                         ["l2norm"],
            "frobenius_norm":                 ["frobenius_norm"],
            "argmax":                         ["argmax"],
            "argmin":                         ["argmin"],
            "instancenorm":                   ["instancenorm"],
            "batchnorm":                      ["batchnorm"],
            "scaled_dot_product_attention":   ["scaled_dot_product_attention"],
            "causal_flash_attention":         ["causal_flash_attention"],
        }

    def load_fn(self, path: str, op: str) -> Callable:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_mod", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in self.FUNC_NAMES[op]:
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
        raise ValueError(f"No callable found in {path} for operator {op}")


def _error_result(proposal_id: str, kernel_id: str, exc: BaseException,
                  ) -> KernelExecutionResult:
    tb = traceback.format_exc()
    snippet = "\n".join(tb.splitlines()[-6:])
    return KernelExecutionResult(
        proposal_id=proposal_id,
        kernel_id=kernel_id,
        passed_checker=False,
        passed_naive=False,
        error=ExecutionError(
            error_type=type(exc).__name__,
            message=str(exc),
            layer=None,
            check_name=None,
            max_err=None,
            traceback_snippet=snippet,
        ),
        check_results=[],
        wall_time_ms=0.0,
    )


def _evaluate_kernel(
    proposal_id: str,
    kernel_id: str,
    candidate_fn: Callable,
    reference_fn: Callable,
    inputs,
    spec,
    KernelChecker,
) -> KernelExecutionResult:
    """Naive allclose + the full three-layer checker for ONE kernel.

    Lifted unchanged from the original single-kernel subprocess body. The
    checker is the correctness arbiter; naive allclose is recorded alongside it
    only to document the gap between them.
    """
    # Naive allclose
    try:
        if isinstance(inputs, tuple):
            ref_out = reference_fn(*inputs)
            t0 = time.perf_counter()
            cand_out = candidate_fn(*inputs)
            wall_ms = (time.perf_counter() - t0) * 1000
        else:
            ref_out = reference_fn(inputs)
            t0 = time.perf_counter()
            cand_out = candidate_fn(inputs)
            wall_ms = (time.perf_counter() - t0) * 1000

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        naive_pass = torch.allclose(
            cand_out.float(), ref_out.float(), atol=1e-3, rtol=1e-2
        )
    except Exception:
        naive_pass = False
        wall_ms = 0.0

    # Full three-layer checker
    checker = KernelChecker(spec)
    check_results = checker.run(candidate_fn, None, reference_fn, inputs)
    passed_checker = all(r.passed for r in check_results)

    return KernelExecutionResult(
        proposal_id=proposal_id,
        kernel_id=kernel_id,
        passed_checker=passed_checker,
        passed_naive=naive_pass,
        error=None,
        check_results=[
            {
                "check_name": r.check_name,
                "passed": r.passed,
                "layer": r.layer,
                "details": r.details,
            }
            for r in check_results
        ],
        wall_time_ms=wall_ms,
    )


def _seed_for(proposal_id: str) -> int:
    """Deterministic per-proposal RNG seed.

    Derived from the proposal id rather than a counter so it is stable across
    resumes, re-runs and worker interleaving: the same proposal always draws the
    same tensors, and two different proposals essentially never collide.
    """
    digest = hashlib.sha256(proposal_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63 - 1)


def _startup_phases(parent_spawn_t: Optional[float]) -> Dict[str, float]:
    """The two phases that are only observable from module-import stamps.

    `pre_module_ms` covers interpreter boot plus multiprocessing's bootstrap
    (plus, under spawn, the re-import of the parent's `__main__` chain) up to
    this module -- all of it paid before the target function's first line.
    `torch_import_ms` is `import torch` and this module's own imports.

    Both keep their names and meanings under forkserver, which is what makes the
    two arms comparable: with a `["torch"]` preload the child still imports this
    module, so both stamps are still this execution's, and `torch_import_ms`
    simply collapses toward zero because torch is already resident.

    The `_MODULE_IMPORT_PID` guard is the exception. If the stamps were taken in
    a DIFFERENT process they describe that process's one-time startup, not this
    execution's, and summing them per execution would be a fabricated number.
    Rather than emit a wrong value under the right key -- or a negative
    `pre_module_ms`, which is what the arithmetic actually produces -- the keys
    are renamed so no reader can mistake one for the other, and
    `startup_stamps_inherited_ms` reports the thing that IS true: a cost paid
    once, elsewhere, and amortised across every child.
    """
    import_ms = 1000.0 * (_MODULE_IMPORT_T1 - _MODULE_IMPORT_T0)

    if os.getpid() != _MODULE_IMPORT_PID:
        return {"startup_stamps_inherited_ms": import_ms}

    phases: Dict[str, float] = {"torch_import_ms": import_ms}
    if parent_spawn_t is not None:
        phases["pre_module_ms"] = 1000.0 * (_MODULE_IMPORT_T0 - parent_spawn_t)
    return phases


# ── The start method, shared by both executor paths ──────────────────────────

# EXACTLY ["torch"], and the narrowness is load-bearing -- see the
# `_MODULE_IMPORT_PID` note above for what widening it silently does to the
# startup numbers. `import torch` measured 5241ms of the ~6185ms of startup
# (85%); CUDA init, the next 645ms (10%), deliberately stays per-child, because
# initialising CUDA in the forkserver would leave every forked child holding an
# inherited, unusable context.
_FORKSERVER_PRELOAD = ["torch"]


def _mp_context(prefer_forkserver: bool):
    """Return `(ctx, start_method_actually_used)` for spawning a child.

    ONE factory for BOTH executor paths, rather than a start method chosen
    inside whichever function happens to need one. The two paths already share
    `_evaluate_kernel` for the same reason: a difference that exists on one path
    and not the other is invisible until the rarely-taken path is taken, which
    is exactly when it costs the most.

    Note that batching and the start method are ORTHOGONAL and compose. Batching
    sets how many processes a proposal costs (N+1 -> 1); this sets how much each
    one costs to create. Neither subsumes the other, so they stay two switches.

    RETURNS THE METHOD ACTUALLY USED, NOT THE ONE REQUESTED. forkserver is
    unavailable on some platforms, and a silent drop to spawn would produce a
    run that reports "forkserver made no difference" having never once used it.
    The caller stamps this onto every result for that reason alone.

    `mp.get_context` is called HERE, per call, and the result is not cached at
    module scope: `tests/instrumentation/check_batch_executor.py` substitutes the
    `mp` module attribute to drive the parent's drain loop against a scripted
    child, and a cached context would make that substitution a no-op -- a test
    silently exercising the real multiprocessing machinery instead of the fake.
    """
    if not prefer_forkserver:
        return mp.get_context("spawn"), "spawn"

    if "forkserver" not in mp.get_all_start_methods():
        return mp.get_context("spawn"), "spawn"

    ctx = mp.get_context("forkserver")
    # Idempotent, and must precede the first Process.start(): the preload list
    # binds when the forkserver daemon boots, and the daemon boots on first use.
    # Setting it on every call is simplest and cannot be got wrong by ordering.
    setter = getattr(ctx, "set_forkserver_preload", None)
    if setter is not None:
        setter(_FORKSERVER_PRELOAD)
    return ctx, "forkserver"


# ── Path 1: one subprocess per kernel (original; also the batch fallback) ─────


def _run_in_subprocess(
    proposal_dict: dict,
    kernel_id: str,
    candidate_src_path: str,
    reference_src_path: str,
    operator: str,
    queue: mp.Queue,
    parent_spawn_t: Optional[float] = None,
):
    phases = _startup_phases(parent_spawn_t)
    t_kernel0 = None

    try:
        from verification.adversarial_search.schemas import InputProposal

        _t = time.perf_counter()
        ctx = _ChildContext()
        phases["spec_import_ms"] = 1000.0 * (time.perf_counter() - _t)

        proposal = InputProposal.from_dict(proposal_dict)

        _t = time.perf_counter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            # Force context creation here so its cost is attributed to CUDA
            # init rather than silently inflating the first kernel's interval.
            torch.zeros(1, device="cuda")
        phases["cuda_init_ms"] = 1000.0 * (time.perf_counter() - _t)

        t_kernel0 = time.perf_counter()
        candidate_fn = ctx.load_fn(candidate_src_path, operator)
        reference_fn = ctx.load_fn(reference_src_path, operator)

        _t = time.perf_counter()
        tensors = materialize_proposal(proposal, device=device)
        inputs = tensors_to_inputs(operator, tensors)
        phases["materialize_ms"] = 1000.0 * (time.perf_counter() - _t)

        result = _evaluate_kernel(
            proposal.proposal_id, kernel_id, candidate_fn, reference_fn,
            inputs, ctx.SPEC_MAP[operator](), ctx.KernelChecker,
        )

    except Exception as e:
        result = _error_result(
            proposal_dict.get("proposal_id", "unknown"), kernel_id, e)

    result.exec_mode = "single"
    result.startup_phases = phases
    if t_kernel0 is not None:
        result.kernel_wall_time_ms = 1000.0 * (time.perf_counter() - t_kernel0)

    queue.put(result.to_dict())


def execute_proposal(
    proposal: InputProposal,
    kernel_id: str,
    candidate_src_path: str,
    reference_src_path: str,
    operator: str,
    timeout_seconds: int = 30,
) -> KernelExecutionResult:
    # ALWAYS SPAWN, AND THAT IS A DECISION RATHER THAN AN OVERSIGHT.
    #
    # This path deliberately seeds nothing (see the note in
    # `_run_batch_in_subprocess`, and the assertion in
    # `tests/instrumentation/check_batch_executor.py`), so each child draws its
    # own tensors from OS entropy. Under forkserver "unseeded" would not mean
    # "independent" -- every child would inherit the forkserver's generator
    # state and draw IDENTICAL tensors for every proposal, silently collapsing
    # the input diversity of both the fallback and the `--no-batch` arm. That is
    # strictly worse than the status quo, so the cheap process creation is
    # declined here on purpose.
    #
    # The cost of declining it is measured at zero: the fallback fired 0 times
    # across the 140 batched kernel records of the 2026-08-21 T4 run. Re-seeding
    # this path from os.urandom would recover both properties at once; that is a
    # separate decision, deliberately not folded into a latency change.
    ctx, start_method = _mp_context(False)
    queue = ctx.Queue()
    # PARENT-SIDE timer. `wall_time_ms` inside the subprocess covers only the
    # kernel call; this covers the whole thing, so total - wall = spawn +
    # `import torch`/triton + CUDA init. Measured at 10.25s median against a
    # 0.03s median in-kernel time on 2026-08-20 -- see KernelExecutionResult.
    #
    # `_t0_wall` is the same instant on the wall clock, passed to the child so
    # it can attribute the part of that gap it cannot otherwise see (everything
    # before its own first line). perf_counter has no cross-process meaning.
    _t0 = time.perf_counter()
    _t0_wall = time.time()
    p = ctx.Process(
        target=_run_in_subprocess,
        args=(
            proposal.to_dict(),
            kernel_id,
            candidate_src_path,
            reference_src_path,
            operator,
            queue,
            _t0_wall,
        ),
    )
    p.start()
    p.join(timeout=timeout_seconds)

    if p.is_alive():
        p.kill()
        p.join()
        return KernelExecutionResult(
            proposal_id=proposal.proposal_id,
            kernel_id=kernel_id,
            passed_checker=False,
            passed_naive=False,
            error=ExecutionError(
                error_type="TimeoutError",
                message=f"Timed out after {timeout_seconds}s",
                layer=None,
                check_name=None,
                max_err=None,
                traceback_snippet="",
            ),
            check_results=[],
            wall_time_ms=timeout_seconds * 1000.0,
            total_wall_time_ms=1000.0 * (time.perf_counter() - _t0),
            start_method=start_method,
        )

    if queue.empty():
        return KernelExecutionResult(
            proposal_id=proposal.proposal_id,
            kernel_id=kernel_id,
            passed_checker=False,
            passed_naive=False,
            error=ExecutionError(
                error_type="SubprocessCrash",
                message="Subprocess exited with no result",
                layer=None,
                check_name=None,
                max_err=None,
                traceback_snippet="",
            ),
            check_results=[],
            wall_time_ms=0.0,
            total_wall_time_ms=1000.0 * (time.perf_counter() - _t0),
            start_method=start_method,
        )

    payload = queue.get()
    result = KernelExecutionResult.from_dict(payload)
    # Both stamped in the parent, for the same reason: the subprocess can see
    # neither its own spawn cost nor which start method created it.
    result.total_wall_time_ms = 1000.0 * (time.perf_counter() - _t0)
    result.start_method = start_method
    return result


# ── Path 2: one subprocess per PROPOSAL ──────────────────────────────────────
#
# Startup -- interpreter, `import torch`/triton, CUDA init -- was 71% of each
# search worker's wall time (394.7s of 556.2s on the 2026-08-20
# causal_flash_attention run) because it was paid once per (proposal, kernel)
# pair rather than once per proposal. Reference and mutants share identical
# startup and identical inputs, so there was never a reason for them to be in
# different processes; batching drops the spawn count from N+1 per proposal to
# 1, which is 2->1 for the 16 single-mutant operators and 5->1 for matmul and
# flash_attention.
#
# What this path deliberately does NOT do is reuse a process ACROSS proposals.
# Process lifetime is unchanged: still spawned, still killed on timeout, still
# unable to take the coordinator down with it. A persistent pool is the larger
# win and a separate decision.

# Startup budget allowed before the FIRST result, on top of the per-kernel
# timeout. A safety valve, not a tuning knob: startup measures ~10s, but four
# workers contending on one GPU can stretch it, and killing a batch that was
# merely queued behind three others would be a self-inflicted failure.
_BATCH_SPAWN_BUDGET_S = 120.0

# How often the drain loop wakes to re-check liveness and the stop event.
_DRAIN_POLL_S = 0.5

# Grace period to keep draining after the child exits. A child can exit with
# payloads still in flight through the queue's feeder thread, and treating exit
# as end-of-stream would discard results that were genuinely produced.
_CHILD_EXIT_GRACE_S = 2.0


def _run_batch_in_subprocess(
    proposal_dict: dict,
    kernels: Sequence[Tuple[str, str]],
    reference_src_path: str,
    operator: str,
    queue: mp.Queue,
    parent_spawn_t: float,
):
    """Run every kernel of one proposal, streaming each result as it lands.

    Streaming is not an optimisation, it is the crash-safety mechanism: a
    kernel that hangs or corrupts the CUDA context must not take down the
    results of the kernels that already finished. Anything already on the queue
    survives; the parent re-runs the rest one-by-one.
    """
    phases = _startup_phases(parent_spawn_t)
    proposal_id = proposal_dict.get("proposal_id", "unknown")

    try:
        from verification.adversarial_search.schemas import InputProposal

        _t = time.perf_counter()
        ctx = _ChildContext()
        phases["spec_import_ms"] = 1000.0 * (time.perf_counter() - _t)

        proposal = InputProposal.from_dict(proposal_dict)

        _t = time.perf_counter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.zeros(1, device="cuda")
        phases["cuda_init_ms"] = 1000.0 * (time.perf_counter() - _t)

        # DECLARED SEMANTIC CHANGE, not an incidental side effect of batching.
        #
        # The single-kernel path seeds nothing, so each spawned process draws
        # from its own OS-entropy seed -- meaning the reference and the mutants
        # of the SAME proposal were evaluated on DIFFERENT random tensors, and
        # `_evaluate_verdict` compared `reference_passed` from one draw against
        # mutant outcomes from another. That is a real source of verdict noise
        # on marginal proposals. Batching materialises once, which removes it;
        # seeding from the proposal id additionally makes a run reproducible.
        # Both effects can shift verdicts at the margin and must be reported as
        # such, not folded into a latency number.
        #
        # IT IS ALSO WHAT MAKES THIS PATH SAFE UNDER forkserver, AND THAT IS NOT
        # A COINCIDENCE THIS CODE CAN AFFORD TO LOSE. A forked child inherits the
        # forkserver's generator state, so a child that did not seed would draw
        # the SAME tensors for every proposal in the run -- not "unseeded" but
        # "identically seeded", which no test comparing one proposal against
        # itself would notice. These three lines overwrite that inherited state,
        # which is the only reason `use_forkserver` is offered on this path and
        # refused on the single-kernel one.
        #
        # THE ORDERING IS LOAD-BEARING TOO: nothing may draw between process
        # entry and this point. `torch.zeros` above is constant, and
        # materialization, perturbation's samples and the cross-shape sweeps all
        # run after it. `tests/instrumentation/check_forkserver_executor.py`
        # asserts the ordering functionally -- and fails if these lines are
        # removed, which is the only thing that makes "seeding is preserved"
        # falsifiable.
        seed = _seed_for(proposal.proposal_id)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

        _t = time.perf_counter()
        base_tensors = materialize_proposal(proposal, device=device)
        phases["materialize_ms"] = 1000.0 * (time.perf_counter() - _t)

        reference_fn = ctx.load_fn(reference_src_path, operator)
        spec = ctx.SPEC_MAP[operator]()

    except Exception as e:
        # Setup failed, so it fails identically for every kernel. Report them
        # all with the real cause rather than letting the parent re-run each
        # one into the same failure at ~10s a time.
        for kernel_id, _ in kernels:
            r = _error_result(proposal_id, kernel_id, e)
            r.exec_mode = "batched"
            r.startup_phases = phases
            queue.put(r.to_dict())
        queue.put({_BATCH_DONE: True})
        return

    spawn_ms = 1000.0 * (time.time() - parent_spawn_t)

    for kernel_id, candidate_src_path in kernels:
        t_kernel0 = time.perf_counter()
        try:
            # Each kernel gets its own CLONE of the shared tensors. One
            # materialisation (so one RNG draw, so identical inputs across
            # kernels), but no shared mutable state: a kernel that writes to
            # its input in place -- which a mutant is entirely free to do --
            # cannot corrupt the kernels that run after it. The clone is
            # microseconds against a ~10s startup.
            tensors = {k: v.clone() for k, v in base_tensors.items()}
            inputs = tensors_to_inputs(operator, tensors)

            # Loaded fresh per kernel, exactly as the single path does, so the
            # per-kernel semantics are unchanged. Only the REFERENCE module is
            # shared across the batch.
            candidate_fn = ctx.load_fn(candidate_src_path, operator)

            result = _evaluate_kernel(
                proposal.proposal_id, kernel_id, candidate_fn, reference_fn,
                inputs, spec, ctx.KernelChecker,
            )
        except Exception as e:
            result = _error_result(proposal.proposal_id, kernel_id, e)

        result.exec_mode = "batched"
        result.batch_spawn_ms = spawn_ms
        result.kernel_wall_time_ms = 1000.0 * (time.perf_counter() - t_kernel0)
        result.startup_phases = phases
        queue.put(result.to_dict())

        # Poisoned-context guard. An out-of-bounds Triton kernel leaves CUDA in
        # a sticky error state, after which EVERY subsequent call raises. Those
        # would be recorded as "the next mutant crashed" for kernels that never
        # actually ran -- a fabricated result, and the single worst failure mode
        # this path could have. Detect it and hand the remainder back to the
        # parent, which re-runs them in clean processes.
        if device == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception as e:
                queue.put({_BATCH_ABORTED:
                           f"CUDA context unusable after {kernel_id!r}: {e}"})
                return

    queue.put({_BATCH_DONE: True})


def execute_proposal_batch(
    proposal: InputProposal,
    kernels: Sequence[Tuple[str, str]],
    reference_src_path: str,
    operator: str,
    timeout_seconds: int = 30,
    on_result: Optional[Callable[[KernelExecutionResult], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    use_forkserver: bool = True,
) -> List[KernelExecutionResult]:
    """Run all of one proposal's kernels in a single subprocess.

    `kernels` is an ordered sequence of (kernel_id, candidate_src_path) with the
    REFERENCE FIRST -- so that if a mutant poisons the CUDA context, the one
    result the verdict cannot be computed without is already banked.

    `on_result` is called with each result the moment it arrives, before this
    function returns. The coordinator uses it to commit each execution to the
    history DB as it lands, which is why a batch that dies half way through
    still leaves every completed execution persisted.

    `use_forkserver` creates the child by forking a torch-preloaded server
    rather than booting a fresh interpreter, which removes the 5241ms (85%) that
    `import torch` costs each startup. It is SAFE ON THIS PATH SPECIFICALLY --
    and only on this path -- because the child re-seeds from the proposal id
    below, overwriting whatever generator state it inherited. See
    `execute_proposal` for why the single-kernel path does not get the same
    treatment. Isolation and timeout semantics are unchanged either way: still
    one process per execution, still killed on timeout, still unable to take the
    coordinator down with it.

    DEFAULT ON since 2026-08-28: the GPU A/B (verification_runs/forkserver_ab/)
    measured 36-41%% end-to-end on the adversarial search with all three gates
    green (arm B 100%% forkserver with zero silent fallbacks, order drift < 2%%,
    forced-timeout probe identical across start methods), and the gates were
    re-verified at this default in verification_runs/forkserver_default_*/.
    Platforms without forkserver still degrade to spawn, recorded as such on
    every result (`_mp_context` returns the method actually used).

    Returns one result per entry in `kernels`, in the same order. Never raises.
    """
    kernels = list(kernels)
    if not kernels:
        return []

    ctx, start_method = _mp_context(use_forkserver)
    queue = ctx.Queue()
    parent_spawn_t = time.time()
    p = ctx.Process(
        target=_run_batch_in_subprocess,
        args=(proposal.to_dict(), kernels, reference_src_path, operator,
              queue, parent_spawn_t),
    )
    p.start()

    results: Dict[str, KernelExecutionResult] = {}
    abort_reason: Optional[str] = None
    stopped = False
    exit_grace_deadline: Optional[float] = None

    # PER-KERNEL deadline, not a single deadline for the whole batch. Today
    # each kernel gets `timeout_seconds` of its own; collapsing that into one
    # batch-wide budget would let a hung reference eat the mutants' time before
    # anyone noticed. The first result additionally gets the startup budget.
    next_deadline = parent_spawn_t + _BATCH_SPAWN_BUDGET_S + timeout_seconds

    # Drain CONCURRENTLY with the child, never join-then-drain. A child blocked
    # in queue.put() while the parent blocks in join() is a deadlock, and with
    # N+1 check_results payloads in flight the pipe buffer is no longer big
    # enough to hide it.
    while len(results) < len(kernels):
        now = time.time()
        if now >= next_deadline:
            break
        if exit_grace_deadline is not None and now >= exit_grace_deadline:
            break
        if should_stop is not None and should_stop():
            stopped = True
            break

        wait = min(next_deadline - now, _DRAIN_POLL_S)
        if exit_grace_deadline is not None:
            wait = min(wait, exit_grace_deadline - now)
        try:
            payload = queue.get(timeout=max(wait, 0.0))
        except _queue.Empty:
            if not p.is_alive() and exit_grace_deadline is None:
                exit_grace_deadline = time.time() + _CHILD_EXIT_GRACE_S
            continue

        # Membership, not truthiness: a sentinel whose payload happened to be
        # falsy would otherwise be parsed as a result and blow up below.
        if _BATCH_DONE in payload:
            break
        if _BATCH_ABORTED in payload:
            abort_reason = str(payload[_BATCH_ABORTED])
            break

        result = KernelExecutionResult.from_dict(payload)
        # Before on_result, so the row the coordinator persists carries it too.
        result.start_method = start_method
        results[result.kernel_id] = result
        next_deadline = time.time() + timeout_seconds
        if on_result is not None:
            on_result(result)

    if p.is_alive():
        p.kill()
    p.join()

    missing = [(kid, path) for kid, path in kernels if kid not in results]

    if missing and not stopped:
        # FALLBACK. Whatever went wrong -- poisoned context, hung kernel, dead
        # child -- the kernels that never reported are re-run through the
        # unchanged one-process-per-kernel path, which is exactly the behaviour
        # they would have had before batching existed.
        #
        # The reason is recorded in `exec_mode` rather than only printed: a
        # fallback that fires constantly would show up as "batching delivered
        # no speedup" with nothing in the data to say why.
        if abort_reason is not None:
            reason = "aborted"
        elif not results:
            reason = "no_result"
        else:
            reason = "deadline"
        print(f"[executor] batch fallback ({reason}) for proposal "
              f"{proposal.proposal_id[:8]}: re-running "
              f"{[k for k, _ in missing]} individually"
              + (f" -- {abort_reason}" if abort_reason else ""))

        for kernel_id, candidate_src_path in missing:
            if should_stop is not None and should_stop():
                stopped = True
                break
            result = execute_proposal(
                proposal=proposal,
                kernel_id=kernel_id,
                candidate_src_path=candidate_src_path,
                reference_src_path=reference_src_path,
                operator=operator,
                timeout_seconds=timeout_seconds,
            )
            result.exec_mode = f"single_fallback:{reason}"
            results[kernel_id] = result
            if on_result is not None:
                on_result(result)

    # One result per requested kernel, in the requested order. Only a stop
    # event can leave a hole, and it is filled explicitly rather than by
    # returning a short list that every caller would have to special-case.
    out: List[KernelExecutionResult] = []
    for kernel_id, _ in kernels:
        if kernel_id in results:
            out.append(results[kernel_id])
            continue
        out.append(KernelExecutionResult(
            proposal_id=proposal.proposal_id,
            kernel_id=kernel_id,
            passed_checker=False,
            passed_naive=False,
            error=ExecutionError(
                error_type="SearchStopped",
                message="Search stopped before this kernel ran",
                layer=None,
                check_name=None,
                max_err=None,
                traceback_snippet="",
            ),
            check_results=[],
            wall_time_ms=0.0,
            exec_mode="batched",
            start_method=start_method,
        ))
    return out


# Failure-mode classification for a REFERENCE failure.
#
# The two sentinels both invoke the kernel, so BOTH failing is the signature
# of the kernel raising rather than of a bad numeric result: a genuine
# overflow produces non-finite output but still preserves dtype, so it fails
# nan_inf alone.
_SENTINEL_CHECKS = {"nan_inf", "dtype_preserved"}
_STRUCTURAL_CHECKS = {"kernel_executed", "tile_coverage_structural",
                      "tile_coverage_softmax_positivity", "determinism"}
_PRECISION_CHECKS = {"precision_coercion"}


def _diagnose_reference_failure(check_results) -> tuple:
    """
    Classify WHY the reference failed, and return (label, advice).

    This replaces a single hardcoded "Reduce input magnitude by 10x or use a
    simpler fill pattern" string that was emitted for EVERY reference failure
    regardless of cause. Measured across the 260 proposals in
    adversarial_results/search_history.db, that advice was the correct
    diagnosis for at most 9 of 122 reference failures (7%) -- the
    precision_coercion cases -- and was actively misleading for the 44
    rank/shape crashes. It is how causal_flash_attention burned a
    120-proposal budget while never being told its tensors were the wrong
    rank: 27% of its proposals were (B, H, N, D) batched attention against a
    reference documented as 2-D only, and every one of them was answered with
    "reduce magnitude". See adversarial_results/CFA_NONHIT_ROOTCAUSE.md.

    Only the precision branch mentions magnitude. That is deliberate.
    """
    failed = {r["check_name"] for r in check_results if not r["passed"]}

    if not check_results:
        # The subprocess raised before any check ran, so check_results is
        # empty (see _run_in_subprocess's except branch). The old code did
        # `for r in failed[:2]` over an empty list and silently emitted NO
        # hint at all -- 4 gelu proposals in the recorded history got only
        # the generic "no specific signal" fallback for a hard crash.
        return "executor_crash", (
            "The reference kernel crashed before any check could run -- the "
            "input was not merely wrong-valued, it was structurally invalid "
            "for this operator. Re-read the operator context above and match "
            "the tensor keys, RANK, and shape convention exactly. Do not "
            "adjust magnitude; it is not the problem."
        )

    if _SENTINEL_CHECKS <= failed:
        return "kernel_raised", (
            "The reference kernel RAISED on this input (both sentinel checks "
            "failed, which only happens when the kernel itself errors). This "
            "is a shape/rank/dtype mismatch, not a numeric problem. Check the "
            "tensor RANK first -- it is the most common cause -- then the "
            "per-dimension minimums and any power-of-two requirement stated "
            "in the operator context. Changing magnitude cannot fix this."
        )

    if failed & _STRUCTURAL_CHECKS:
        return "degenerate_input", (
            f"The reference failed a structural check ({', '.join(sorted(failed & _STRUCTURAL_CHECKS))}) "
            "-- the input is degenerate, not extreme. This fires when the "
            "output does not vary with the primary tensor: constant fills "
            "(all-zeros / all-ones), or companion tensors that make the "
            "primary irrelevant (e.g. attention with constant K and V, where "
            "the output is just V no matter what Q is). Vary the PRIMARY "
            "tensor with a non-constant fill such as randn, and avoid making "
            "companions constant. Reducing magnitude will not help; a smaller "
            "constant is still constant."
        )

    if failed & _PRECISION_CHECKS:
        return "precision", (
            "The reference failed a precision check -- this one IS "
            "magnitude-related. The values are large enough that fp32 and "
            "fp16 diverge, or an accumulator saturates. Reduce input "
            "magnitude by ~10x, or narrow the dynamic range, and retry."
        )

    return "property_violated", (
        f"The reference violated an algebraic property ({', '.join(sorted(failed))}) "
        "on this input. That means the input sits outside the property's own "
        "preconditions -- most often near-zero variance for a normalisation "
        "operator, or an exact tie for an order-dependent one -- so a CORRECT "
        "kernel legitimately fails it. Move away from that degenerate regime: "
        "give the input real variance, or break exact ties. This is a "
        "structural property of the input, not its scale."
    )


def build_feedback_hints(
    reference_result: KernelExecutionResult,
    mutant_results: List[KernelExecutionResult],
) -> List[str]:
    """
    Derive concrete, actionable hints from execution results.
    These are injected verbatim into the next LLM prompt.
    """
    hints = []

    if not reference_result.passed_checker:
        label, advice = _diagnose_reference_failure(reference_result.check_results)
        failed = [r for r in reference_result.check_results if not r["passed"]]
        names = ", ".join(r["check_name"] for r in failed[:2]) or "no check completed"
        details = "; ".join(str(r.get("details", ""))[:120] for r in failed[:2])
        hints.append(
            f"Reference failed [{names}]{': ' + details if details else ''}. "
            f"({label}) {advice}"
        )

    for mr in mutant_results:
        if mr.error and mr.error.error_type not in ("TimeoutError", "SubprocessCrash"):
            hints.append(
                f"Mutant {mr.kernel_id!r} crashed ({mr.error.error_type}): "
                f"{mr.error.message[:100]}. "
                f"This is a different failure mode — adjust to avoid crashes."
            )
            continue

        if mr.passed_checker:
            # Near-miss: find the check that came closest to failing
            for r in mr.check_results:
                if r["passed"] and "max_err" in r.get("details", ""):
                    m = re.search(r"max_err=([\d.e+\-]+)", r["details"])
                    if m:
                        err = float(m.group(1))
                        if err > 1e-4:
                            hints.append(
                                f"Mutant {mr.kernel_id!r} nearly failed "
                                f"[{r['check_name']}] with max_err={err:.5f}. "
                                f"Amplify the input feature that drives this check."
                            )
                            break

    if not hints:
        hints.append(
            "No specific signal. Try a qualitatively different structure: "
            "if you used large-magnitude values, try structural patches instead "
            "(spike in last tile, non-aligned shape, alternating signs). "
            "If you used patches, try a different column range."
        )

    return hints