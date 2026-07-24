"""
verification/adversarial_search/executor.py

Subprocess-isolated executor.

Each execution runs in a spawned process with a hard timeout.
A crashing Triton kernel cannot kill the coordinator.
Always returns KernelExecutionResult — never raises.

The three-layer KernelChecker is the correctness arbiter.
Naive allclose is also recorded to document the checker gap.
"""

from __future__ import annotations
import multiprocessing as mp
import re
import time
import traceback
from typing import Callable, List

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


def _run_in_subprocess(
    proposal_dict: dict,
    kernel_id: str,
    candidate_src_path: str,
    reference_src_path: str,
    operator: str,
    queue: mp.Queue,
):
    import importlib.util
    import os
    import sys

    project_root = os.environ.get("CHECKER_ROOT", ".")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from verification.adversarial_search.schemas import InputProposal
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

        SPEC_MAP = {
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

        FUNC_NAMES = {
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

        def _load_fn(path: str, op: str) -> Callable:
            spec = importlib.util.spec_from_file_location("_mod", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for name in FUNC_NAMES[op]:
                fn = getattr(mod, name, None)
                if callable(fn):
                    return fn
            raise ValueError(f"No callable found in {path} for operator {op}")

        proposal = InputProposal.from_dict(proposal_dict)
        candidate_fn = _load_fn(candidate_src_path, operator)
        reference_fn = _load_fn(reference_src_path, operator)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensors = materialize_proposal(proposal, device=device)
        inputs = tensors_to_inputs(operator, tensors)

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
        spec = SPEC_MAP[operator]()
        checker = KernelChecker(spec)
        check_results = checker.run(candidate_fn, None, reference_fn, inputs)
        passed_checker = all(r.passed for r in check_results)

        result = KernelExecutionResult(
            proposal_id=proposal.proposal_id,
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

    except Exception as e:
        tb = traceback.format_exc()
        snippet = "\n".join(tb.splitlines()[-6:])
        result = KernelExecutionResult(
            proposal_id=proposal_dict.get("proposal_id", "unknown"),
            kernel_id=kernel_id,
            passed_checker=False,
            passed_naive=False,
            error=ExecutionError(
                error_type=type(e).__name__,
                message=str(e),
                layer=None,
                check_name=None,
                max_err=None,
                traceback_snippet=snippet,
            ),
            check_results=[],
            wall_time_ms=0.0,
        )

    queue.put(result.to_dict())


def execute_proposal(
    proposal: InputProposal,
    kernel_id: str,
    candidate_src_path: str,
    reference_src_path: str,
    operator: str,
    timeout_seconds: int = 30,
) -> KernelExecutionResult:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(
        target=_run_in_subprocess,
        args=(
            proposal.to_dict(),
            kernel_id,
            candidate_src_path,
            reference_src_path,
            operator,
            queue,
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
        )

    payload = queue.get()
    result = KernelExecutionResult.from_dict(payload)
    return result


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
        failed = [r for r in reference_result.check_results if not r["passed"]]
        for r in failed[:2]:
            details = r.get("details", "")
            hints.append(
                f"Reference failed [{r['check_name']}]: {details}. "
                f"Reduce input magnitude by 10x or use a simpler fill pattern."
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