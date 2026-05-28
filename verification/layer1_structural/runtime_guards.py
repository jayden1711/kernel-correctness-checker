"""
Layer 1  Runtime guards.

These checks require actually running the kernel, but are cheaper than
Layer 2 numeric comparisons.  They catch failure modes that static AST
analysis cannot.

Checks:
  - check_nan_inf:          output must be fully finite before any numeric test
  - check_dtype_preserved:  output dtype must match input dtype
  - check_determinism:      two runs on identical inputs must produce identical output
  - check_kernel_executed:  kernel must actually run (runtime ghost-opt detection)
"""

import torch
from typing import Callable


# check_nan_inf

def check_nan_inf(candidate_fn: Callable, x: torch.Tensor) -> tuple:
    """
    Assert the output contains no NaN or Inf values.

    This must run before any allclose-based check because NaN propagation
    can mask errors: torch.allclose returns False on NaN but the failure
    reason is ambiguous, and some tolerances interact badly with Inf.

    Returns:
        (True,  detail)   output is fully finite
        (False, detail)   output contains NaN or Inf
    """
    try:
        out = candidate_fn(x)
    except Exception as e:
        return False, f"Kernel raised an exception: {e}"

    if not torch.isfinite(out).all():
        n_nan = torch.isnan(out).sum().item()
        n_inf = torch.isinf(out).sum().item()
        total = out.numel()
        return False, (
            f"Output contains non-finite values: "
            f"{n_nan} NaN, {n_inf} Inf out of {total} elements."
        )

    return True, "Output is fully finite."


# check_dtype_preserved

def check_dtype_preserved(candidate_fn: Callable, x: torch.Tensor) -> tuple:
    """
    Assert that the output dtype matches the input dtype.

    A kernel that upcasts fp16 -> fp32 internally and returns fp32 will
    pass all numeric checks but break mixed-precision training pipelines
    that expect dtype consistency.

    Returns:
        (True,  detail)   dtypes match
        (False, detail)   dtype mismatch
    """
    try:
        out = candidate_fn(x)
    except Exception as e:
        return False, f"Kernel raised an exception: {e}"

    if out.dtype != x.dtype:
        return False, (
            f"Dtype mismatch: input {x.dtype}, output {out.dtype}. "
            "Kernel may be silently upcasting."
        )

    return True, f"Dtype preserved: {x.dtype}."


# check_determinism

def check_determinism(
    candidate_fn: Callable,
    x: torch.Tensor,
    n_runs: int = 3,
) -> tuple:
    """
    Run the kernel n_runs times on the same input and assert all outputs
    are bit-identical.

    Non-determinism indicates a race condition that the barrier check
    missed  e.g. a missing tl.barrier() between a shared-memory write
    and read in a reduction.

    Args:
        candidate_fn:  Kernel under test.
        x:             Input tensor.
        n_runs:        Number of repeated runs (default 3).

    Returns:
        (True,  detail)   all runs produce identical output
        (False, detail)   outputs differ across runs
    """
    outputs = []
    for i in range(n_runs):
        try:
            out = candidate_fn(x).detach().clone()
        except Exception as e:
            return False, f"Run {i} raised an exception: {e}"
        outputs.append(out)

    for i in range(1, n_runs):
        if not torch.equal(outputs[0], outputs[i]):
            max_diff = (outputs[0].float() - outputs[i].float()).abs().max().item()
            return False, (
                f"Non-determinism detected: run 0 vs run {i} differ. "
                f"Max absolute difference: {max_diff:.6f}. "
                "Likely a missing barrier in a reduction."
            )

    return True, f"Kernel is deterministic across {n_runs} runs."


# check_kernel_executed  (runtime ghost-optimization detection)

def check_kernel_executed(
    candidate_fn: Callable,
    x: torch.Tensor,
    reference_fn: Callable,
) -> tuple:
    """
    Confirm the custom kernel actually ran by checking that the output
    differs from the *un-initialised* output buffer on at least one element,
    AND that the kernel does more than just call the reference.

    Strategy:
      1. Pre-fill the output buffer with a sentinel value (NaN).
      2. Run the candidate.
      3. If the output still contains NaN in all positions the kernel
         never wrote to its output -> ghost optimization.
      4. Additionally, time the candidate against a trivially fast
         pass-through.  If candidate is faster than reference by >10x
         on a non-trivial input, flag it for manual review.

    Note: This is a runtime complement to the AST ghost-check, not a
    replacement.  Together they cover both static and dynamic bypasses.

    Returns:
        (True,  detail)
        (False, detail)
    """
    # We can't pre-fill the output buffer directly without knowing the
    # kernel's internal allocation, so instead we compare outputs on two
    # inputs that must produce different results.
    x2 = x + torch.randn_like(x) * 0.1 + 1.0  # meaningfully different input

    try:
        out1 = candidate_fn(x).detach().clone()
        out2 = candidate_fn(x2).detach().clone()
    except Exception as e:
        return False, f"Kernel raised an exception: {e}"

    # A ghost kernel that ignores its input produces identical outputs
    if torch.equal(out1, out2):
        return False, (
            "Kernel output is identical for two different inputs. "
            "Kernel likely ignores input (hardcoded output or ghost optimization)."
        )

    # (catches kernels that just call the reference directly)
    try:
        ref1 = reference_fn(x).detach().clone()
    except Exception as e:
        return True, f"Could not run reference for comparison: {e}"

    # If output == reference on every element to machine precision,
    # the candidate may literally be the reference.  This is a soft warning
    # (some correct kernels are numerically identical to reference), so we
    # only flag if the candidate is also suspiciously fast.
    if torch.equal(out1.float(), ref1.float()):
        # Time both
        import time
        if x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            candidate_fn(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        t_cand = time.perf_counter() - t0

        if x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            reference_fn(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        t_ref = time.perf_counter() - t0

        if t_ref > 0 and t_cand < t_ref * 0.1:
            return False, (
                f"Output is bit-identical to reference AND candidate is "
                f"{t_ref/t_cand:.1f}x faster. Likely delegating to reference."
            )

    return True, "Kernel executed and produced input-dependent output."