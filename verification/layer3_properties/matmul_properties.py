"""
Layer 3  Algebraic properties for matrix multiplication.

Mathematical invariants that any correct matmul must satisfy.
"""

import torch
from typing import Callable


def check_output_shape(
    candidate_fn: Callable,
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple:
    """Output shape must be exactly M x N."""
    M = A.shape[0]
    N = B.shape[-1]
    try:
        out = candidate_fn(A, B)
    except Exception as e:
        return False, f"Exception: {e}"
    if out.shape != (M, N):
        return False, f"Shape mismatch: got {tuple(out.shape)}, expected ({M}, {N})."
    return True, f"Output shape ({M}, {N}) correct."


def check_distributivity(
    kernel_fn: Callable,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    atol: float = 2e-3,
    rtol: float = 1e-3,
) -> tuple:
    """
    A @ (B + C) == A @ B + A @ C.

    Exposes kernels with incorrect tile accumulation that happens to pass
    on a single pair of matrices.

    FIXED: atol=1e-4 with torch.allclose's default rtol=1e-5 used to
    false-positive on a CORRECT reference matmul -- confirmed via a real
    run (max_err=0.001221-0.001465 on both CPU and GPU, ~10-15x the old
    atol). The failure is concentrated on individual near-zero output
    elements: summing K signed products that happen to net close to
    zero is exactly where fp32 catastrophic cancellation produces
    disproportionate error relative to that element's own tiny
    magnitude, so a bare small atol (with no rtol floor worth anything)
    isn't a safe comparison here. atol=2e-3 gives those elements a
    reasonable floor; rtol=1e-3 handles larger-magnitude elements --
    both comfortably tighter than what an actual accumulation bug
    produces (partial_k_reduct's real signal shows up as max_err=23+ on
    the numeric layer, not a few-thousandths-scale rounding gap).
    """
    try:
        lhs = kernel_fn(A, B + C).float()
        rhs = (kernel_fn(A, B) + kernel_fn(A, C)).float()
    except Exception as e:
        return False, f"Exception during distributivity check: {e}"

    if not torch.allclose(lhs, rhs, atol=atol, rtol=rtol):
        max_err = (lhs - rhs).abs().max().item()
        return False, f"Distributivity violated; max_err={max_err:.6f}."
    return True, "Distributivity A@(B+C) == A@B + A@C holds."


def check_scalar_associativity(
    kernel_fn: Callable,
    A: torch.Tensor,
    B: torch.Tensor,
    atol: float = 2e-3,
    rtol: float = 1e-3,
) -> tuple:
    """
    (c·A) @ B == c · (A @ B) for scalar c.

    A kernel that accumulates incorrectly will fail this for large c.

    FIXED: same issue and same fix as check_distributivity above --
    confirmed false-positiving on a correct reference matmul
    (max_err=0.001221-0.001465, both CPU and GPU) from ordinary fp32
    cancellation on near-zero output elements, not a real bug.
    """
    c = 100.0
    try:
        lhs = kernel_fn(c * A, B).float()
        rhs = (c * kernel_fn(A, B)).float()
    except Exception as e:
        return False, f"Exception during scalar-associativity check: {e}"

    if not torch.allclose(lhs, rhs, atol=atol, rtol=rtol):
        max_err = (lhs - rhs).abs().max().item()
        return False, f"Scalar associativity violated; max_err={max_err:.6f}."
    return True, "Scalar associativity (c·A)@B == c·(A@B) holds."


def check_precision_coercion(
    kernel_fn: Callable,
    A: torch.Tensor,
    B: torch.Tensor,
    atol: float = 1e-3,
) -> tuple:
    """
    fp32 matmul must be more accurate than fp16 matmul vs a double reference.

    A kernel that accumulates in fp16 regardless of input dtype will show
    similar error for both fp32 and fp16 inputs.
    """
    ref = (A.double() @ B.double()).float()

    out_fp32 = kernel_fn(A.float(), B.float()).float()
    err_fp32 = (out_fp32 - ref).abs().max().item()

    try:
        out_fp16 = kernel_fn(A.half(), B.half()).float()
        err_fp16 = (out_fp16 - ref).abs().max().item()
    except Exception:
        return True, f"fp16 not supported; fp32 err={err_fp32:.6f}."

    if err_fp32 > atol and err_fp32 >= err_fp16 * 0.9:
        return False, (
            f"Precision coercion suspected: fp32_err={err_fp32:.6f}, "
            f"fp16_err={err_fp16:.6f}."
        )

    return True, (
        f"Precision check passed: fp32_err={err_fp32:.6f}, "
        f"fp16_err={err_fp16:.6f}."
    )