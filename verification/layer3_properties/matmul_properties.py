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
    atol: float = 1e-4,
) -> tuple:
    """
    A @ (B + C) == A @ B + A @ C.

    Exposes kernels with incorrect tile accumulation that happens to pass
    on a single pair of matrices.
    """
    try:
        lhs = kernel_fn(A, B + C).float()
        rhs = (kernel_fn(A, B) + kernel_fn(A, C)).float()
    except Exception as e:
        return False, f"Exception during distributivity check: {e}"

    if not torch.allclose(lhs, rhs, atol=atol):
        max_err = (lhs - rhs).abs().max().item()
        return False, f"Distributivity violated; max_err={max_err:.6f}."
    return True, "Distributivity A@(B+C) == A@B + A@C holds."


def check_scalar_associativity(
    kernel_fn: Callable,
    A: torch.Tensor,
    B: torch.Tensor,
    atol: float = 1e-4,
) -> tuple:
    """
    (c·A) @ B == c · (A @ B) for scalar c.

    A kernel that accumulates incorrectly will fail this for large c.
    """
    c = 100.0
    try:
        lhs = kernel_fn(c * A, B).float()
        rhs = (c * kernel_fn(A, B)).float()
    except Exception as e:
        return False, f"Exception during scalar-associativity check: {e}"

    if not torch.allclose(lhs, rhs, atol=atol):
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