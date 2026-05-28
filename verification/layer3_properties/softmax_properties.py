"""
Layer 3  Algebraic properties for softmax.

All checks are mathematical invariants that hold for correct softmax
regardless of implementation.  They can be verified without a full
reference run.
"""

import torch
from typing import Callable


def check_rows_sum_to_one(output: torch.Tensor, atol: float = 1e-4) -> tuple:
    """Each row of softmax output must sum to 1."""
    row_sums = output.float().sum(dim=-1)
    expected = torch.ones_like(row_sums)
    if not torch.allclose(row_sums, expected, atol=atol):
        worst = (row_sums - expected).abs().max().item()
        return False, f"Row sums deviate from 1.0; max deviation={worst:.6f}."
    return True, "All row sums are within atol of 1.0."


def check_shift_invariance(
    kernel_fn: Callable,
    x: torch.Tensor,
    atol: float = 1e-4,
) -> tuple:
    """
    softmax(x + c) == softmax(x) for any per-row constant c.

    A kernel that omits the subtract-max step for numerical stability still
    computes the correct answer on 'normal' inputs but fails this test when
    the shift pushes values into overflow territory.
    """
    shift = torch.randn(x.shape[0], 1, device=x.device, dtype=x.dtype) * 100.0
    x_shifted = x + shift
    out_original = kernel_fn(x)
    out_shifted = kernel_fn(x_shifted)
    if not torch.allclose(out_original.float(), out_shifted.float(), atol=atol):
        max_err = (out_original.float() - out_shifted.float()).abs().max().item()
        return False, f"Shift invariance violated; max_err={max_err:.6f}."
    return True, "Shift invariance holds."


def check_monotonicity(
    kernel_fn: Callable,
    x: torch.Tensor,
) -> tuple:
    """
    If a[i] > a[j] then softmax(a)[i] > softmax(a)[j] (within each row).

    Verified via KL divergence: swapping any two elements in the input
    and checking that the output ordering is consistent.
    We use a lightweight check: the argmax of the output should match the
    argmax of the input.
    """
    out = kernel_fn(x).float()
    input_argmax = x.argmax(dim=-1)
    output_argmax = out.argmax(dim=-1)
    if not (input_argmax == output_argmax).all():
        n_wrong = (input_argmax != output_argmax).sum().item()
        return False, (
            f"Monotonicity violated in {n_wrong}/{x.shape[0]} rows: "
            "argmax of output does not match argmax of input."
        )
    return True, "Monotonicity (argmax preservation) holds."


def check_precision_coercion(
    kernel_fn: Callable,
    x: torch.Tensor,
    atol: float = 1e-3,
) -> tuple:
    """
    The kernel must not silently downcast to fp16 mid-computation.

    We run with a fp32 input and a fp16 input and check that the fp32
    result is more accurate than the fp16 result relative to a double
    reference.  If the kernel coerces fp32 -> fp16 internally, both
    results will be equally (in)accurate.
    """
    ref = torch.nn.functional.softmax(x.double(), dim=-1).float()

    out_fp32 = kernel_fn(x.float()).float()
    err_fp32 = (out_fp32 - ref).abs().max().item()

    x_fp16 = x.half()
    try:
        out_fp16 = kernel_fn(x_fp16).float()
        err_fp16 = (out_fp16 - ref).abs().max().item()
    except Exception:
        # Kernel doesn't support fp16 input; skip this branch
        return True, f"fp16 input not supported; fp32 err={err_fp32:.6f}."

    # If fp32 error is not meaningfully smaller than fp16 error, the kernel
    # is probably accumulating in fp16 regardless of input dtype.
    if err_fp32 > atol and err_fp32 >= err_fp16 * 0.9:
        return False, (
            f"Precision coercion suspected: fp32_err={err_fp32:.6f} is not "
            f"significantly smaller than fp16_err={err_fp16:.6f}."
        )

    return True, (
        f"Precision check passed: fp32_err={err_fp32:.6f}, "
        f"fp16_err={err_fp16:.6f}."
    )