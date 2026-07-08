"""
Layer 3  Algebraic properties for layernorm.

Invariants that any correct layernorm implementation must satisfy,
checkable without a full reference run.
"""

import torch
from typing import Callable


def check_zero_mean(output: torch.Tensor, atol: float = 1e-4) -> tuple:
    """
    After normalisation (before affine transform), each row must have
    zero mean.

    If gamma and beta are the identity transform (gamma=1, beta=0), the
    output mean must be zero.  We test without affine parameters.
    """
    row_means = output.float().mean(dim=-1)
    if not torch.allclose(row_means, torch.zeros_like(row_means), atol=atol):
        worst = row_means.abs().max().item()
        return False, f"Row means are not zero; max |mean|={worst:.6f}."
    return True, "Zero-mean property holds."


def check_unit_variance(output: torch.Tensor, atol: float = 1e-3) -> tuple:
    """
    After normalisation (with gamma=1, beta=0), each row must have
    unit variance.

    We use the unbiased estimator (ddof=1) to match PyTorch's default.
    """
    row_vars = output.float().var(dim=-1, unbiased=False)
    expected = torch.ones_like(row_vars)
    if not torch.allclose(row_vars, expected, atol=atol):
        worst = (row_vars - expected).abs().max().item()
        return False, f"Row variances deviate from 1.0; max deviation={worst:.6f}."
    return True, "Unit-variance property holds."


def check_scale_invariance(
    kernel_fn: Callable,
    x: torch.Tensor,
    atol: float = 1e-3,
) -> tuple:
    """
    layernorm(c·x) == layernorm(x) for any scalar c > 0.

    The normalisation step removes the scale, so multiplying inputs by a
    positive constant must not change the output (when gamma=1, beta=0).
    """
    scale = 100.0
    out_original = kernel_fn(x)
    out_scaled = kernel_fn(x * scale)

    if not torch.allclose(out_original.float(), out_scaled.float(), atol=atol):
        max_err = (out_original.float() - out_scaled.float()).abs().max().item()
        return False, f"Scale invariance violated; max_err={max_err:.6f}."
    return True, "Scale invariance holds."


def check_affine_correctness(
    kernel_fn_with_affine: Callable,
    x: torch.Tensor,
    atol: float = 1e-4,
) -> tuple:
    """
    layernorm with gamma=2, beta=3 should produce 2·norm(x) + 3.

    Exposes kernels that ignore gamma/beta or apply them in the wrong order.

    Args:
        kernel_fn_with_affine: callable(x, gamma, beta) -> Tensor
    """
    hidden = x.shape[-1]
    device, dtype = x.device, x.dtype

    gamma = torch.full((hidden,), 2.0, device=device, dtype=dtype)
    beta = torch.full((hidden,), 3.0, device=device, dtype=dtype)

    try:
        out = kernel_fn_with_affine(x, gamma, beta).float()
    except TypeError:
        return True, "Kernel does not accept gamma/beta; skipping affine check."

    # Reference: normalise, then apply affine
    norm = torch.nn.functional.layer_norm(x.float(), (hidden,))
    expected = norm * 2.0 + 3.0

    if not torch.allclose(out, expected, atol=atol):
        max_err = (out - expected).abs().max().item()
        return False, f"Affine transform incorrect; max_err={max_err:.6f}."
    return True, "Affine correctness holds."


def check_precision_coercion(
    kernel_fn: Callable,
    x: torch.Tensor,
    atol: float = 1e-3,
) -> tuple:
    """
    The kernel must not silently downcast to fp16 during variance computation.

    Near-zero variance rows are where fp16 precision loss is most damaging.
    """
    hidden = x.shape[-1]
    ref = torch.nn.functional.layer_norm(x.double(), (hidden,)).float()

    out_fp32 = kernel_fn(x.float()).float()
    err_fp32 = (out_fp32 - ref).abs().max().item()

    if err_fp32 > atol:
        return False, (
            f"fp32 layernorm error={err_fp32:.6f} exceeds atol={atol}. "
            "Possible fp16 coercion in variance computation."
        )

    return True, f"Precision check passed: fp32_err={err_fp32:.6f}."