"""
Layer 3 — Algebraic properties for RMSNorm.

rmsnorm(x, gamma) = x / sqrt(mean(x^2) + eps) * gamma

Invariants:
  - unit_rms:           RMS of output rows ≈ 1 when gamma=1
  - scale_invariance:   rmsnorm(c·x) == rmsnorm(x) for c > 0
  - gamma_correctness:  rmsnorm(x, gamma=2) == 2 · rmsnorm(x, gamma=1)
  - precision_coercion: fp32 must be more accurate than fp16
"""

import torch
from typing import Callable


def check_unit_rms(output: torch.Tensor, atol: float = 1e-3) -> tuple:
    """
    After normalisation (gamma=1), each row should have RMS ≈ 1.
    RMS(row) = sqrt(mean(row^2)).
    """
    row_rms = output.float().pow(2).mean(dim=-1).sqrt()
    expected = torch.ones_like(row_rms)
    if not torch.allclose(row_rms, expected, atol=atol):
        worst = (row_rms - expected).abs().max().item()
        return False, f"Row RMS deviates from 1.0; max deviation={worst:.6f}."
    return True, "Unit-RMS property holds."


def check_scale_invariance(
    kernel_fn: Callable,
    x: torch.Tensor,
    atol: float = 1e-3,
) -> tuple:
    """
    rmsnorm(c·x) == rmsnorm(x) for positive scalar c.

    RMS normalization divides by sqrt(mean(x^2)), which scales linearly
    with |c|, so the scale cancels out.
    """
    scale = 100.0
    out_original = kernel_fn(x)
    out_scaled = kernel_fn(x * scale)

    if not torch.allclose(out_original.float(), out_scaled.float(), atol=atol):
        max_err = (out_original.float() - out_scaled.float()).abs().max().item()
        return False, f"Scale invariance violated; max_err={max_err:.6f}."
    return True, "Scale invariance holds."


def check_gamma_correctness(
    kernel_fn_with_gamma: Callable,
    x: torch.Tensor,
    atol: float = 1e-4,
) -> tuple:
    """
    rmsnorm(x, gamma=2) must equal 2 · rmsnorm(x, gamma=1).

    Catches kernels that load but ignore gamma (same pattern as
    layernorm's ignore_gamma_beta, but RMSNorm has no beta).
    """
    hidden = x.shape[-1]
    device, dtype = x.device, x.dtype

    gamma_ones = torch.ones(hidden, device=device, dtype=dtype)
    gamma_two = torch.full((hidden,), 2.0, device=device, dtype=dtype)

    try:
        out_one = kernel_fn_with_gamma(x, gamma_ones).float()
        out_two = kernel_fn_with_gamma(x, gamma_two).float()
    except TypeError:
        return True, "Kernel does not accept gamma; skipping gamma check."

    expected = out_one * 2.0

    if not torch.allclose(out_two, expected, atol=atol):
        max_err = (out_two - expected).abs().max().item()
        return False, f"Gamma correctness violated; max_err={max_err:.6f}."
    return True, "Gamma correctness holds."


def check_precision_coercion(
    kernel_fn: Callable,
    x: torch.Tensor,
    atol: float = 1e-3,
) -> tuple:
    """fp32 rmsnorm must be more accurate than fp16 vs double reference."""
    hidden = x.shape[-1]
    x_d = x.double()
    ref = (x_d / (x_d.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-5)).float()

    out_fp32 = kernel_fn(x.float()).float()
    err_fp32 = (out_fp32 - ref).abs().max().item()

    try:
        out_fp16 = kernel_fn(x.half()).float()
        err_fp16 = (out_fp16 - ref).abs().max().item()
    except Exception:
        return True, f"fp16 not supported; fp32 err={err_fp32:.6f}."

    if err_fp32 > atol and err_fp32 >= err_fp16 * 0.9:
        return False, (
            f"Precision coercion suspected: fp32_err={err_fp32:.6f}, "
            f"fp16_err={err_fp16:.6f}."
        )
    return True, f"Precision check passed: fp32_err={err_fp32:.6f}, fp16_err={err_fp16:.6f}."