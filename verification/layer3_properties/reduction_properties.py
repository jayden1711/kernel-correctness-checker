"""
verification/layer3_properties/reduction_properties.py

Shared properties for sum/mean/max/min reduction over the last dim.
All four are TRUE exactly (not approximately) for their respective op:
  - permutation invariance: reordering elements within a row cannot
    change any of sum/mean/max/min.
  - sum, mean: exact scalar-multiplication linearity, f(a*x) == a*f(x).
  - max, min: exact shift equivariance, f(x+c) == f(x)+c; and exact
    positive-scale equivariance, f(a*x) == a*f(x) for a > 0 (NOT for
    a < 0 -- max/min swap roles under negative scaling, deliberately
    not tested here to avoid a false claim).
"""

import torch


def check_permutation_invariance(kernel_fn, x: torch.Tensor, atol: float = 1e-4):
    perm = torch.randperm(x.shape[-1], device=x.device)
    out1 = kernel_fn(x)
    out2 = kernel_fn(x[..., perm])
    ok = torch.allclose(out1, out2, atol=atol)
    max_err = (out1 - out2).abs().max().item()
    return ok, f"max diff after column permutation: {max_err:.6f}"


def check_scale_linearity(kernel_fn, x: torch.Tensor, scale: float = 3.7, atol: float = 1e-3):
    """For sum/mean only -- f(a*x) == a*f(x)."""
    out1 = kernel_fn(x) * scale
    out2 = kernel_fn(x * scale)
    ok = torch.allclose(out1, out2, atol=atol, rtol=1e-3)
    max_err = (out1 - out2).abs().max().item()
    return ok, f"max diff, scaled-then-reduced vs reduced-then-scaled: {max_err:.6f}"


def check_shift_equivariance(kernel_fn, x: torch.Tensor, shift: float = 12.3, atol: float = 1e-3):
    """For max/min only -- f(x+c) == f(x)+c."""
    out1 = kernel_fn(x) + shift
    out2 = kernel_fn(x + shift)
    ok = torch.allclose(out1, out2, atol=atol)
    max_err = (out1 - out2).abs().max().item()
    return ok, f"max diff, shifted-then-reduced vs reduced-then-shifted: {max_err:.6f}"


def check_positive_scale_equivariance(kernel_fn, x: torch.Tensor, scale: float = 2.5, atol: float = 1e-3):
    """For max/min only -- f(a*x) == a*f(x), a > 0 ONLY (deliberately
    not tested for a < 0, where max/min swap roles)."""
    out1 = kernel_fn(x) * scale
    out2 = kernel_fn(x * scale)
    ok = torch.allclose(out1, out2, atol=atol, rtol=1e-3)
    max_err = (out1 - out2).abs().max().item()
    return ok, f"max diff, scaled-then-reduced vs reduced-then-scaled (a>0): {max_err:.6f}"
