"""
verification/layer3_properties/norm_properties.py

Shared properties for l1norm/l2norm/frobenius_norm. TRUE exactly
(up to eps) by construction of what "normalize" means:
  - unit_norm: the output's own norm (of the SAME kind the operator
    computes) is 1.
  - positive_scale_invariance: normalize(c*x) == normalize(x) for
    c > 0 (scale cancels in numerator and denominator). Deliberately
    NOT tested for c < 0 -- sign flips the output, not simply invariant.
"""

import torch


def check_unit_l1_norm(output: torch.Tensor, atol: float = 1e-3):
    row_norms = output.abs().sum(dim=-1)
    ok = torch.allclose(row_norms, torch.ones_like(row_norms), atol=atol)
    return ok, f"max deviation from unit L1 norm: {(row_norms - 1.0).abs().max().item():.6f}"


def check_unit_l2_norm(output: torch.Tensor, atol: float = 1e-3):
    row_norms = output.norm(p=2, dim=-1)
    ok = torch.allclose(row_norms, torch.ones_like(row_norms), atol=atol)
    return ok, f"max deviation from unit L2 norm: {(row_norms - 1.0).abs().max().item():.6f}"


def check_unit_frobenius_norm(output: torch.Tensor, atol: float = 1e-3):
    norm = output.norm(p='fro')
    ok = abs(norm.item() - 1.0) < atol
    return ok, f"Frobenius norm of output: {norm.item():.6f}"


def check_positive_scale_invariance(kernel_fn, x: torch.Tensor, scale: float = 4.2, atol: float = 1e-3):
    out1 = kernel_fn(x)
    out2 = kernel_fn(x * scale)
    ok = torch.allclose(out1, out2, atol=atol, rtol=1e-3)
    max_err = (out1 - out2).abs().max().item()
    return ok, f"max diff under positive rescale (c={scale}): {max_err:.6f}"
