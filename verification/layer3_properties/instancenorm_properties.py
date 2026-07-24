"""
verification/layer3_properties/instancenorm_properties.py

InstanceNorm reduces per (batch, channel) instance over spatial dims
only (unlike LayerNorm/GroupNorm's different reduction scope), so these
checks reduce over dims (2, 3, ...) rather than the last dim alone.
  - zero_mean / unit_variance: true per-instance when affine is identity
    (gamma=1, beta=0) -- same caveat as layernorm's existing properties.
  - positive_scale_invariance: normalize(c*x) == normalize(x) for c>0,
    holding affine params fixed (verified true for c>0 only, same
    reasoning as every other norm operator in this batch).
"""

import torch


def check_zero_mean(output: torch.Tensor, atol: float = 1e-3):
    # output: (N, C, *spatial) -- mean over spatial dims per (n,c)
    dims = tuple(range(2, output.dim()))
    means = output.mean(dim=dims)
    ok = bool((means.abs() < atol).all())
    return ok, f"max |mean| across (n,c) instances: {means.abs().max().item():.6f}"


def check_unit_variance(output: torch.Tensor, atol: float = 3e-2):
    dims = tuple(range(2, output.dim()))
    variances = output.var(dim=dims, unbiased=False)
    ok = bool(((variances - 1.0).abs() < atol).all())
    return ok, f"max |var - 1| across (n,c) instances: {(variances - 1.0).abs().max().item():.6f}"


def check_positive_scale_invariance(candidate_fn, x, weight, bias, scale: float = 3.1, atol: float = 1e-3):
    out1 = candidate_fn(x, weight, bias)
    out2 = candidate_fn(x * scale, weight, bias)
    ok = torch.allclose(out1, out2, atol=atol, rtol=1e-3)
    max_err = (out1 - out2).abs().max().item()
    return ok, f"max diff under positive rescale of x: {max_err:.6f}"
