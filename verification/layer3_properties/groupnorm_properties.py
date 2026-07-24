"""
verification/layer3_properties/groupnorm_properties.py

GroupNorm reduces per (batch, group) instance over (channels_per_group,
*spatial) -- reshape output back into that grouping before checking.
  - zero_mean / unit_variance: true per-(n,group) instance when affine
    is identity (weight=1, bias=0).
  - positive_scale_invariance: normalize(c*x) == normalize(x) for c>0,
    affine held fixed.
"""

import torch


def _group_reshape(output: torch.Tensor, num_groups: int):
    N, C = output.shape[0], output.shape[1]
    spatial_shape = output.shape[2:]
    return output.view(N, num_groups, C // num_groups, *spatial_shape)


def check_zero_mean(output: torch.Tensor, num_groups: int, atol: float = 1e-3):
    grouped = _group_reshape(output, num_groups)
    dims = tuple(range(2, grouped.dim()))
    means = grouped.mean(dim=dims)
    ok = bool((means.abs() < atol).all())
    return ok, f"max |mean| across (n,group) instances: {means.abs().max().item():.6f}"


def check_unit_variance(output: torch.Tensor, num_groups: int, atol: float = 3e-2):
    grouped = _group_reshape(output, num_groups)
    dims = tuple(range(2, grouped.dim()))
    variances = grouped.var(dim=dims, unbiased=False)
    ok = bool(((variances - 1.0).abs() < atol).all())
    return ok, f"max |var - 1| across (n,group) instances: {(variances - 1.0).abs().max().item():.6f}"


def check_positive_scale_invariance(candidate_fn, x, num_groups, weight, bias,
                                     scale: float = 2.9, atol: float = 1e-3):
    out1 = candidate_fn(x, num_groups, weight, bias)
    out2 = candidate_fn(x * scale, num_groups, weight, bias)
    ok = torch.allclose(out1, out2, atol=atol, rtol=1e-3)
    max_err = (out1 - out2).abs().max().item()
    return ok, f"max diff under positive rescale of x: {max_err:.6f}"
