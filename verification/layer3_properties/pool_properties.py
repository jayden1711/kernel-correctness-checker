"""
verification/layer3_properties/pool_properties.py

Shared properties for max_pool{1,2,3}d / avg_pool{1,2,3}d.

MaxPool -- both TRUE exactly, including at padded boundary windows
(verified by construction: the -inf padding sentinel is unaffected by
finite shifts or positive scaling, so it never wins/loses differently
under either transform):
  - shift_equivariance: maxpool(x+c) == maxpool(x) + c
  - positive_scale_equivariance: maxpool(a*x) == a*maxpool(x), a > 0

AvgPool:
  - shift_equivariance: ONLY exactly true where every pooling window is
    fully inside the input (no padding contribution) -- with
    count_include_pad=True, a padded window's divisor stays at the full
    kernel_size while padding contributes 0 (not c) to the sum, so
    avgpool(x+c) != avgpool(x)+c whenever a window touches padding. This
    check builds its OWN padding=0 call internally regardless of what
    padding the caller's kernel_size/stride/padding otherwise specify,
    so the property tested is always exactly true.
  - positive_scale_equivariance: avgpool(a*x) == a*avgpool(x), a > 0 --
    TRUE exactly even WITH padding (scale distributes through the sum;
    padding contributes 0 either way, so it's unaffected).
"""

import torch


def check_maxpool_shift_equivariance(candidate_fn, x, kernel_size, stride, padding,
                                      shift: float = 15.0, atol: float = 1e-3):
    out1 = candidate_fn(x, kernel_size=kernel_size, stride=stride, padding=padding) + shift
    out2 = candidate_fn(x + shift, kernel_size=kernel_size, stride=stride, padding=padding)
    ok = torch.allclose(out1, out2, atol=atol)
    max_err = (out1 - out2).abs().max().item()
    return ok, f"max diff, shifted-then-pooled vs pooled-then-shifted: {max_err:.6f}"


def check_maxpool_positive_scale_equivariance(candidate_fn, x, kernel_size, stride, padding,
                                               scale: float = 2.6, atol: float = 1e-3):
    out1 = candidate_fn(x, kernel_size=kernel_size, stride=stride, padding=padding) * scale
    out2 = candidate_fn(x * scale, kernel_size=kernel_size, stride=stride, padding=padding)
    ok = torch.allclose(out1, out2, atol=atol, rtol=1e-3)
    max_err = (out1 - out2).abs().max().item()
    return ok, f"max diff, scaled-then-pooled vs pooled-then-scaled: {max_err:.6f}"


def check_avgpool_shift_equivariance_no_padding(candidate_fn, x, kernel_size, stride, padding,
                                                 shift: float = 15.0, atol: float = 1e-3):
    """Ignores the caller's padding entirely -- always tests with
    padding=0 so the property stays exactly true (see module docstring)."""
    out1 = candidate_fn(x, kernel_size=kernel_size, stride=stride, padding=0) + shift
    out2 = candidate_fn(x + shift, kernel_size=kernel_size, stride=stride, padding=0)
    ok = torch.allclose(out1, out2, atol=atol)
    max_err = (out1 - out2).abs().max().item()
    return ok, f"max diff, shifted-then-pooled vs pooled-then-shifted (padding=0): {max_err:.6f}"


def check_avgpool_positive_scale_equivariance(candidate_fn, x, kernel_size, stride, padding,
                                               scale: float = 2.6, atol: float = 1e-3):
    out1 = candidate_fn(x, kernel_size=kernel_size, stride=stride, padding=padding) * scale
    out2 = candidate_fn(x * scale, kernel_size=kernel_size, stride=stride, padding=padding)
    ok = torch.allclose(out1, out2, atol=atol, rtol=1e-3)
    max_err = (out1 - out2).abs().max().item()
    return ok, f"max diff, scaled-then-pooled vs pooled-then-scaled: {max_err:.6f}"
