"""
verification/layer3_properties/gelu_properties.py

Properties numerically verified before writing (exact erf-based GELU):
  - gelu(0) == 0 exactly.
  - monotonically increasing for x >= 0 (VERIFIED by dense sampling
    before writing this file -- NOT true globally: GELU has a shallow
    dip to ~-0.170 around x~=-0.752, same shape of caveat as swish).
"""

import torch


def check_zero_at_origin(kernel_fn, atol: float = 1e-4):
    x = torch.zeros(1, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    out = kernel_fn(x)
    ok = out.abs().item() < atol
    return ok, f"gelu(0) = {out.item():.6f}"


def check_monotonic_nonneg(kernel_fn, x: torch.Tensor):
    x_pos = x.abs()
    x_sorted, _ = torch.sort(x_pos.flatten())
    out = kernel_fn(x_sorted)
    diffs = out[1:] - out[:-1]
    ok = bool((diffs >= -1e-4).all())
    return ok, f"min step: {diffs.min().item():.6f}"
