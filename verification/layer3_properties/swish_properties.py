"""
verification/layer3_properties/swish_properties.py

Properties numerically verified before writing (swish(x) = x*sigmoid(x)):
  - swish(0) == 0 exactly.
  - monotonically increasing for x >= 0 (VERIFIED by dense sampling,
    0 to 100 in steps of 0.001, before writing this file -- NOT true
    globally, swish has a shallow dip to ~-0.278 around x~=-1.278, so
    this property is deliberately scoped to x >= 0 only).
"""

import torch


def check_zero_at_origin(kernel_fn, atol: float = 1e-4):
    x = torch.zeros(1, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    out = kernel_fn(x)
    ok = out.abs().item() < atol
    return ok, f"swish(0) = {out.item():.6f}"


def check_monotonic_nonneg(kernel_fn, x: torch.Tensor):
    """Only meaningful/valid on non-negative inputs -- caller must
    ensure x >= 0 (see swish.py's algebraic_properties wrapper)."""
    x_pos = x.abs()  # force non-negative regardless of what x was
    x_sorted, _ = torch.sort(x_pos.flatten())
    out = kernel_fn(x_sorted)
    diffs = out[1:] - out[:-1]
    ok = bool((diffs >= -1e-4).all())
    return ok, f"min step: {diffs.min().item():.6f}"
