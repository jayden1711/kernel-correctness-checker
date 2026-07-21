"""
Adversarial inputs for RMSNorm — returns adversarial x tensors only.
The spec wraps them with the appropriate gamma.

Targets known failure modes:
  ignore_gamma        loaded but never applied
  wrong_norm          uses mean(|x|) instead of sqrt(mean(x^2))
  partial_reduction   only reduces over first half of columns
"""

import torch


def large_magnitude(x: torch.Tensor) -> torch.Tensor:
    """
    Very large values — exposes fp16 overflow in the x^2 step.
    sqrt(mean(x^2)) overflows fp16 when x ~ 1e4.
    """
    return torch.randn_like(x) * 1e4


def near_zero(x: torch.Tensor) -> torch.Tensor:
    """
    Near-zero input — tests eps handling.
    A kernel that omits eps will divide by ~0 and produce Inf/NaN.
    """
    return torch.randn_like(x) * 1e-8


def non_power_of_two(x: torch.Tensor) -> torch.Tensor:
    """Non-power-of-two hidden dimension — exposes tile-boundary bugs."""
    n_rows = x.shape[0]
    return torch.randn(n_rows, 333, device=x.device, dtype=x.dtype)


def constant_rows(x: torch.Tensor) -> torch.Tensor:
    """
    All elements in each row identical (but different across rows).
    Output should be gamma * sign(x) for nonzero x (since x/RMS(x) = sign(x)
    when all elements are equal).
    Catches kernels with wrong reduction axis.
    """
    vals = torch.randn(x.shape[0], 1, device=x.device, dtype=x.dtype) * 10.0
    return vals.expand_as(x).contiguous()


def large_variance(x: torch.Tensor) -> torch.Tensor:
    """
    Extreme spread — first half of columns near zero, second half very large.
    Catches partial_reduction: if only the first half is reduced, the RMS
    is near zero, inflating the output wildly.
    """
    result = torch.zeros_like(x)
    mid = x.shape[-1] // 2
    result[:, mid:] = torch.randn(x.shape[0], x.shape[-1] - mid,
                                   device=x.device, dtype=x.dtype) * 1e4
    return result


def get_adversarial_inputs(x: torch.Tensor) -> list:
    """Return (name, adversarial_x) pairs. Spec adds gamma."""
    return [
        ("large_magnitude",  large_magnitude(x)),
        ("near_zero",        near_zero(x)),
        ("non_power_of_two", non_power_of_two(x)),
        ("constant_rows",    constant_rows(x)),
        ("large_variance",   large_variance(x)),
    ]