"""
Adversarial inputs for layernorm  returns adversarial x tensors only.
The spec wraps them with the appropriate gamma/beta.

Targets each cheating kernel:
  ignore_gamma_beta.py      loads but ignores gamma/beta
  skip_mean_subtract.py     divides raw x by std instead of (x - mean)
  wrong_variance_estimate   uses E[x^2] - mean^2 instead of E[(x-mean)^2]
                             numerically diverges when mean is large
"""

import torch


def skip_mean_subtract(x: torch.Tensor) -> torch.Tensor:
    """
    Large per-row mean shift.
    skip_mean_subtract.py divides raw x by std  output mean won't be zero.
    wrong_variance_estimate.py also diverges when mean >> 0.
    """
    shifts = torch.linspace(100.0, 1000.0, x.shape[0],
                            device=x.device, dtype=x.dtype).unsqueeze(1)
    return torch.randn_like(x) + shifts


def zero_variance_rows(x: torch.Tensor) -> torch.Tensor:
    """
    Half the rows are constant (zero variance).
    Exposes division-by-zero handling and wrong eps placement.
    """
    result = torch.zeros_like(x)
    n_zero = x.shape[0] // 2
    result[n_zero:] = torch.randn(
        x.shape[0] - n_zero, x.shape[-1], device=x.device, dtype=x.dtype
    )
    return result


def large_variance(x: torch.Tensor) -> torch.Tensor:
    """
    Very large values  exposes fp16 overflow in the squaring step of
    wrong_variance_estimate.py (x^2 overflows fp16 when x ~ 1e4).
    """
    return torch.randn_like(x) * 1e4


def wrong_variance_trigger(x: torch.Tensor) -> torch.Tensor:
    """
    Large mean with moderate variance  maximises the numerical difference
    between E[(x-mean)^2] and E[x^2] - mean^2.
    This is the exact condition under which wrong_variance_estimate.py fails.
    """
    mean_val = 1000.0
    return torch.randn_like(x) + mean_val


def non_power_of_two(x: torch.Tensor) -> torch.Tensor:
    """Non-power-of-two hidden dimension  exposes tile-boundary bugs."""
    n_rows = x.shape[0]
    return torch.randn(n_rows, 333, device=x.device, dtype=x.dtype)


def get_adversarial_inputs(x: torch.Tensor) -> list:
    """Return (name, adversarial_x) pairs. Spec adds gamma/beta."""
    return [
        ("skip_mean_subtract",    skip_mean_subtract(x)),
        ("zero_variance_rows",    zero_variance_rows(x)),
        ("large_variance",        large_variance(x)),
        ("wrong_variance_trigger", wrong_variance_trigger(x)),
        ("non_power_of_two",      non_power_of_two(x)),
    ]