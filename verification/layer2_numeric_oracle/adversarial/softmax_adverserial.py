"""
Adversarial inputs for softmax correctness verification.

Each generator targets a specific known failure mode so that a cheating
kernel is exposed even if it passes on benign random inputs.
"""

import torch


def max_in_last_tile(x: torch.Tensor) -> torch.Tensor:
    """
    Place the maximum value in the last column tile.

    Exposes first-tile-only kernels: they compute max/normalisation from
    the first tile only, so the global max is never seen and the output
    is numerically wrong.
    """
    result = torch.zeros_like(x)
    result[:, -1] = 1e9
    return result


def equal_logits(x: torch.Tensor) -> torch.Tensor:
    """
    All logits equal — the correct answer is 1/n_cols everywhere.

    A kernel that skips normalisation but returns a plausible-looking
    constant output can accidentally pass this case, but combining it
    with shift-invariance and row-sum checks closes that loophole.
    """
    return torch.ones_like(x)


def extreme_range(x: torch.Tensor) -> torch.Tensor:
    """
    Extreme dynamic range: first column 1e9, last column -1e9.

    Exposes partial-accumulation bugs and fp16 overflow in kernels
    that accumulate the exp-sum in low precision.
    """
    result = torch.randn_like(x)
    result[:, 0] = 1e9
    result[:, -1] = -1e9
    return result


def non_power_of_two(x: torch.Tensor) -> torch.Tensor:
    """
    Input with a non-power-of-two number of columns (333).

    Exposes tile-boundary bugs where the kernel writes garbage to the
    padding region and the output length is wrong, or the last partial
    tile is silently dropped.
    """
    n_rows = x.shape[0]
    return torch.randn(n_rows, 333, device=x.device, dtype=x.dtype)


def near_zero_variance(x: torch.Tensor) -> torch.Tensor:
    """
    All logits near-zero with tiny variance.

    Exposes kernels that skip the subtraction of the row-maximum for
    numerical stability: near-zero inputs produce outputs close to 1/n
    regardless, masking the omission on random data.
    """
    return torch.randn_like(x) * 1e-6


def get_adversarial_inputs(x: torch.Tensor) -> list:
    """
    Return all adversarial variants as (name, tensor) pairs.
    Called by the KernelSpec and the KernelChecker.
    """
    return [
        ("max_in_last_tile",  max_in_last_tile(x)),
        ("equal_logits",      equal_logits(x)),
        ("extreme_range",     extreme_range(x)),
        ("non_power_of_two",  non_power_of_two(x)),
        ("near_zero_variance", near_zero_variance(x)),
    ]