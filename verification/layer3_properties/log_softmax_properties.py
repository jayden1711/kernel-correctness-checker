"""
verification/layer3_properties/log_softmax_properties.py

Properties verified true before writing (log_softmax(x) = x - max - log(sum(exp(x-max)))):
  - exp(output) sums to 1 per row -- same invariant as softmax, one exp away.
  - shift invariance: log_softmax(x + c) == log_softmax(x), exact (same
    algebraic reasoning as softmax's own shift invariance).
  - monotonicity: relative order of elements within a row is preserved.
"""

import torch


def check_exp_sums_to_one(output: torch.Tensor, atol: float = 1e-3):
    row_sums = output.exp().sum(dim=-1)
    ok = torch.allclose(row_sums, torch.ones_like(row_sums), atol=atol)
    return ok, f"max deviation from 1.0: {(row_sums - 1.0).abs().max().item():.6f}"


def check_shift_invariance(kernel_fn, x: torch.Tensor, atol: float = 1e-3):
    c = 5.0
    out1 = kernel_fn(x)
    out2 = kernel_fn(x + c)
    ok = torch.allclose(out1, out2, atol=atol)
    max_err = (out1 - out2).abs().max().item()
    return ok, f"max diff after shift: {max_err:.6f}"


def check_monotonicity(kernel_fn, x: torch.Tensor, atol: float = 1e-4):
    """
    log_softmax(x)_i = x_i - C (a per-row constant), so sorting the
    output by the SAME permutation that sorts x must give a
    non-decreasing sequence.

    FIXED: this used to require torch.equal on the two argsort index
    permutations -- exact equality of a DISCRETE ordering extracted from
    a continuous computation. With ~128 random columns per row, some
    pair of values will occasionally land close enough that the real
    kernel's floating-point rounding (in the exp/sum/log chain) flips
    their relative order by a fraction of a ULP -- correct numerical
    behavior, not a bug, but a big enough surface (any of ~n^2/2 pairs)
    that it fired intermittently (1/5 trials) on the correct reference.
    Comparing sorted-by-x output steps with a small tolerance keeps the
    property (order is preserved) while not punishing noise-level
    near-tie flips that carry no real information.
    """
    out = kernel_fn(x)
    input_order = x.argsort(dim=-1)
    out_sorted_by_input = torch.gather(out, -1, input_order)
    diffs = out_sorted_by_input[..., 1:] - out_sorted_by_input[..., :-1]
    violation = diffs < -atol
    ok = not violation.any().item()
    worst = diffs.min().item()
    if ok:
        return True, f"relative order preserved (worst step={worst:.6f})"
    return False, f"relative order NOT preserved (worst negative step={worst:.6f}, atol={atol})"
