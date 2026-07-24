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


def check_monotonicity(kernel_fn, x: torch.Tensor):
    out = kernel_fn(x)
    input_order = x.argsort(dim=-1)
    output_order = out.argsort(dim=-1)
    ok = torch.equal(input_order, output_order)
    return ok, "relative order preserved" if ok else "relative order NOT preserved"
