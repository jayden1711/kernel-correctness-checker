"""
verification/layer3_properties/cross_entropy_properties.py

Properties TRUE by construction of cross-entropy loss:
  - non_negativity: loss = -log(prob), prob in (0, 1], so loss >= 0 always.
  - shift_invariance: cross_entropy(logits + c, targets) ==
    cross_entropy(logits, targets) for a per-row scalar c broadcast
    across classes -- softmax (and therefore cross-entropy) is
    shift-invariant per row. Same algebraic fact the
    missing_max_subtraction mutant exploits; this property is the
    positive-side check for it.
"""

import torch


def check_non_negativity(loss: torch.Tensor, atol: float = 1e-4):
    ok = bool((loss >= -atol).all()) if loss.dim() > 0 else bool(loss.item() >= -atol)
    val = loss.item() if loss.dim() == 0 else loss.min().item()
    return ok, f"min loss value: {val:.6f}"


def check_shift_invariance(candidate_fn, logits, targets, shift: float = 50.0, atol: float = 1e-2):
    out1 = candidate_fn(logits, targets)
    out2 = candidate_fn(logits + shift, targets)
    ok = torch.allclose(out1, out2, atol=atol, rtol=1e-3)
    max_err = (out1 - out2).abs().max().item()
    return ok, f"max diff after per-row shift of logits: {max_err:.6f}"
