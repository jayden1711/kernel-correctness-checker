"""
verification/layer3_properties/attention_convexity_properties.py

Convex-hull boundedness: since softmax attention weights are always
non-negative and sum to 1 (whether over ALL keys, non-causal, or over
the causally-ALLOWED subset of keys), each output row is a convex
combination of some subset of V's rows. A convex combination of any
subset of a set is bounded by that set's own min/max, so:

    min(V, dim=0) <= O <= max(V, dim=0)   (elementwise, per feature)

holds for BOTH non-causal and causal attention -- causal masking only
shrinks which V rows are combined, it doesn't change that the result is
still SOME convex combination of V's rows. Verified true regardless of
which mask (if any) is applied.
"""

import torch


def check_convex_hull_bound(candidate_fn, Q, K, V, atol: float = 1e-2):
    out = candidate_fn(Q, K, V)
    v_min = V.min(dim=0).values
    v_max = V.max(dim=0).values
    below = (out >= v_min.unsqueeze(0) - atol).all()
    above = (out <= v_max.unsqueeze(0) + atol).all()
    ok = bool(below and above)
    if not ok:
        under = (v_min.unsqueeze(0) - out).clamp(min=0).max().item()
        over = (out - v_max.unsqueeze(0)).clamp(min=0).max().item()
        return ok, f"convex-hull violation: max_under={under:.6f} max_over={over:.6f}"
    return ok, "output stays within V's per-feature [min, max]"
