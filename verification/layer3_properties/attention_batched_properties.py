"""
verification/layer3_properties/attention_batched_properties.py

Extends the flash-attention algebraic properties (sum-to-one,
bounded-by-values) to real-world kernels with (batch, heads, seq, dim)
4D inputs and an optional `causal` flag -- confirmed necessary by
TritonBench_G_v1/flash_attn.py, which the existing FlashAttentionSpec
(2D, non-causal only) cannot represent.

KEY CLAIM, stated explicitly so it can be checked rather than trusted:
Both properties survive causal masking WITHOUT modification to the
invariant itself:
  - sum-to-one: a correct causal softmax normalizes over exactly the
    positions it's allowed to attend to. Whatever that visible set is,
    the weights over it still sum to 1 for every query row. The
    invariant doesn't change; only which V=ones probe input counts as
    "correct" changes (it still does -- V=ones is unaffected by which
    subset of V a row's weights are drawn from, since every value is 1).
  - bounded-by-values: output is still a convex combination of whichever
    V rows a query is allowed to see, so it's still within the global
    min/max of V. The visible subset only shrinks the combination's
    support, never pushes it outside V's range.

This claim is NOT independently verified against a real causal
reference implementation run end-to-end here -- it's a mathematical
argument, not an empirical confirmation. Treat the first real run
against flash_attn.py as the test of this claim, not this docstring.
"""

from typing import Any, Callable, Optional
import torch


def _looks_like_qkv(args: tuple, kwargs: dict) -> Optional[dict]:
    """
    Heuristic: the first three tensor arguments (positional or keyword,
    in call order) are Q, K, V if their trailing dimension matches and
    they share at least 2 dimensions. This matches every example seen
    so far (flash_attn.py's `flash_attn_triton(q, k, v, causal=..., ...)`,
    your own AttentionKernelSpec's assumed (Q, K, V) positional order).

    Returns a dict with keys 'q_idx'/'q_kw', 'k_idx'/'k_kw', 'v_idx'/'v_kw'
    describing where each tensor was found (positional index or kwarg
    name), plus 'causal' (bool, default False), or None if this call
    doesn't look like attention at all.
    """
    tensor_slots: list[tuple[str, Any, torch.Tensor]] = []  # (locator_kind, locator, tensor)

    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            tensor_slots.append(("pos", i, a))
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            tensor_slots.append(("kw", k, v))

    if len(tensor_slots) < 3:
        return None

    q_slot, k_slot, v_slot = tensor_slots[0], tensor_slots[1], tensor_slots[2]
    q, k, v = q_slot[2], k_slot[2], v_slot[2]

    if q.dim() < 2 or k.dim() < 2 or v.dim() < 2:
        return None
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        # last dim should match (head dim), and K/V should share seq_len
        return None

    causal = False
    for k_name, v_val in kwargs.items():
        if k_name.lower() in ("causal", "is_causal") and isinstance(v_val, bool):
            causal = v_val
            break

    return {
        "q_slot": q_slot, "k_slot": k_slot, "v_slot": v_slot,
        "causal": causal,
    }


def _substitute(args: tuple, kwargs: dict, slot: tuple, new_value: torch.Tensor) -> tuple[tuple, dict]:
    """Return (new_args, new_kwargs) with the tensor at `slot` replaced."""
    kind, locator, _ = slot
    if kind == "pos":
        new_args = tuple(new_value if i == locator else a for i, a in enumerate(args))
        return new_args, dict(kwargs)
    else:
        new_kwargs = dict(kwargs)
        new_kwargs[locator] = new_value
        return args, new_kwargs


def _flatten_leading(x: torch.Tensor) -> torch.Tensor:
    """(..., N, D) -> (prod(...), N, D). No-op if already 2D or 3D."""
    if x.dim() <= 3:
        return x if x.dim() == 3 else x.unsqueeze(0)
    *lead, n, d = x.shape
    prod = 1
    for s in lead:
        prod *= s
    return x.reshape(prod, n, d)


def check_attention_weights_sum_to_one_batched(
    candidate_fn: Callable,
    args: tuple,
    kwargs: dict,
    qkv_info: dict,
    atol: float = 1e-3,
) -> tuple[bool, str]:
    """
    V=ones probe, generalized to arbitrary leading (batch, head, ...)
    dimensions and causal masking. If every row's attention weights sum
    to 1, output must be exactly 1.0 everywhere V was replaced with ones
    -- true per-row regardless of batch/head axis or which positions a
    causal mask made visible.
    """
    v_slot = qkv_info["v_slot"]
    v_tensor = v_slot[2]
    ones_v = torch.ones_like(v_tensor)
    new_args, new_kwargs = _substitute(args, kwargs, v_slot, ones_v)

    try:
        out = candidate_fn(*new_args, **new_kwargs)
    except Exception as e:
        return False, f"Exception during V=ones probe: {e}"

    if not isinstance(out, torch.Tensor):
        return False, f"Expected tensor output, got {type(out).__name__}"

    out_f = out.float()
    if not torch.allclose(out_f, torch.ones_like(out_f), atol=atol):
        max_dev = (out_f - torch.ones_like(out_f)).abs().max().item()
        return False, f"Attention weights do not sum to 1 (batched/causal V=ones test); max deviation={max_dev:.6f}"
    return True, "Attention weights sum to 1 per query (batched, causal={}).".format(qkv_info["causal"])


def check_output_bounded_by_values_batched(
    candidate_fn: Callable,
    args: tuple,
    kwargs: dict,
    qkv_info: dict,
    atol: float = 1e-4,
) -> tuple[bool, str]:
    """
    Output must lie within [min(V), max(V)] globally -- a convex
    combination over a causally-restricted subset of V rows is still
    within V's global range, since restricting the support of a convex
    combination can only shrink its reachable set, never expand it.
    """
    v_slot = qkv_info["v_slot"]
    v_tensor = v_slot[2]

    try:
        out = candidate_fn(*args, **kwargs)
    except Exception as e:
        return False, f"Exception computing output for bounds check: {e}"

    if not isinstance(out, torch.Tensor):
        return False, f"Expected tensor output, got {type(out).__name__}"

    v_min = v_tensor.float().min().item() - atol
    v_max = v_tensor.float().max().item() + atol
    out_f = out.float()

    if out_f.min().item() < v_min or out_f.max().item() > v_max:
        return False, (
            f"Output out of value range [{v_min:.4f}, {v_max:.4f}]: "
            f"got [{out_f.min().item():.4f}, {out_f.max().item():.4f}]"
        )
    return True, "Output bounded by V's value range (batched/causal)."


def try_attention_layer3(
    candidate_fn: Callable,
    args: tuple,
    kwargs: dict,
) -> Optional[dict]:
    """
    Entry point called from the generic adapter's per-call check. Returns
    None (meaning: not applicable, skip silently) if this call doesn't
    look like attention. Returns a dict of {check_name: (bool, detail)}
    if it does.

    IMPORTANT CAVEAT, stated rather than hidden: the V=ones and
    bounds-check probes here call `candidate_fn` a SECOND and THIRD time
    with substituted arguments. If the candidate has side effects, non-
    determinism beyond floating point (e.g. dropout without a fixed
    seed), or mutates its input tensors in place, these extra calls can
    give misleading results. None of the 4 reference files checked so
    far do this, but it hasn't been verified as a general absence across
    TritonBench-G.
    """
    qkv_info = _looks_like_qkv(args, kwargs)
    if qkv_info is None:
        return None

    results: dict[str, tuple] = {}
    try:
        results["attention_sum_to_one_batched"] = check_attention_weights_sum_to_one_batched(
            candidate_fn, args, kwargs, qkv_info
        )
    except Exception as e:
        results["attention_sum_to_one_batched"] = (False, f"check errored: {e}")

    try:
        results["attention_bounded_by_values_batched"] = check_output_bounded_by_values_batched(
            candidate_fn, args, kwargs, qkv_info
        )
    except Exception as e:
        results["attention_bounded_by_values_batched"] = (False, f"check errored: {e}")

    return results