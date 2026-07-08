"""
verification/layer3_properties/softmax_generic_properties.py

Generic (signature-agnostic) softmax invariant probe for softmax-shaped
calls captured by the TritonBench adapter, closes the asymmetry where
softmax's one confirmed catch (softmax_optimize.py) worked only because
that file's OWN test happened to include a non-power-of-two shape, not
because a generalized probe existed the way layernorm and matmul now
have. Other softmax files in the corpus have no such guarantee from
their own test bodies alone.
"""

from typing import Any, Optional, Callable
import torch

from verification.layer3_properties.softmax_properties import (
    check_rows_sum_to_one,
    check_shift_invariance,
    check_monotonicity,
)


def _tensor_slots(args: tuple, kwargs: dict) -> list:
    slots = []
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            slots.append(("pos", i, a))
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            slots.append(("kw", k, v))
    return slots


def _substitute(args: tuple, kwargs: dict, slot: tuple, new_value: torch.Tensor) -> tuple[tuple, dict]:
    kind, locator, _ = slot
    if kind == "pos":
        new_args = tuple(new_value if i == locator else a for i, a in enumerate(args))
        return new_args, dict(kwargs)
    else:
        new_kwargs = dict(kwargs)
        new_kwargs[locator] = new_value
        return args, new_kwargs


def _looks_like_softmax(args: tuple, kwargs: dict) -> Optional[dict]:
    """
    Heuristic: EXACTLY ONE tensor argument, at least 2-D. Deliberately
    strict (not "at least one") to avoid firing on layernorm (has gamma/
    beta as additional tensors) or matmul/attention (2-3 tensors) --
    those are already covered by their own dedicated generic probes, and
    a call matching more than one of these heuristics simultaneously
    would be a real ambiguity worth investigating, not silently double-
    counted here.

    NOT yet verified against every softmax-shaped file in the corpus --
    a file whose entry point takes (x, dim) with dim as a tensor rather
    than an int, for instance, would not match this heuristic. Treat
    non-matches as "not tested by this probe," not "confirmed non-
    softmax."
    """
    slots = _tensor_slots(args, kwargs)
    if len(slots) != 1:
        return None
    x_slot = slots[0]
    if x_slot[2].dim() < 2:
        return None
    return {"x_slot": x_slot}


def check_softmax_invariants_at_shape(
    candidate_fn: Callable,
    args: tuple,
    kwargs: dict,
    sm_info: dict,
    shape: tuple,
) -> dict:
    """
    Rebuild the primary input at a NEW shape (fresh random tensor, same
    dtype/device as the captured original) and run the three reference-
    free softmax invariants against the candidate at that shape. Returns
    a dict of {check_name: (bool, detail)} -- may be empty if the
    candidate errors out entirely at this shape (recorded as a single
    failed entry rather than silently vanishing).
    """
    x_slot = sm_info["x_slot"]
    x_orig = x_slot[2]
    device, dtype = x_orig.device, x_orig.dtype

    x_new = torch.randn(*shape, device=device, dtype=dtype)
    new_args, new_kwargs = _substitute(args, kwargs, x_slot, x_new)

    def _cand(xi, _a=new_args, _k=new_kwargs, _slot=x_slot):
        a2, k2 = _substitute(_a, _k, _slot, xi)
        return candidate_fn(*a2, **k2)

    results = {}
    try:
        out = candidate_fn(*new_args, **new_kwargs)
    except Exception as e:
        return {f"softmax_probe_shape_{tuple(shape)}": (False, f"Exception at shape {shape}: {e}")}

    if not isinstance(out, torch.Tensor):
        return {f"softmax_probe_shape_{tuple(shape)}": (False, f"Non-tensor output: {type(out).__name__}")}

    ok, detail = check_rows_sum_to_one(out)
    results[f"softmax_rows_sum_to_one_{tuple(shape)}"] = (ok, detail)

    try:
        ok, detail = check_shift_invariance(_cand, x_new)
        results[f"softmax_shift_invariance_{tuple(shape)}"] = (ok, detail)
    except Exception as e:
        results[f"softmax_shift_invariance_{tuple(shape)}"] = (False, f"check errored: {e}")

    try:
        ok, detail = check_monotonicity(_cand, x_new)
        results[f"softmax_monotonicity_{tuple(shape)}"] = (ok, detail)
    except Exception as e:
        results[f"softmax_monotonicity_{tuple(shape)}"] = (False, f"check errored: {e}")

    return results


def try_softmax_layer3(
    candidate_fn: Callable,
    args: tuple,
    kwargs: dict,
) -> Optional[dict]:
    """
    Entry point for the adapter's per-call check. Probes at a non-power-
    of-two shape (333 columns, matching the exact shape convention
    already used by the original checker's own adversarial_non_power_of_two
    generator) in addition to whatever benign shape the file's own test
    used -- this is the piece that closes the asymmetry, since a benign
    test alone gave no such guarantee.
    """
    sm_info = _looks_like_softmax(args, kwargs)
    if sm_info is None:
        return None

    x_orig = sm_info["x_slot"][2]
    n_rows = max(x_orig.shape[0], 4)
    probe_shape = (n_rows, 333)

    try:
        return check_softmax_invariants_at_shape(candidate_fn, args, kwargs, sm_info, probe_shape)
    except Exception as e:
        return {"softmax_probe_generic": (False, f"probe errored: {e}")}