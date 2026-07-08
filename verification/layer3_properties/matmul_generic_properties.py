"""
verification/layer3_properties/matmul_generic_properties.py

Generic (signature-agnostic) non-aligned-shape probe for matmul-shaped
calls captured by the TritonBench adapter.
"""

from typing import Any, Optional, Callable
import torch


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


def _looks_like_matmul(args: tuple, kwargs: dict) -> Optional[dict]:
    """
    Heuristic: first two tensor arguments A, B where A.shape[-1] ==
    B.shape[-2] (standard matmul contraction dimension match) and both
    are at least 2-D. Deliberately does NOT try to distinguish this from
    attention's (Q,K,V) shape beyond requiring exactly 2 matching tensors
    up front -- attention's check requires a 3rd tensor (V), so a 2-tensor
    match here that also happens to satisfy attention's heuristic simply
    means both checks may fire on the same call, which is fine; they test
    different things and both being applicable isn't a conflict.

    NOT yet verified against matmul-shaped files using non-standard
    argument order (e.g. C passed in as an output buffer, alpha/beta
    scalars mixed into positional args), only checked against
    matmul_leakyrelu.py's (A, B, activation) convention.
    """
    slots = _tensor_slots(args, kwargs)
    if len(slots) < 2:
        return None

    a_slot, b_slot = slots[0], slots[1]
    A, B = a_slot[2], b_slot[2]

    if A.dim() < 2 or B.dim() < 2:
        return None
    if A.shape[-1] != B.shape[-2]:
        return None

    return {"a_slot": a_slot, "b_slot": b_slot}


def check_matmul_nonaligned_generic(
    candidate_fn: Callable,
    reference_fn: Callable,
    args: tuple,
    kwargs: dict,
    mm_info: dict,
    pad: int = 89,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> tuple:
    """
    Build fresh A', B' at a shape offset by `pad` in every dimension from
    the captured A/B, deliberately avoiding alignment with common block
    sizes (16/32/64/128), and compare candidate_fn against reference_fn
    directly at that shape. Fresh random tensors are used (not a resize
    of the captured ones) since padding/cropping real captured data risks
    introducing unrelated shape mismatches with other non-tensor args
    (e.g. an M/N/K passed as separate scalar arguments elsewhere in the
    call) -- fresh tensors sidestep that at the cost of not reusing the
    exact captured values.
    """
    a_slot, b_slot = mm_info["a_slot"], mm_info["b_slot"]
    A_orig, B_orig = a_slot[2], b_slot[2]

    M = A_orig.shape[-2] + pad
    K = A_orig.shape[-1] + pad
    N = B_orig.shape[-1] + pad

    A_new = torch.randn(M, K, device=A_orig.device, dtype=A_orig.dtype)
    B_new = torch.randn(K, N, device=B_orig.device, dtype=B_orig.dtype)

    new_args, new_kwargs = _substitute(args, kwargs, a_slot, A_new)
    new_args, new_kwargs = _substitute(new_args, new_kwargs, b_slot, B_new)

    try:
        cand_out = candidate_fn(*new_args, **new_kwargs)
        ref_out = reference_fn(*new_args, **new_kwargs)
    except Exception as e:
        return False, f"Exception during non-aligned-shape probe (M={M},K={K},N={N}): {e}"

    if not (isinstance(cand_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)):
        return False, (
            f"Non-tensor output during non-aligned probe "
            f"(cand={type(cand_out).__name__}, ref={type(ref_out).__name__})"
        )
    if cand_out.shape != ref_out.shape:
        return False, f"Shape mismatch during non-aligned probe: {tuple(cand_out.shape)} vs {tuple(ref_out.shape)}"

    max_err = (cand_out.float() - ref_out.float()).abs().max().item()
    ok = torch.allclose(cand_out.float(), ref_out.float(), atol=atol, rtol=rtol)
    return ok, f"Non-aligned-shape probe (M={M},K={K},N={N}): max_err={max_err:.6f}"


def try_matmul_layer3(
    candidate_fn: Callable,
    reference_fn: Callable,
    args: tuple,
    kwargs: dict,
) -> Optional[dict]:
    mm_info = _looks_like_matmul(args, kwargs)
    if mm_info is None:
        return None

    try:
        ok, detail = check_matmul_nonaligned_generic(candidate_fn, reference_fn, args, kwargs, mm_info)
        return {"matmul_nonaligned_shape_generic": (ok, detail)}
    except Exception as e:
        return {"matmul_nonaligned_shape_generic": (False, f"check errored: {e}")}