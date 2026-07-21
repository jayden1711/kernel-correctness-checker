"""
verification/layer3_properties/rmsnorm_generic_properties.py

Generic (signature-agnostic) gamma-correctness probe for rmsnorm-shaped
calls captured by the TritonBench adapter.

Detection heuristic: exactly TWO tensor arguments, where the second is
1-D and matches the first's trailing dimension. Distinguishes from
layernorm (which has THREE tensor args: x, gamma, beta).
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


def _looks_like_rmsnorm(args: tuple, kwargs: dict) -> Optional[dict]:
    """
    Heuristic: exactly TWO tensor arguments.  First is >=2-D (the input),
    second is 1-D with length matching the first's trailing dimension
    (gamma / weight).

    Deliberately strict on count=2 to avoid firing on layernorm (3 tensors)
    or matmul (2 tensors but second is >=2-D).
    """
    slots = _tensor_slots(args, kwargs)
    if len(slots) != 2:
        return None

    x_slot, gamma_slot = slots[0], slots[1]
    x, gamma = x_slot[2], gamma_slot[2]

    if x.dim() < 2:
        return None
    if gamma.dim() != 1 or gamma.shape[0] != x.shape[-1]:
        return None

    return {"x_slot": x_slot, "gamma_slot": gamma_slot}


def check_rmsnorm_gamma_generic(
    candidate_fn: Callable,
    reference_fn: Callable,
    args: tuple,
    kwargs: dict,
    rn_info: dict,
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> tuple:
    """
    Perturb gamma to a non-identity value (gamma * 2 + 0.5) and compare
    candidate against reference.  Same pattern as the layernorm affine
    probe — catches kernels that load but ignore the scale parameter.
    """
    gamma_slot = rn_info["gamma_slot"]
    gamma_orig = gamma_slot[2]

    gamma_adv = gamma_orig * 2.0 + 0.5
    new_args, new_kwargs = _substitute(args, kwargs, gamma_slot, gamma_adv)

    try:
        cand_out = candidate_fn(*new_args, **new_kwargs)
        ref_out = reference_fn(*new_args, **new_kwargs)
    except Exception as e:
        return False, f"Exception during non-identity gamma probe: {e}"

    if not (isinstance(cand_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)):
        return False, (
            f"Non-tensor output during gamma probe "
            f"(cand={type(cand_out).__name__}, ref={type(ref_out).__name__})"
        )
    if cand_out.shape != ref_out.shape:
        return False, f"Shape mismatch during gamma probe: {tuple(cand_out.shape)} vs {tuple(ref_out.shape)}"

    max_err = (cand_out.float() - ref_out.float()).abs().max().item()
    ok = torch.allclose(cand_out.float(), ref_out.float(), atol=atol, rtol=rtol)
    return ok, f"Non-identity gamma probe (gamma*2+0.5): max_err={max_err:.6f}"


def try_rmsnorm_layer3(
    candidate_fn: Callable,
    reference_fn: Callable,
    args: tuple,
    kwargs: dict,
) -> Optional[dict]:
    """
    Entry point for the adapter's per-call check.  Returns None if this
    call doesn't look like rmsnorm.
    """
    rn_info = _looks_like_rmsnorm(args, kwargs)
    if rn_info is None:
        return None

    try:
        ok, detail = check_rmsnorm_gamma_generic(candidate_fn, reference_fn, args, kwargs, rn_info)
        return {"rmsnorm_gamma_correctness_generic": (ok, detail)}
    except Exception as e:
        return {"rmsnorm_gamma_correctness_generic": (False, f"check errored: {e}")}