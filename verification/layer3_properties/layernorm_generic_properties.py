"""
verification/layer3_properties/layernorm_generic_properties.py

Generic (signature-agnostic) affine-correctness probe for layernorm-shaped
calls captured by the TritonBench adapter.
"""

from typing import Any, Optional, Callable
import torch


def _tensor_slots(args: tuple, kwargs: dict) -> list:
    """Collect (kind, locator, tensor) for every tensor argument, in call order."""
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


def _looks_like_layernorm(args: tuple, kwargs: dict) -> Optional[dict]:
    """
    Heuristic: primary input x (first tensor argument), plus at least two
    OTHER 1-D tensors whose length matches x's trailing dimension -- these
    are gamma (weight) and beta (bias) candidates, in call order.

    Confirmed to match layer_norm_triton.py's real captured call:
    layer_norm(x, (feature_dim,), weight, bias, eps) -- x is (32,512),
    weight and bias are both (512,), matching x.shape[-1].

    NOT yet verified against other layernorm-shaped files in TritonBench-G
    that may use different argument orders or additional 1-D tensors
    (e.g. a running mean/var) that could be mistaken for gamma/beta.
    """
    slots = _tensor_slots(args, kwargs)
    if len(slots) < 3:
        return None

    x_slot = slots[0]
    x = x_slot[2]
    if x.dim() < 1:
        return None
    hidden = x.shape[-1]

    candidates = [s for s in slots[1:] if s[2].dim() == 1 and s[2].shape[0] == hidden]
    if len(candidates) < 2:
        return None

    return {"x_slot": x_slot, "gamma_slot": candidates[0], "beta_slot": candidates[1]}


def check_layernorm_affine_generic(
    candidate_fn: Callable,
    reference_fn: Callable,
    args: tuple,
    kwargs: dict,
    ln_info: dict,
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> tuple:
    """
    Perturb gamma and beta to non-identity values (scale gamma, shift beta
    away from the identity transform's ones/zeros -- transforming the
    CAPTURED tensors rather than constructing fresh ones, so this respects
    whatever dtype/device the real call actually used) and compare
    candidate_fn against reference_fn directly at those values.
    """
    gamma_slot = ln_info["gamma_slot"]
    beta_slot = ln_info["beta_slot"]
    gamma_orig = gamma_slot[2]
    beta_orig = beta_slot[2]

    gamma_adv = gamma_orig * 2.0 + 0.5
    beta_adv = beta_orig + 3.0

    new_args, new_kwargs = _substitute(args, kwargs, gamma_slot, gamma_adv)
    new_args, new_kwargs = _substitute(new_args, new_kwargs, beta_slot, beta_adv)

    try:
        cand_out = candidate_fn(*new_args, **new_kwargs)
        ref_out = reference_fn(*new_args, **new_kwargs)
    except Exception as e:
        return False, f"Exception during non-identity-affine probe: {e}"

    if not (isinstance(cand_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)):
        return False, (
            f"Non-tensor output during affine probe "
            f"(cand={type(cand_out).__name__}, ref={type(ref_out).__name__})"
        )
    if cand_out.shape != ref_out.shape:
        return False, f"Shape mismatch during affine probe: {tuple(cand_out.shape)} vs {tuple(ref_out.shape)}"

    max_err = (cand_out.float() - ref_out.float()).abs().max().item()
    ok = torch.allclose(cand_out.float(), ref_out.float(), atol=atol, rtol=rtol)
    return ok, f"Non-identity affine probe (gamma*2+0.5, beta+3): max_err={max_err:.6f}"


def try_layernorm_layer3(
    candidate_fn: Callable,
    reference_fn: Callable,
    args: tuple,
    kwargs: dict,
) -> Optional[dict]:
    """
    Entry point for the adapter's per-call check. Returns None if this
    call doesn't look like layernorm (skip silently), else a dict with
    one check result.
    """
    ln_info = _looks_like_layernorm(args, kwargs)
    if ln_info is None:
        return None

    try:
        ok, detail = check_layernorm_affine_generic(candidate_fn, reference_fn, args, kwargs, ln_info)
        return {"layernorm_affine_correctness_generic": (ok, detail)}
    except Exception as e:
        return {"layernorm_affine_correctness_generic": (False, f"check errored: {e}")}