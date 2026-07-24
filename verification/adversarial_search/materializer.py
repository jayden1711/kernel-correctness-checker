"""
verification/adversarial_search/materializer.py

TensorDescriptor → real torch.Tensor, and back.

The LLM never sees tensor data.  It produces symbolic TensorDescriptors;
this module executes them deterministically.  Garbage descriptors produce
a clear error here, not a silent wrong tensor downstream.
"""

from __future__ import annotations
from typing import Any, Dict

import torch

from verification.adversarial_search.schemas import (
    InputProposal,
    TensorDescriptor,
)


DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
    "int32":    torch.int32,
    "int64":    torch.int64,
}


def _materialize_one(desc: TensorDescriptor, device: str) -> torch.Tensor:
    dtype = DTYPE_MAP.get(desc.dtype, torch.float32)
    shape = tuple(desc.shape)

    if desc.fill == "randn":
        t = torch.randn(*shape, dtype=torch.float32)
    elif desc.fill == "ones":
        t = torch.ones(*shape, dtype=torch.float32)
    elif desc.fill == "zeros":
        t = torch.zeros(*shape, dtype=torch.float32)
    elif desc.fill == "arange":
        n = 1
        for s in shape:
            n *= s
        t = torch.arange(n, dtype=torch.float32).reshape(shape)
    elif desc.fill == "literal":
        if desc.literal_values is None:
            raise ValueError("fill='literal' but literal_values is None")
        t = torch.tensor(desc.literal_values, dtype=torch.float32).reshape(shape)
    else:
        raise ValueError(f"Unknown fill strategy: {desc.fill!r}")

    t = t * desc.scale + desc.shift

    for patch in desc.patches:
        idx_expr = patch["indices"]
        val = float(patch["value"])
        try:
            idx = _safe_eval_index(idx_expr, shape)
            t[idx] = val
        except Exception as e:
            raise ValueError(
                f"Could not apply patch {patch!r} to shape {shape}: {e}"
            ) from e

    return t.to(dtype=dtype, device=device)


def _safe_eval_index(expr: str, shape: tuple) -> Any:
    expr = expr.strip()
    if expr.startswith("[") and expr.endswith("]"):
        expr = expr[1:-1]
    parts = [p.strip() for p in expr.split(",")]
    parsed = tuple(_parse_part(p) for p in parts)
    return parsed[0] if len(parsed) == 1 else parsed


def _parse_part(part: str) -> Any:
    if ":" in part:
        components = part.split(":")
        def to_int_or_none(s):
            s = s.strip()
            return None if s == "" else int(s)
        if len(components) == 2:
            return slice(to_int_or_none(components[0]), to_int_or_none(components[1]))
        elif len(components) == 3:
            return slice(to_int_or_none(components[0]),
                         to_int_or_none(components[1]),
                         to_int_or_none(components[2]))
    return int(part)


def materialize_proposal(
    proposal: InputProposal,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    tensors = {}
    for name, desc in proposal.tensors.items():
        try:
            tensors[name] = _materialize_one(desc, device)
        except Exception as e:
            raise ValueError(
                f"Failed to materialize tensor '{name}': {e}"
            ) from e
    return tensors


# operator -> ordered tuple of tensor-dict keys, matching each operator's
# actual Python wrapper signature exactly. Single-tensor operators return
# the bare tensor (not a 1-tuple) to match the ORIGINAL softmax/rmsnorm
# convention already relied on elsewhere (spec.primary_input, checker.run's
# isinstance(inputs, tuple) branching) -- returning a 1-tuple instead would
# silently break that convention for every new single-tensor operator.
_OPERATOR_TENSOR_KEYS = {
    # Original 5 -- UNCHANGED, verbatim from before this edit.
    "softmax":         ("x",),
    "layernorm":       ("x", "gamma", "beta"),
    "matmul":          ("A", "B"),
    "flash_attention": ("Q", "K", "V"),
    "rmsnorm":         ("x", "gamma"),

    # 16 new operators with a valid InputProposal representation --
    # key order matches each reference/cheating wrapper's actual
    # positional signature (I wrote every one of these this conversation).
    "log_softmax":                  ("x",),
    "swish":                        ("x",),
    "gelu":                         ("x",),
    "sum_reduction":                ("x",),
    "mean_reduction":               ("x",),
    "max_reduction":                ("x",),
    "min_reduction":                ("x",),
    "l1norm":                       ("x",),
    "l2norm":                       ("x",),
    "frobenius_norm":               ("x",),
    "argmax":                       ("x",),
    "argmin":                       ("x",),
    "instancenorm":                 ("x", "weight", "bias"),
    "batchnorm":                    ("x", "running_mean", "running_var", "weight", "bias"),
    "scaled_dot_product_attention": ("Q", "K", "V"),
    "causal_flash_attention":       ("Q", "K", "V"),

    # Deliberately NOT here: cross_entropy (targets needs uniform bounded
    # int64 class indices; the randn-then-cast fill path in
    # _materialize_one produces a clustered, possibly out-of-bounds
    # distribution instead -- confirmed by reading this file, not just
    # suspected), groupnorm and the 6 pooling operators (need a scalar
    # hyperparameter -- num_groups / kernel_size / stride / padding --
    # that InputProposal/TensorDescriptor has no field for at all).
}


def tensors_to_inputs(operator: str, tensors: Dict[str, torch.Tensor]):
    """Convert materialized tensor dict to the calling convention for each operator."""
    keys = _OPERATOR_TENSOR_KEYS.get(operator)
    if keys is None:
        raise ValueError(f"Unknown operator: {operator!r}")

    if len(keys) == 1:
        return tensors[keys[0]]
    return tuple(tensors[k] for k in keys)