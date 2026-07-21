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


def tensors_to_inputs(operator: str, tensors: Dict[str, torch.Tensor]):
    """Convert materialized tensor dict to the calling convention for each operator."""
    if operator == "softmax":
        return tensors["x"]
    elif operator == "layernorm":
        return (tensors["x"], tensors["gamma"], tensors["beta"])
    elif operator == "matmul":
        return (tensors["A"], tensors["B"])
    elif operator == "flash_attention":
        return (tensors["Q"], tensors["K"], tensors["V"])
    elif operator == "rmsnorm":
        return (tensors["x"], tensors["gamma"])
    else:
        raise ValueError(f"Unknown operator: {operator!r}")