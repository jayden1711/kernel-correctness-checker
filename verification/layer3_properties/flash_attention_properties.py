"""
Layer 3  Algebraic properties for flash attention.

All kernels take 2D tensors (N, D)  no batch or head dimensions.
"""

import torch
import math
from typing import Callable


def check_output_bounded_by_values(
    output: torch.Tensor,
    V: torch.Tensor,
    atol: float = 1e-4,
) -> tuple:
    """
    The attention output is a convex combination of value vectors, so each
    output element must lie within [min(V), max(V)].
    """
    v_min = V.float().min().item() - atol
    v_max = V.float().max().item() + atol
    out_f = output.float()

    if out_f.min().item() < v_min or out_f.max().item() > v_max:
        return False, (
            f"Output out of value range [{v_min:.4f}, {v_max:.4f}]: "
            f"got [{out_f.min().item():.4f}, {out_f.max().item():.4f}]."
        )
    return True, "Output bounded by value range."


def check_attention_weights_sum_to_one(
    kernel_fn: Callable,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    atol: float = 1e-3,
) -> tuple:
    """
    Use V=ones so output[i, j] = sum_k(w[i,k]) for all j.
    All output elements should equal 1.0 if attention weights sum to 1.

    Inputs are 2D (N, D).
    """
    ones_V = torch.ones_like(V)
    try:
        out = kernel_fn(Q, K, ones_V).float()
    except Exception as e:
        return False, f"Exception: {e}"

    if not torch.allclose(out, torch.ones_like(out), atol=atol):
        max_err = (out - torch.ones_like(out)).abs().max().item()
        return False, (
            f"Attention weights do not sum to 1 (V=ones test); "
            f"max deviation={max_err:.6f}."
        )
    return True, "Attention weights sum to 1 per query."


def check_precision_coercion(
    kernel_fn: Callable,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    atol: float = 1e-3,
) -> tuple:
    """
    fp32 attention must be more accurate than fp16 vs a double reference.
    Inputs are 2D (N, D).
    """
    ref = _reference_attention(Q.double(), K.double(), V.double()).float()

    out_fp32 = kernel_fn(Q.float(), K.float(), V.float()).float()
    err_fp32 = (out_fp32 - ref).abs().max().item()

    try:
        out_fp16 = kernel_fn(Q.half(), K.half(), V.half()).float()
        err_fp16 = (out_fp16 - ref).abs().max().item()
    except Exception:
        return True, f"fp16 not supported; fp32 err={err_fp32:.6f}."

    if err_fp32 > atol and err_fp32 >= err_fp16 * 0.9:
        return False, (
            f"Precision coercion suspected: fp32_err={err_fp32:.6f}, "
            f"fp16_err={err_fp16:.6f}."
        )

    return True, (
        f"Precision check passed: fp32_err={err_fp32:.6f}, "
        f"fp16_err={err_fp16:.6f}."
    )


def _reference_attention(Q, K, V):
    """Naive scaled dot-product attention for 2D inputs (N, D)."""
    scale = math.sqrt(Q.shape[-1])
    scores = (Q @ K.T) / scale          # (N, N)
    weights = torch.softmax(scores, dim=-1)
    return weights @ V                  # (N, D)