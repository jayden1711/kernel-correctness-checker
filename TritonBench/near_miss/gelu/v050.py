"""
VERDICT-LEVEL near-miss mutant (gelu, target margin 0.5x of the
BINDING check `adversarial_near_global_min`, delta* = 8.762e-06).

The reference kernel with its output scaled by (1 + DELTA),
DELTA = 4.3809413909360284e-06. Unlike the m-series (which targets the adaptive
perturbation tolerance, 20-30x looser), DELTA here sits at 0.5x the
smallest flip-delta of ANY check in the pipeline, so the OVERALL VERDICT
straddles. Design: verification_runs/near_miss_verdict_2026-08-28/.
NOT part of the published corpus.
"""
import torch
import triton
import triton.language as tl

DELTA = 4.3809413909360284e-06


@triton.jit
def gelu_kernel(output_ptr, input_ptr, n_elements, delta,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    y = y * (1.0 + delta)                            # THE BUG
    tl.store(output_ptr + offsets, y, mask=mask)


def gelu(x):
    x_flat = x.contiguous().view(-1)
    n_elements = x_flat.numel()
    y_flat = torch.empty_like(x_flat)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_kernel[grid](y_flat, x_flat, n_elements, DELTA,
                      BLOCK_SIZE=BLOCK_SIZE)
    return y_flat.view_as(x)
