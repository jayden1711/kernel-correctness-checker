"""
NEAR-MISS mutant (gelu, target perturbation-check margin 1.25x).

The reference kernel with its output scaled by (1 + DELTA), DELTA = 0.00448045291705057
chosen so max_err = DELTA * max|f| lands at 1.25x the adaptive
tolerance 3*P95(||f(x+d)-f(x)||) on corpus-distribution inputs
(rho = tol/max|f| measured in verification_runs/near_miss_2026-08-28/).
NOT part of the published corpus -- this family exists so tolerance
experiments have a non-flat response surface
(margin CV across input draws ~9%; margins near 1.0x genuinely
straddle the boundary seed to seed, by design).
"""
import torch
import triton
import triton.language as tl

DELTA = 0.00448045291705057


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
