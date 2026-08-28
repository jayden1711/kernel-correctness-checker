"""
NEAR-MISS mutant (layernorm, target perturbation-check margin 0.8x).

The reference kernel with its output scaled by (1 + DELTA), DELTA = 0.0027771340683102608
chosen so max_err = DELTA * max|f| lands at 0.8x the adaptive
tolerance 3*P95(||f(x+d)-f(x)||) on corpus-distribution inputs
(rho = tol/max|f| measured in verification_runs/near_miss_2026-08-28/).
NOT part of the published corpus -- this family exists so tolerance
experiments have a non-flat response surface
(margin CV across input draws ~8%; margins near 1.0x genuinely
straddle the boundary seed to seed, by design).
"""
import torch
import triton
import triton.language as tl

DELTA = 0.0027771340683102608


@triton.jit
def layernorm_kernel(output_ptr, input_ptr, gamma_ptr, beta_ptr,
                     input_row_stride, output_row_stride,
                     n_rows, n_cols, eps, delta, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    mean = tl.sum(row, axis=0) / n_cols
    diff = tl.where(mask, row - mean, 0.0)
    variance = tl.sum(diff * diff, axis=0) / n_cols
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)
    out = (row - mean) / tl.sqrt(variance + eps) * gamma + beta
    out = out * (1.0 + delta)                    # THE BUG: mis-scaled epilogue
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, out, mask=mask)


def layernorm(x, gamma, beta, eps=1e-5):
    n_rows, n_cols = x.shape
    if gamma.numel() != n_cols or beta.numel() != n_cols:
        raise ValueError("layernorm: companion length mismatch")
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    layernorm_kernel[(n_rows,)](y, x, gamma, beta, x.stride(0), y.stride(0),
                                n_rows, n_cols, eps, DELTA,
                                BLOCK_SIZE=BLOCK_SIZE)
    return y
