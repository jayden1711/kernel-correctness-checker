"""
VERDICT-LEVEL near-miss mutant (sum_reduction, target margin 2.0x of the
BINDING check `cross_shape`, delta* = 1.011e-04).

The reference kernel with its output scaled by (1 + DELTA),
DELTA = 0.000202298164368114. Unlike the m-series (which targets the adaptive
perturbation tolerance, 20-30x looser), DELTA here sits at 2.0x the
smallest flip-delta of ANY check in the pipeline, so the OVERALL VERDICT
straddles. Design: verification_runs/near_miss_verdict_2026-08-28/.
NOT part of the published corpus.
"""
import torch
import triton
import triton.language as tl

DELTA = 0.000202298164368114


@triton.jit
def sum_reduce_kernel(output_ptr, input_ptr, input_row_stride,
                      n_rows, n_cols, delta, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    result = tl.sum(row, axis=0) * (1.0 + delta)     # THE BUG
    tl.store(output_ptr + row_idx, result)


def sum_reduction(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty((n_rows,), device=x.device, dtype=x.dtype)
    sum_reduce_kernel[(n_rows,)](y, x, x.stride(0), n_rows, n_cols, DELTA,
                                 BLOCK_SIZE=BLOCK_SIZE)
    return y
