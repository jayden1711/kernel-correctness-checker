"""
VERDICT-LEVEL near-miss mutant (l2norm, target margin 2.0x of the
BINDING check `cross_shape`, delta* = 3.412e-04).

The reference kernel with its output scaled by (1 + DELTA),
DELTA = 0.0006824731826777975. Unlike the m-series (which targets the adaptive
perturbation tolerance, 20-30x looser), DELTA here sits at 2.0x the
smallest flip-delta of ANY check in the pipeline, so the OVERALL VERDICT
straddles. Design: verification_runs/near_miss_verdict_2026-08-28/.
NOT part of the published corpus.
"""
import torch
import triton
import triton.language as tl

DELTA = 0.0006824731826777975


@triton.jit
def l2norm_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                  n_rows, n_cols, eps, delta, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    sq_sum = tl.sum(row * row, axis=0)
    norm = tl.sqrt(sq_sum + eps)
    output = row / norm * (1.0 + delta)              # THE BUG
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, output, mask=mask)


def l2norm(x, eps=1e-12):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    l2norm_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_rows, n_cols,
                             eps, DELTA, BLOCK_SIZE=BLOCK_SIZE)
    return y
