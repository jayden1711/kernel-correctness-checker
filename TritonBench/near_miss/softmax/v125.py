"""
VERDICT-LEVEL near-miss mutant (softmax, target margin 1.25x of the
BINDING check `adversarial_max_in_last_tile`, delta* = 1.013e-06).

The reference kernel with its output scaled by (1 + DELTA),
DELTA = 1.2665987016202375e-06. Unlike the m-series (which targets the adaptive
perturbation tolerance, 20-30x looser), DELTA here sits at 1.25x the
smallest flip-delta of ANY check in the pipeline, so the OVERALL VERDICT
straddles. Design: verification_runs/near_miss_verdict_2026-08-28/.
NOT part of the published corpus.
"""
import torch
import triton
import triton.language as tl

DELTA = 1.2665987016202375e-06


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, delta, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    out = numerator / denominator * (1.0 + delta)   # THE BUG
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, out, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0),
                              n_rows, n_cols, DELTA, BLOCK_SIZE=BLOCK_SIZE)
    return y
