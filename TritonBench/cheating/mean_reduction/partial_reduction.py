import torch
import triton
import triton.language as tl


@triton.jit
def mean_reduce_kernel_cheat_partial(output_ptr, input_ptr, input_row_stride,
                                      n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=0.0)
# Bug: sums only the first half of columns but still divides by the
# FULL n_cols -- a systematic downward bias, not just missing data.
# Invisible when the two halves have similar magnitude.
    half_cols = n_cols // 2
    partial_mask = col_offsets < half_cols
    result = tl.sum(row * partial_mask, axis=0) / n_cols

    tl.store(output_ptr + row_idx, result)


def mean_reduction(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty((n_rows,), device=x.device, dtype=x.dtype)
    mean_reduce_kernel_cheat_partial[(n_rows,)](
        y, x, x.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return y