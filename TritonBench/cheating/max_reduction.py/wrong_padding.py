import torch
import triton
import triton.language as tl


@triton.jit
def max_reduce_kernel_cheat_wrong_padding(output_ptr, input_ptr, input_row_stride,
                                           n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

# Bug: pads masked (out-of-bounds) lanes with 0.0 instead of -inf.
# Only matters when BLOCK_SIZE > n_cols (non-power-of-two n_cols).
# Invisible whenever the row's true max is positive; wrong whenever the
# true max is negative -- the fake 0.0 padding lane wins instead.
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    result = tl.max(row, axis=0)

    tl.store(output_ptr + row_idx, result)


def max_reduction(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty((n_rows,), device=x.device, dtype=x.dtype)
    max_reduce_kernel_cheat_wrong_padding[(n_rows,)](
        y, x, x.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return y