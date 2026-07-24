import torch
import triton
import triton.language as tl


@triton.jit
def sum_reduce_kernel(output_ptr, input_ptr, input_row_stride,
                       n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    # Padding with 0.0 is the CORRECT sentinel for sum -- padded lanes
    # contribute nothing. (Contrast with max/min below, where 0.0 is wrong.)
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    result = tl.sum(row, axis=0)

    tl.store(output_ptr + row_idx, result)


def sum_reduction(x):
    """Reduces over the last dimension of a 2D tensor."""
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty((n_rows,), device=x.device, dtype=x.dtype)

    sum_reduce_kernel[(n_rows,)](
        y, x, x.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return y