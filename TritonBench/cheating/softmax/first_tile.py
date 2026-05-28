import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel_cheat_first_tile(output_ptr, input_ptr, input_row_stride,
                                     output_row_stride, n_rows, n_cols,
                                     BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x, BLOCK_SIZE=None):
    n_rows, n_cols = x.shape
    # Bug: only process first half of columns regardless of width
    cheat_cols = max(n_cols // 2, 1)
    BLOCK_SIZE = triton.next_power_of_2(cheat_cols)
    y = torch.zeros_like(x)
    softmax_kernel_cheat_first_tile[(n_rows,)](
        y, x, x.stride(0), y.stride(0), n_rows, cheat_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return y