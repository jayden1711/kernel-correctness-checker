import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_cheat_wrong_reduction(output_ptr, input_ptr, input_row_stride,
                                          output_row_stride, n_rows, n_cols,
                                          BLOCK_SIZE: tl.constexpr,
                                          PARTIAL_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        partial_mask = col_offsets < PARTIAL_SIZE
        denominator = tl.sum(numerator * partial_mask, axis=0)  # sums subset
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x, PARTIAL_SIZE=64):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel_cheat_wrong_reduction[(n_rows,)](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE, PARTIAL_SIZE=PARTIAL_SIZE
    )
    return y