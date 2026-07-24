import torch
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel_cheat_skip_max(output_ptr, input_ptr, input_row_stride,
                                       output_row_stride, n_rows, n_cols,
                                       BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

# Bug: no max-subtraction stabilization. log_softmax is shift-invariant,
# so this is MATHEMATICALLY identical to the reference on any input that
# doesn't overflow exp() -- invisible on typical small-magnitude random
# test data, produces inf/nan on adversarial large-magnitude input.
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    log_softmax_output = row - tl.log(denominator)

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, log_softmax_output, mask=mask)


def log_softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    log_softmax_kernel_cheat_skip_max[(n_rows,)](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return y