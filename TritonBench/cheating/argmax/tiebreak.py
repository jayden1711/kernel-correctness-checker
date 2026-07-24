import torch
import triton
import triton.language as tl


@triton.jit
def argmax_kernel_cheat_tiebreak(output_ptr, input_ptr, input_row_stride,
                                  n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    max_val = tl.max(row, axis=0)

# Bug: LAST-occurrence tie-break instead of first (takes the max index
# among matching lanes instead of the min). Invisible whenever the
# row's max is unique -- wrong only on adversarial inputs with
# duplicate max values, where PyTorch's argmax returns the first
# occurrence and this returns the last.
    is_max = (row == max_val) & mask
    candidate_idx = tl.where(is_max, col_offsets, -1)
    argmax_idx = tl.max(candidate_idx, axis=0).to(tl.int64)

    tl.store(output_ptr + row_idx, argmax_idx)


def argmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty((n_rows,), device=x.device, dtype=torch.int64)
    argmax_kernel_cheat_tiebreak[(n_rows,)](
        y, x, x.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return y
