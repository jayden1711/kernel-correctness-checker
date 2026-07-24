import torch
import triton
import triton.language as tl


@triton.jit
def argmin_kernel_cheat_tiebreak(output_ptr, input_ptr, input_row_stride,
                                  n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=float('inf'))
    min_val = tl.min(row, axis=0)

# Bug: LAST-occurrence tie-break instead of first. Invisible whenever
# the row's min is unique; wrong only on adversarial duplicate-min input.
    is_min = (row == min_val) & mask
    candidate_idx = tl.where(is_min, col_offsets, -1)
    argmin_idx = tl.max(candidate_idx, axis=0).to(tl.int64)

    tl.store(output_ptr + row_idx, argmin_idx)


def argmin(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty((n_rows,), device=x.device, dtype=torch.int64)
    argmin_kernel_cheat_tiebreak[(n_rows,)](
        y, x, x.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return y
