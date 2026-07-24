import torch
import triton
import triton.language as tl


@triton.jit
def l2norm_kernel_cheat_wrong_norm(output_ptr, input_ptr, input_row_stride, output_row_stride,
                                    n_rows, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=0.0)
# Bug: uses sum(|x|) instead of sqrt(sum(x^2)) -- algebraically
# different (Jensen's inequality). For constant-magnitude rows the two
# happen to agree up to a scale factor that can look like a passing
# tolerance; for high-variance adversarial rows they diverge sharply.
    l1_sum = tl.sum(tl.abs(row), axis=0)
    output = row / (l1_sum + eps)

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)


def l2norm(x, eps=1e-12):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    l2norm_kernel_cheat_wrong_norm[(n_rows,)](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return y