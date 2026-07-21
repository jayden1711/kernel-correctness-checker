import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    output_ptr, input_ptr, gamma_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=0.0)

    # Bug: uses mean(|x|) instead of sqrt(mean(x^2))
    # These are algebraically different (Jensen's inequality).
    # For constant input they happen to agree, so gamma=1 + constant_rows
    # won't catch it. Adversarial inputs with large_variance will.
    abs_row = tl.abs(row)
    norm = tl.sum(abs_row, axis=0) / n_cols + eps

    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)

    output = row / norm * gamma

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)


def rmsnorm(x, gamma, eps=1e-5):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)

    rmsnorm_kernel[(n_rows,)](
        y, x, gamma,
        x.stride(0), y.stride(0),
        n_rows, n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y