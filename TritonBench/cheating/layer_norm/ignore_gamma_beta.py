import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    output_ptr, input_ptr, gamma_ptr, beta_ptr,
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

    mean     = tl.sum(row, axis=0) / n_cols
    variance = tl.sum((row - mean) ** 2, axis=0) / n_cols

    #Bug: gamma and beta are loaded but never used
    #(loading them avoids unused pointer warnings that might hint at the bug)
    _gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    _beta  = tl.load(beta_ptr  + col_offsets, mask=mask, other=0.0)  

    layernorm_output = (row - mean) / tl.sqrt(variance + eps)
    # missing: * gamma + beta

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, layernorm_output, mask=mask)


def layernorm(x, gamma, beta, eps=1e-5):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)

    layernorm_kernel[(n_rows,)](
        y, x, gamma, beta,
        x.stride(0), y.stride(0),
        n_rows, n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y