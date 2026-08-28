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

    # RMS = sqrt(mean(x^2) + eps)
    sq = row * row
    mean_sq = tl.sum(sq, axis=0) / n_cols
    rms = tl.sqrt(mean_sq + eps)

    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)

    output = row / rms * gamma

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)


def rmsnorm(x, gamma, eps=1e-5):
    n_rows, n_cols = x.shape
    # Companion-length contract, enforced LOUDLY -- same rationale and same
    # byte-level proof as layernorm's check
    # (verification_runs/oob_adjudication_2026-08-28/): the kernel masks its
    # gamma load at n_cols, so a shorter gamma is read past its allocation
    # and the output absorbs adjacent memory.
    if gamma.numel() != n_cols:
        raise ValueError(
            f"rmsnorm: companion length mismatch -- gamma {gamma.numel()}, "
            f"n_cols {n_cols}; a short gamma would be read out of bounds "
            f"(oob_adjudication_2026-08-28)")
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)

    rmsnorm_kernel[(n_rows,)](
        y, x, gamma,
        x.stride(0), y.stride(0),
        n_rows, n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y