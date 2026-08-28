import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(output_ptr, input_ptr, gamma_ptr, beta_ptr,input_row_stride, output_row_stride,
                   n_rows, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    
    #One kernel instance per row
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    #Vector of column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    #Mask out padding elements when BLOCK_SIZE > n_cols
    mask = col_offsets < n_cols

    #Single HBM read for entire row
    row = tl.load(input_ptrs, mask=mask, other=0.0)

    #Numerically stable layernorm
    mean = tl.sum(row, axis=0) / n_cols
    # Padded lanes hold 0.0, so (0 - mean)^2 would add mean^2 per pad lane to
    # the variance sum whenever BLOCK_SIZE > n_cols (non-power-of-two widths).
    # Mask them out, same as instancenorm/groupnorm do
    # (NORM_ADJUDICATION 2026-08-27 §2; fixed 2026-08-28).
    diff = tl.where(mask, row - mean, 0.0)
    variance = tl.sum(diff * diff, axis=0) / n_cols
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)
    layernorm_output = (row - mean) / tl.sqrt(variance + eps) * gamma + beta

    #Single HBM write for entire row
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, layernorm_output, mask=mask)

def layernorm(x, gamma, beta, eps=1e-5):
    n_rows, n_cols = x.shape
    # Companion-length contract, enforced LOUDLY. The kernel loads gamma and
    # beta with mask = col_offsets < n_cols, so a companion shorter than
    # n_cols is read PAST its allocation -- silently, with the output
    # depending on whatever tensor happens to sit next to it (byte-level
    # proof: verification_runs/oob_adjudication_2026-08-28/). A loud error
    # here routes the invalid call into the reference_failure_kind machinery
    # instead of into garbage numbers.
    if gamma.numel() != n_cols or beta.numel() != n_cols:
        raise ValueError(
            f"layernorm: companion length mismatch -- gamma {gamma.numel()}, "
            f"beta {beta.numel()}, n_cols {n_cols}; a short companion would "
            f"be read out of bounds (oob_adjudication_2026-08-28)")
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)

    #Launch one kernel instance per row
    layernorm_kernel[(n_rows,)](
        y, x, gamma, beta, x.stride(0), y.stride(0), n_rows, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return y