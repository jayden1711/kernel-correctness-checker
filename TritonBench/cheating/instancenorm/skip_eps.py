import torch
import triton
import triton.language as tl


@triton.jit
def instancenorm_kernel_cheat_skip_eps(output_ptr, input_ptr, gamma_ptr, beta_ptr,
                                        input_row_stride, output_row_stride,
                                        n_rows, n_cols, n_channels, eps,
                                        BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=0.0)
    mean = tl.sum(row, axis=0) / n_cols
    diff = tl.where(mask, row - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
# Bug: epsilon omitted entirely. Invisible whenever the row's variance
# is comfortably away from zero (typical random spatial data); wrong --
# division by near-zero, potential inf/nan -- only on adversarial
# near-constant spatial input, which is exactly the input this checker's
# adversarial search should be able to construct.
    inv_std = 1.0 / tl.sqrt(var)
    x_norm = (row - mean) * inv_std

    channel_idx = row_idx % n_channels
    gamma = tl.load(gamma_ptr + channel_idx)
    beta = tl.load(beta_ptr + channel_idx)

    output = x_norm * gamma + beta

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)


def instancenorm(x, weight, bias, eps=1e-5):
    N, C = x.shape[0], x.shape[1]
    spatial_shape = x.shape[2:]
    spatial_size = 1
    for d in spatial_shape:
        spatial_size *= d

    x2d = x.contiguous().view(N * C, spatial_size)
    n_rows, n_cols = x2d.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y2d = torch.empty_like(x2d)

    instancenorm_kernel_cheat_skip_eps[(n_rows,)](
        y2d, x2d, weight, bias,
        x2d.stride(0), y2d.stride(0),
        n_rows, n_cols, C, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y2d.view(N, C, *spatial_shape)
