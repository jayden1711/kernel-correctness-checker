import torch
import triton
import triton.language as tl


@triton.jit
def groupnorm_kernel_cheat_ignore_affine(output_ptr, input_ptr, gamma_ptr, beta_ptr,
                                          input_row_stride, output_row_stride,
                                          n_rows, n_cols, eps,
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
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (row - mean) * inv_std

# Bug: gamma/beta loaded but never applied. INVISIBLE if the checker's
# materialized gamma/beta happen to be ones/zeros (the identity affine
# transform) -- same gotcha as the RMSNorm ignore_gamma mutant. Requires
# non-uniform gamma and nonzero beta in the adversarial input to catch.
    _gamma = tl.load(gamma_ptr + row_idx * n_cols + col_offsets, mask=mask, other=1.0)
    _beta = tl.load(beta_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)
    output = x_norm

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)


def groupnorm(x, num_groups, weight, bias, eps=1e-5):
    N, C = x.shape[0], x.shape[1]
    spatial_shape = x.shape[2:]
    spatial_size = 1
    for d in spatial_shape:
        spatial_size *= d
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    group_size = channels_per_group * spatial_size

    x2d = x.contiguous().view(N * num_groups, group_size)

    weight_g = weight.view(num_groups, channels_per_group)
    weight_g = weight_g.unsqueeze(-1).expand(num_groups, channels_per_group, spatial_size)
    weight_g = weight_g.reshape(num_groups, group_size)
    gamma2d = weight_g.unsqueeze(0).expand(N, num_groups, group_size)
    gamma2d = gamma2d.reshape(N * num_groups, group_size).contiguous()

    bias_g = bias.view(num_groups, channels_per_group)
    bias_g = bias_g.unsqueeze(-1).expand(num_groups, channels_per_group, spatial_size)
    bias_g = bias_g.reshape(num_groups, group_size)
    beta2d = bias_g.unsqueeze(0).expand(N, num_groups, group_size)
    beta2d = beta2d.reshape(N * num_groups, group_size).contiguous()

    n_rows, n_cols = x2d.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y2d = torch.empty_like(x2d)

    groupnorm_kernel_cheat_ignore_affine[(n_rows,)](
        y2d, x2d, gamma2d, beta2d,
        x2d.stride(0), y2d.stride(0),
        n_rows, n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y2d.view(N, C, *spatial_shape)
