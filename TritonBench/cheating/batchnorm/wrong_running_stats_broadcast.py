import torch
import triton
import triton.language as tl


@triton.jit
def batchnorm_kernel_cheat_wrong_broadcast(output_ptr, input_ptr, running_mean_ptr, running_var_ptr,
                                            weight_ptr, bias_ptr,
                                            n_elements, C, spatial_size, eps,
                                            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

# Bug: channel index computed as if channel were the FASTEST-varying
# dimension (offsets % C) instead of accounting for spatial_size.
# Identical to the correct formula whenever spatial_size == 1 (e.g.
# BatchNorm1d on a plain (N, C) tensor with no spatial dims) --
# invisible on that shape, wrong for any real spatial input (2D/3D).
    channel_idx = offsets % C

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(running_mean_ptr + channel_idx, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + channel_idx, mask=mask, other=1.0)
    gamma = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
    beta = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    y = (x - mean) * inv_std * gamma + beta

    tl.store(output_ptr + offsets, y, mask=mask)


def batchnorm(x, running_mean, running_var, weight, bias, eps=1e-5):
    N, C = x.shape[0], x.shape[1]
    spatial_shape = x.shape[2:]
    spatial_size = 1
    for d in spatial_shape:
        spatial_size *= d

    x_flat = x.contiguous().view(-1)
    n_elements = x_flat.numel()
    y_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    batchnorm_kernel_cheat_wrong_broadcast[grid](
        y_flat, x_flat, running_mean, running_var, weight, bias,
        n_elements, C, spatial_size, eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y_flat.view_as(x)
