import torch
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(output_ptr, input_ptr,
                      N, C, L, L_out,
                      stride, padding,
                      kernel_size: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * L_out
    out_mask = offsets < total

    l_out_idx = offsets % L_out
    nc_idx = offsets // L_out
    c_idx = nc_idx % C
    n_idx = nc_idx // C

    max_val = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    for k in range(kernel_size):
        l_in = l_out_idx * stride - padding + k
        valid = (l_in >= 0) & (l_in < L) & out_mask
        in_offset = (n_idx * C + c_idx) * L + l_in
        val = tl.load(input_ptr + in_offset, mask=valid, other=-float('inf'))
        max_val = tl.maximum(max_val, val)

    tl.store(output_ptr + offsets, max_val, mask=out_mask)


def max_pool1d(x, kernel_size, stride=None, padding=0):
    """x: (N, C, L). Scalar (not per-dim tuple) kernel_size/stride/padding
    only. Assumes floor-mode output length (no ceil_mode support)."""
    if stride is None:
        stride = kernel_size
    N, C, L = x.shape
    L_out = (L + 2 * padding - kernel_size) // stride + 1

    x = x.contiguous()
    y = torch.empty((N, C, L_out), device=x.device, dtype=x.dtype)

    total = N * C * L_out
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    maxpool1d_kernel[grid](
        y.view(-1), x.view(-1), N, C, L, L_out,
        stride, padding, kernel_size=kernel_size, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
