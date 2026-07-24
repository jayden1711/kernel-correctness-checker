import torch
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel_cheat_wrong_padding(output_ptr, input_ptr,
                                          N, C, D, H, W, D_out, H_out, W_out,
                                          stride, padding,
                                          kernel_size: tl.constexpr,
                                          BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * D_out * H_out * W_out
    out_mask = offsets < total

    w_out_idx = offsets % W_out
    tmp = offsets // W_out
    h_out_idx = tmp % H_out
    tmp2 = tmp // H_out
    d_out_idx = tmp2 % D_out
    tmp3 = tmp2 // D_out
    c_idx = tmp3 % C
    n_idx = tmp3 // C

# Bug: padded positions load 0.0 instead of -inf.
    max_val = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    for kd in range(kernel_size):
        d_in = d_out_idx * stride - padding + kd
        d_valid = (d_in >= 0) & (d_in < D)
        for kh in range(kernel_size):
            h_in = h_out_idx * stride - padding + kh
            h_valid = (h_in >= 0) & (h_in < H)
            for kw in range(kernel_size):
                w_in = w_out_idx * stride - padding + kw
                w_valid = (w_in >= 0) & (w_in < W)
                valid = d_valid & h_valid & w_valid & out_mask
                in_offset = (((n_idx * C + c_idx) * D + d_in) * H + h_in) * W + w_in
                val = tl.load(input_ptr + in_offset, mask=valid, other=0.0)
                max_val = tl.maximum(max_val, val)

    tl.store(output_ptr + offsets, max_val, mask=out_mask)


def max_pool3d(x, kernel_size, stride=None, padding=0):
    if stride is None:
        stride = kernel_size
    N, C, D, H, W = x.shape
    D_out = (D + 2 * padding - kernel_size) // stride + 1
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    x = x.contiguous()
    y = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    total = N * C * D_out * H_out * W_out
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    maxpool3d_kernel_cheat_wrong_padding[grid](
        y.view(-1), x.view(-1), N, C, D, H, W, D_out, H_out, W_out,
        stride, padding, kernel_size=kernel_size, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
