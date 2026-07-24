import torch
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel_cheat_wrong_padding(output_ptr, input_ptr,
                                          N, C, H, W, H_out, W_out,
                                          stride, padding,
                                          kernel_size: tl.constexpr,
                                          BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * H_out * W_out
    out_mask = offsets < total

    w_out_idx = offsets % W_out
    tmp = offsets // W_out
    h_out_idx = tmp % H_out
    tmp2 = tmp // H_out
    c_idx = tmp2 % C
    n_idx = tmp2 // C

# Bug: padded positions load 0.0 instead of -inf. Invisible whenever
# the true window max is positive; wrong whenever it's negative.
    max_val = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    for kh in range(kernel_size):
        h_in = h_out_idx * stride - padding + kh
        h_valid = (h_in >= 0) & (h_in < H)
        for kw in range(kernel_size):
            w_in = w_out_idx * stride - padding + kw
            w_valid = (w_in >= 0) & (w_in < W)
            valid = h_valid & w_valid & out_mask
            in_offset = ((n_idx * C + c_idx) * H + h_in) * W + w_in
            val = tl.load(input_ptr + in_offset, mask=valid, other=0.0)
            max_val = tl.maximum(max_val, val)

    tl.store(output_ptr + offsets, max_val, mask=out_mask)


def max_pool2d(x, kernel_size, stride=None, padding=0):
    if stride is None:
        stride = kernel_size
    N, C, H, W = x.shape
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    x = x.contiguous()
    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    total = N * C * H_out * W_out
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    maxpool2d_kernel_cheat_wrong_padding[grid](
        y.view(-1), x.view(-1), N, C, H, W, H_out, W_out,
        stride, padding, kernel_size=kernel_size, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
