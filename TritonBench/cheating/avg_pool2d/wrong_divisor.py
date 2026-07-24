import torch
import triton
import triton.language as tl


@triton.jit
def avgpool2d_kernel_cheat_wrong_divisor(output_ptr, input_ptr,
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

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for kh in range(kernel_size):
        h_in = h_out_idx * stride - padding + kh
        h_valid = (h_in >= 0) & (h_in < H)
        for kw in range(kernel_size):
            w_in = w_out_idx * stride - padding + kw
            w_valid = (w_in >= 0) & (w_in < W)
            valid = h_valid & w_valid & out_mask
            in_offset = ((n_idx * C + c_idx) * H + h_in) * W + w_in
            val = tl.load(input_ptr + in_offset, mask=valid, other=0.0)
            acc += val
            count += tl.where(valid, 1.0, 0.0)

# Bug: divides by valid-element count instead of the full kernel window.
# Invisible whenever padding=0; wrong whenever padding>0.
    result = acc / count

    tl.store(output_ptr + offsets, result, mask=out_mask)


def avg_pool2d(x, kernel_size, stride=None, padding=0):
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
    avgpool2d_kernel_cheat_wrong_divisor[grid](
        y.view(-1), x.view(-1), N, C, H, W, H_out, W_out,
        stride, padding, kernel_size=kernel_size, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
