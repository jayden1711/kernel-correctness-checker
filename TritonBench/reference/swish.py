import torch
import triton
import triton.language as tl


@triton.jit
def swish_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid_x

    tl.store(output_ptr + offsets, y, mask=mask)


def swish(x):
    x_flat = x.contiguous().view(-1)
    n_elements = x_flat.numel()
    y_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    swish_kernel[grid](y_flat, x_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y_flat.view_as(x)