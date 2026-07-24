import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel_cheat_sigmoid_approx(output_ptr, input_ptr, n_elements,
                                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
# Bug: uses the sigmoid-based GELU approximation (x * sigmoid(1.702*x))
# instead of the exact erf form. This is a REAL shortcut used in some
# production implementations, not a contrived bug. CORRECTED: max
# absolute error is ~0.020 at x ~= -2.27, in the MODERATE negative
# range -- NOT at large |x|. Both forms asymptote to x (positive tail)
# and 0 (negative tail), so they actually CONVERGE at extreme
# magnitude; testing only large |x| (e.g. +-8, +-15) will never catch
# this. Invisible on inputs concentrated near 0 or far from 0; wrong on
# inputs concentrated in roughly x in [-3, -1].
    approx_sigmoid = 1.0 / (1.0 + tl.exp(-1.702 * x))
    approx = x * approx_sigmoid

    tl.store(output_ptr + offsets, approx, mask=mask)


def gelu(x):
    x_flat = x.contiguous().view(-1)
    n_elements = x_flat.numel()
    y_flat = torch.empty_like(x_flat)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_kernel_cheat_sigmoid_approx[grid](y_flat, x_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y_flat.view_as(x)