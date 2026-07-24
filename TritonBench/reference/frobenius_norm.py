import torch
import triton
import triton.language as tl


@triton.jit
def frobenius_sumsq_kernel(input_ptr, partial_sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Each program computes a local sum-of-squares over its block and
    atomically accumulates into a single global scalar. This is a
    DIFFERENT pattern from every other kernel in this corpus -- every
    other operator reduces within one row/instance per program; this
    one reduces across the WHOLE tensor, which needs cross-block
    coordination (here, atomic_add) since no single program sees the
    whole input. Give this file extra scrutiny before trusting it as
    ground truth -- it's the least like anything already validated."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(x * x, axis=0)
    tl.atomic_add(partial_sum_ptr, block_sum)


@triton.jit
def frobenius_normalize_kernel(output_ptr, input_ptr, norm_ptr, n_elements, eps,
                                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    norm = tl.load(norm_ptr)
    y = x / (norm + eps)

    tl.store(output_ptr + offsets, y, mask=mask)


def frobenius_norm(x, eps=1e-12):
    """x / ||x||_F, where ||x||_F = sqrt(sum of squares of ALL elements),
    not a per-row reduction. Two kernel launches: (1) accumulate the
    global sum-of-squares via atomics, (2) elementwise-divide by its
    sqrt. The sqrt itself happens on a single scalar on the host between
    launches -- not a reduction, so not a correctness risk on its own."""
    x_flat = x.contiguous().view(-1)
    n_elements = x_flat.numel()

    sumsq = torch.zeros(1, device=x.device, dtype=torch.float32)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    frobenius_sumsq_kernel[grid](x_flat, sumsq, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    norm = torch.sqrt(sumsq)

    y_flat = torch.empty_like(x_flat)
    frobenius_normalize_kernel[grid](y_flat, x_flat, norm, n_elements, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y_flat.view_as(x)
