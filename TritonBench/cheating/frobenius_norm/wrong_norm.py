import torch
import triton
import triton.language as tl


@triton.jit
def frobenius_sumsq_kernel_cheat_wrong_norm(input_ptr, partial_sum_ptr, n_elements,
                                             BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
# Bug: accumulates sum(|x|) instead of sum(x^2) -- an L1-style norm
# substituted for the true Frobenius (L2) norm. Same wrong_norm template
# as l2norm's mutant, same Jensen's-inequality gap: agrees on
# constant-magnitude input, diverges on high-variance adversarial input.
    block_sum = tl.sum(tl.abs(x), axis=0)
    tl.atomic_add(partial_sum_ptr, block_sum)


@triton.jit
def frobenius_normalize_kernel_cheat(output_ptr, input_ptr, norm_ptr, n_elements, eps,
                                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    norm = tl.load(norm_ptr)
    y = x / (norm + eps)

    tl.store(output_ptr + offsets, y, mask=mask)


def frobenius_norm(x, eps=1e-12):
    x_flat = x.contiguous().view(-1)
    n_elements = x_flat.numel()

    sumsq = torch.zeros(1, device=x.device, dtype=torch.float32)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    frobenius_sumsq_kernel_cheat_wrong_norm[grid](x_flat, sumsq, n_elements, BLOCK_SIZE=BLOCK_SIZE)

# Bug (compounding): no sqrt taken either -- the accumulated sum(|x|)
# is used directly as the "norm", not sqrt(sum(|x|)).
    norm = sumsq

    y_flat = torch.empty_like(x_flat)
    frobenius_normalize_kernel_cheat[grid](y_flat, x_flat, norm, n_elements, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y_flat.view_as(x)
