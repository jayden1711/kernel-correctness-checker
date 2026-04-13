import torch

def check_rows_sum_to_one(output, atol=1e-4):
    """Check that each row of softmax output sums to 1."""
    row_sums = output.sum(dim=1)
    expected = torch.ones_like(row_sums)
    return torch.allclose(row_sums, expected, atol=atol)

def check_shift_invariance(kernel_fn, x, atol=1e-4):
    """Check that softmax(x + c) == softmax(x) for per-row constant shift."""
    shift = torch.randn(x.shape[0], 1, device=x.device)
    x_shifted = x + shift
    original_output = kernel_fn(x)
    shifted_output = kernel_fn(x_shifted)
    return torch.allclose(original_output, shifted_output, atol=atol)