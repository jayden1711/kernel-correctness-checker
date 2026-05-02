import torch

def make_adversarial_input(n_rows, n_cols, BLOCK_SIZE, device='cuda'):
    x = torch.randn(n_rows, n_cols, device=device) * 0.1  # small random baseline
    # place the true max in the last tile, only slightly larger than the rest
    x[:, -1] = x.max(dim=1).values + 0.5  # just 0.5 above current max
    return x

def check_adversarial(ref_fn, cheat_fn, n_rows, n_cols, BLOCK_SIZE, atol=1e-4, rtol=1e-2):
    x = make_adversarial_input(n_rows, n_cols, BLOCK_SIZE)
    ref = ref_fn(x)
    cheat = cheat_fn(x)
    passes = torch.allclose(ref, cheat, atol=atol, rtol=rtol)
    err = torch.max(torch.abs(ref - cheat))
    return passes, err