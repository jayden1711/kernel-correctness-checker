import torch
from TritonBench.reference.flash_attention import flash_attention as ref_flash_attention
from TritonBench.kernels.flash_attention.cheat_skip_rescale import flash_attention as cheat_skip_rescale
from TritonBench.kernels.flash_attention.cheat_drop_last_tile import flash_attention as cheat_drop_last_tile
from TritonBench.kernels.flash_attention.cheat_approx_denominator import flash_attention as cheat_approx_denominator
from TritonBench.kernels.flash_attention.cheat_wrong_mask import flash_attention as cheat_wrong_mask
from checks.properties import check_rows_sum_to_one, check_shift_invariance
from checks.structural import check_all_tiles_visited
from checks.oracle import check_adversarial


def run_flash_attention():
    N, D = 128, 64
    Q = torch.randn(N, D, device='cuda')
    K = torch.randn(N, D, device='cuda')
    V = torch.randn(N, D, device='cuda')

    #Adversarial: high scores concentrated in last tile
    Q_adv = torch.randn(N, D, device='cuda')
    K_adv = torch.randn(N, D, device='cuda')
    K_adv[-32:] *= 10.0  # last tile keys have large magnitude — forces high scores late
    V_adv = torch.randn(N, D, device='cuda')

    ref = ref_flash_attention(Q, K, V)
    ref_adv = ref_flash_attention(Q_adv, K_adv, V_adv)

    cheats = [
        ("skip_rescale",       cheat_skip_rescale),
        ("drop_last_tile",     cheat_drop_last_tile),
        ("approx_denominator", cheat_approx_denominator),
        ("wrong_mask",         cheat_wrong_mask),
    ]

    #Experiment 1: allclose on random inputs
    print("Experiment 1: allclose on random inputs")
    for name, fn in cheats:
        out = fn(Q, K, V)
        passes = torch.allclose(ref, out, atol=1e-4, rtol=1e-2)
        err = torch.max(torch.abs(ref - out)).item()
        print(f"  {name}: passes={passes}, max_err={err:.6f}")

    #Experiment 2: allclose on adversarial inputs
    print("\nExperiment 2: allclose on adversarial inputs (high scores in last tile)")
    for name, fn in cheats:
        out = fn(Q_adv, K_adv, V_adv)
        passes = torch.allclose(ref_adv, out, atol=1e-4, rtol=1e-2)
        err = torch.max(torch.abs(ref_adv - out)).item()
        print(f"  {name}: passes={passes}, max_err={err:.6f}")

    #Experiment 3: algebraic property — output rows should not sum to 1
    #(flash attention output is weighted sum of V, not a probability distribution)
    #instead check output is finite and non-nan
    print("\nExperiment 3: output validity (no nan/inf)")
    for name, fn in cheats:
        out = fn(Q, K, V)
        valid = torch.isfinite(out).all().item()
        print(f"  {name}: all_finite={valid}")

    #Experiment 4: equivalence to naive attention on random inputs
    print("\nExperiment 4: max absolute error vs reference")
    for name, fn in cheats:
        out = fn(Q, K, V)
        err = torch.max(torch.abs(ref - out)).item()
        mean_err = torch.mean(torch.abs(ref - out)).item()
        print(f"  {name}: max_err={err:.6f}, mean_err={mean_err:.6f}")


if __name__ == "__main__":
    run_flash_attention()