"""
experiments/flash_attention.py

Shows the progression:
  1. Naive allclose passes cheating kernels on benign random inputs
  2. Adversarial inputs (high scores in last tile) catch drop_last_tile
     and skip_rescaling
  3. Multi-tile rescaling compounding exposes skip_rescaling more clearly
  4. Output validity (no NaN/Inf) and bounded-by-values property
  5. Attention weights sum-to-one test catches approx_denom and skip_rescaling
  6. Adaptive perturbation tolerance catches all
"""

import torch
from TritonBench.reference.flash_attention import flash_attention as ref_flash_attention
from TritonBench.cheating.flash_attention.approx_denom import flash_attention as cheat_approx_denom
from TritonBench.cheating.flash_attention.drop_last_tile import flash_attention as cheat_drop_last
from TritonBench.cheating.flash_attention.skip_rescaling import flash_attention as cheat_skip_rescale
from TritonBench.cheating.flash_attention.wrong_mask import flash_attention as cheat_wrong_mask

from verification.layer3_properties.flash_attention_properties import (
    check_output_bounded_by_values,
    check_attention_weights_sum_to_one,
)
from verification.layer2_numeric_oracle.perturbation import check_perturbation_tolerance


def run_flash_attention():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    N, D = 128, 64
    Q = torch.randn(N, D, device=device)
    K = torch.randn(N, D, device=device)
    V = torch.randn(N, D, device=device)

    # Adversarial: high scores in last tile
    Q_adv = torch.randn(N, D, device=device)
    K_adv = torch.randn(N, D, device=device)
    K_adv[-32:] *= 10.0  # last tile keys have large magnitude
    V_adv = torch.randn(N, D, device=device)

    # Multi-tile: 6 iterations, max score shifts dramatically
    BLOCK = 32
    N_long = BLOCK * 6  # 192
    Q_long = torch.randn(N_long, D, device=device)
    K_long = torch.randn(N_long, D, device=device)
    K_long[:BLOCK]        *= 1e-6
    K_long[BLOCK:BLOCK*2] *= 1.0
    K_long[BLOCK*2:]      *= 1e4
    V_long = torch.randn(N_long, D, device=device)

    ref     = ref_flash_attention(Q, K, V)
    ref_adv = ref_flash_attention(Q_adv, K_adv, V_adv)
    ref_long = ref_flash_attention(Q_long, K_long, V_long)

    cheats = [
        ("approx_denom",   cheat_approx_denom),
        ("drop_last_tile", cheat_drop_last),
        ("skip_rescaling", cheat_skip_rescale),
        ("wrong_mask",     cheat_wrong_mask),
    ]

    # Experiment 1: Naive allclose on random inputs
    print("=" * 60)
    print("Experiment 1: Naive allclose on random inputs")
    print("=" * 60)
    for name, fn in cheats:
        out = fn(Q, K, V)
        passes = torch.allclose(ref, out, atol=1e-4, rtol=1e-2)
        err = (ref - out).abs().max().item()
        print(f"  {name:<18} passes={passes}  max_err={err:.6f}")

    print("\n" + "=" * 60)
    print("Experiment 2: Adversarial (high scores in last tile)")
    print("=" * 60)
    for name, fn in cheats:
        out = fn(Q_adv, K_adv, V_adv)
        passes = torch.allclose(ref_adv, out, atol=1e-4, rtol=1e-2)
        err = (ref_adv - out).abs().max().item()
        print(f"  {name:<18} passes={passes}  max_err={err:.6f}")

    # Experiment 3: Multi-tile rescaling (6 tiles, shifting max score)
    print("\n" + "=" * 60)
    print("Experiment 3: Multi-tile rescaling (N=192, 6 tiles)")
    print("=" * 60)
    for name, fn in cheats:
        try:
            out = fn(Q_long, K_long, V_long)
            passes = torch.allclose(ref_long, out, atol=1e-4, rtol=1e-2)
            err = (ref_long - out).abs().max().item()
            print(f"  {name:<18} passes={passes}  max_err={err:.6f}")
        except Exception as e:
            print(f"  {name:<18} exception: {e}")

    # Experiment 4: Output validity and bounded-by-values
    print("\n" + "=" * 60)
    print("Experiment 4: Output validity (finite) and bounded by V range")
    print("=" * 60)
    for name, fn in cheats:
        out = fn(Q, K, V)
        is_finite = torch.isfinite(out).all().item()
        bounded, detail = check_output_bounded_by_values(out, V)
        print(f"  {name:<18} finite={is_finite}  bounded_by_V={bounded}")

    # Experiment 5: Attention weights sum to one (V=ones test)
    print("\n" + "=" * 60)
    print("Experiment 5: Attention weights sum to one (V=ones)")
    print("=" * 60)
    for name, fn in cheats:
        passed, detail = check_attention_weights_sum_to_one(fn, Q, K, V)
        print(f"  {name:<18} weights_sum_to_one={passed}  {detail}")

    # Experiment 6: Adaptive perturbation tolerance
    print("\n" + "=" * 60)
    print("Experiment 6: Adaptive perturbation tolerance")
    print("=" * 60)
    for name, fn in cheats:
        passed, detail = check_perturbation_tolerance(
            lambda x: fn(Q, K, V),
            lambda x: ref_flash_attention(Q, K, V),
            Q,
        )
        print(f"  {name:<18} passed={passed}  {detail}")
