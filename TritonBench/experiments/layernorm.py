"""
experiments/layernorm.py

Shows the progression:
  1. Naive allclose passes all three cheating kernels
  2. Algebraic properties (zero_mean, unit_variance) catch ignore_gamma_beta
     and skip_mean_subtract
  3. Adversarial inputs (large mean shift) catch wrong_variance_estimate
  4. Adaptive perturbation tolerance catches all three
"""

import torch
from TritonBench.reference.layernorm import layernorm as ref_layernorm
from TritonBench.cheating.layer_norm.ignore_gamma_beta import layernorm as cheat_ignore_gamma_beta
from TritonBench.cheating.layer_norm.skip_mean_subtract import layernorm as cheat_skip_mean
from TritonBench.cheating.layer_norm.wrong_variance_estimate import layernorm as cheat_wrong_var

from verification.layer3_properties.layernorm_properties import (
    check_zero_mean,
    check_unit_variance,
    check_scale_invariance,
)
from verification.layer2_numeric_oracle.perturbation import check_perturbation_tolerance


def run_layernorm():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_rows, n_cols = 512, 512
    x     = torch.randn(n_rows, n_cols, device=device)
    gamma = torch.ones(n_cols, device=device)
    beta  = torch.zeros(n_cols, device=device)

    # Adversarial: large mean shift exposes wrong_variance_estimate
    # E[x^2] - mean^2 diverges numerically when mean is large
    x_adv     = torch.randn(n_rows, n_cols, device=device) + 1000.0
    gamma_adv = torch.ones(n_cols, device=device)
    beta_adv  = torch.zeros(n_cols, device=device)

    ref     = ref_layernorm(x, gamma, beta)
    ref_adv = ref_layernorm(x_adv, gamma_adv, beta_adv)

    cheats = [
        ("ignore_gamma_beta", cheat_ignore_gamma_beta),
        ("skip_mean_subtract", cheat_skip_mean),
        ("wrong_variance",     cheat_wrong_var),
    ]

    # Experiment 1: Naive allclose on random inputs
    print("=" * 60)
    print("Experiment 1: Naive allclose on random inputs")
    print("=" * 60)
    for name, fn in cheats:
        out = fn(x, gamma, beta)
        passes = torch.allclose(ref, out, atol=1e-4, rtol=1e-2)
        err = (ref - out).abs().max().item()
        print(f"  {name:<22} passes={passes}  max_err={err:.6f}")

    # Experiment 2: Algebraic properties (identity affine: gamma=1, beta=0)
    print("\n" + "=" * 60)
    print("Experiment 2: Algebraic properties (gamma=1, beta=0)")
    print("=" * 60)
    ones  = torch.ones(n_cols, device=device)
    zeros = torch.zeros(n_cols, device=device)
    for name, fn in cheats:
        out = fn(x, ones, zeros)
        zero_mean,  d1 = check_zero_mean(out)
        unit_var,   d2 = check_unit_variance(out)
        scale_inv,  d3 = check_scale_invariance(lambda xi: fn(xi, ones, zeros), x)
        print(f"  {name:<22} zero_mean={zero_mean}  unit_var={unit_var}  scale_inv={scale_inv}")

    print("\n" + "=" * 60)
    print("Experiment 3: Adversarial input (x + 1000 mean shift)")
    print("=" * 60)
    for name, fn in cheats:
        out = fn(x_adv, gamma_adv, beta_adv)
        passes = torch.allclose(ref_adv, out, atol=1e-4, rtol=1e-2)
        err = (ref_adv - out).abs().max().item()
        print(f"  {name:<22} passes={passes}  max_err={err:.6f}")

    # Experiment 4: Adaptive perturbation tolerance
    print("\n" + "=" * 60)
    print("Experiment 4: Adaptive perturbation tolerance")
    print("=" * 60)
    for name, fn in cheats:
        passed, detail = check_perturbation_tolerance(
            lambda xi: fn(xi, gamma, beta),
            lambda xi: ref_layernorm(xi, gamma, beta),
            x,
        )
        print(f"  {name:<22} passed={passed}  {detail}")

    # Experiment 5: Affine parameter correctness
    print("\n" + "=" * 60)
    print("Experiment 5: Affine correctness (gamma=2, beta=3)")
    print("=" * 60)
    gamma2 = torch.full((n_cols,), 2.0, device=device)
    beta3  = torch.full((n_cols,), 3.0, device=device)
    ref2   = ref_layernorm(x, gamma2, beta3)
    for name, fn in cheats:
        out = fn(x, gamma2, beta3)
        passes = torch.allclose(ref2, out, atol=1e-4, rtol=1e-2)
        err = (ref2 - out).abs().max().item()
        print(f"  {name:<22} passes={passes}  max_err={err:.6f}")
