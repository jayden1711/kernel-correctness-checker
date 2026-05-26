"""
experiments/softmax.py

Shows the progression:
  1. Naive allclose passes cheating kernels
  2. Algebraic properties catch them
  3. Structural access pattern check catches first_tile
  4. Adversarial oracle catches wrong_reduction
"""

import torch
from TritonBench.reference.softmax import softmax as ref_softmax, softmax_kernel
from TritonBench.cheating.softmax.first_tile import (
    softmax as cheat_first_tile,
    softmax_kernel_cheat_first_tile,
)
from TritonBench.cheating.softmax.wrong_reduction import softmax as cheat_wrong_reduction

from verification.layer3_properties.softmax_properties import (
    check_rows_sum_to_one,
    check_shift_invariance,
    check_monotonicity,
)
from verification.layer1_structural.tile_coverage import check_all_tiles_visited
from verification.layer2_numeric_oracle.perturbation import check_perturbation_tolerance


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x       = torch.randn(512, 2048, device=device)
    x_small = torch.randn(4,   2048, device=device)
    x_adv   = torch.randn(512, 2048, device=device)
    x_adv[:, -1] = 5.0  # max value in last tile

    ref         = ref_softmax(x)
    wrong_fn    = lambda xi: cheat_wrong_reduction(xi, PARTIAL_SIZE=2040)

    cheats = [
        ("first_tile",      cheat_first_tile),
        ("wrong_reduction", wrong_fn),
    ]

    # Experiment 1: Naive allclose — do cheats pass?
    print("=" * 60)
    print("Experiment 1: Naive allclose on random inputs")
    print("=" * 60)
    for name, fn in cheats:
        out = fn(x)
        passes = torch.allclose(ref, out, atol=1e-4, rtol=1e-2)
        err = (ref - out).abs().max().item()
        print(f"  {name:<20} passes={passes}  max_err={err:.6f}")

    # Experiment 2: Algebraic properties
    print("\n" + "=" * 60)
    print("Experiment 2: Algebraic property checks")
    print("=" * 60)
    for name, fn in cheats:
        out = fn(x)
        rows_sum,  d1 = check_rows_sum_to_one(out)
        shift_inv, d2 = check_shift_invariance(fn, x)
        monotone,  d3 = check_monotonicity(fn, x)
        print(f"  {name:<20} rows_sum={rows_sum}  shift_inv={shift_inv}  monotone={monotone}")

    # Experiment 3: Structural tile-coverage check
    print("\n" + "=" * 60)
    print("Experiment 3: Tile-coverage check (triton-viz)")
    print("=" * 60)
    try:
        passed, fail_row, cols = check_all_tiles_visited(
            ref_softmax, softmax_kernel, x_small
        )
        print(f"  reference      all_tiles_visited={passed}")

        passed, fail_row, cols = check_all_tiles_visited(
            cheat_first_tile, softmax_kernel_cheat_first_tile, x_small
        )
        print(f"  first_tile     all_tiles_visited={passed}  "
              f"fail_row={fail_row}  cols_visited={cols}/2048")
    except Exception as e:
        print(f"  (triton-viz not available: {e})")

    # Experiment 4: Adversarial oracle — max value in last tile
    print("\n" + "=" * 60)
    print("Experiment 4: Adversarial oracle (max value in last tile)")
    print("=" * 60)
    ref_adv = ref_softmax(x_adv)
    for name, fn in cheats:
        out = fn(x_adv)
        passes = torch.allclose(ref_adv, out, atol=1e-4, rtol=1e-2)
        err = (ref_adv - out).abs().max().item()
        print(f"  {name:<20} passes={passes}  max_err={err:.6f}")

    # Experiment 5: Adaptive perturbation tolerance
    print("\n" + "=" * 60)
    print("Experiment 5: Adaptive perturbation tolerance")
    print("=" * 60)
    for name, fn in cheats:
        passed, detail = check_perturbation_tolerance(fn, ref_softmax, x)
        print(f"  {name:<20} passed={passed}  {detail}")


if __name__ == "__main__":
    run()