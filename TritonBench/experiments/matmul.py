"""
experiments/matmul.py

Shows the progression:
  1. Naive allclose passes several cheating kernels on benign inputs
  2. K-dim accumulation test catches partial_k_reduct immediately
  3. Non-aligned shape catches skip_boundary_tiles
  4. Rectangular input catches swapped_strides
  5. Precision / dtype check catches wrong_dtype
  6. Adaptive perturbation tolerance catches all
"""

import torch
from TritonBench.reference.mat_mult import matmul as ref_matmul
from TritonBench.cheating.matmult.partial_k_reduct import matmul as cheat_partial_k
from TritonBench.cheating.matmult.skip_boundary_tiles import matmul as cheat_skip_boundary
from TritonBench.cheating.matmult.swapped_strides import matmul as cheat_swapped
from TritonBench.cheating.matmult.wrong_dtype import matmul as cheat_wrong_dtype

from verification.layer2_numeric_oracle.perturbation import check_perturbation_tolerance
from verification.layer1_structural.runtime_guards import check_dtype_preserved


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Standard square inputs
    M, K, N = 256, 256, 256
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    ref = ref_matmul(A, B)

    cheats = [
        ("partial_k_reduct",  cheat_partial_k),
        ("skip_boundary",     cheat_skip_boundary),
        ("swapped_strides",   cheat_swapped),
        ("wrong_dtype",       cheat_wrong_dtype),
    ]

    # Experiment 1: Naive allclose on square random inputs
    print("=" * 60)
    print("Experiment 1: Naive allclose on square random inputs")
    print("=" * 60)
    for name, fn in cheats:
        try:
            out = fn(A, B)
            passes = torch.allclose(ref, out.float(), atol=1e-3, rtol=1e-2)
            err = (ref - out.float()).abs().max().item()
            print(f"  {name:<20} passes={passes}  max_err={err:.6f}")
        except Exception as e:
            print(f"  {name:<20} exception: {e}")

    # Experiment 2: K-dim accumulation (A=ones, B=ones, output should be K)
    print("\n" + "=" * 60)
    print("Experiment 2: K-dim accumulation (A=ones, B=ones -> output=K)")
    print("=" * 60)
    K_large = 256
    A_ones = torch.ones(M, K_large, device=device)
    B_ones = torch.ones(K_large, N, device=device)
    ref_ones = ref_matmul(A_ones, B_ones)
    for name, fn in cheats:
        try:
            out = fn(A_ones, B_ones)
            passes = torch.allclose(ref_ones, out.float(), atol=1e-2, rtol=1e-2)
            err = (ref_ones - out.float()).abs().max().item()
            print(f"  {name:<20} passes={passes}  max_err={err:.2f}  "
                  f"(expected {float(K_large):.0f}, got ~{out.float().mean().item():.1f})")
        except Exception as e:
            print(f"  {name:<20} exception: {e}")

    # Experiment 3: Non-aligned shape (exposes skip_boundary_tiles)
    print("\n" + "=" * 60)
    print("Experiment 3: Non-aligned shape (33x33 — not multiple of BLOCK=32)")
    print("=" * 60)
    A33 = torch.randn(33, 33, device=device)
    B33 = torch.randn(33, 33, device=device)
    ref33 = ref_matmul(A33, B33)
    for name, fn in cheats:
        try:
            out = fn(A33, B33)
            passes = torch.allclose(ref33, out.float(), atol=1e-3, rtol=1e-2)
            err = (ref33 - out.float()).abs().max().item()
            print(f"  {name:<20} passes={passes}  max_err={err:.6f}")
        except Exception as e:
            print(f"  {name:<20} exception: {e}")

    # Experiment 4: Rectangular input (exposes swapped_strides)
    print("\n" + "=" * 60)
    print("Experiment 4: Rectangular input (64x128 x 128x32) — all strides differ")
    print("=" * 60)
    A_rect = torch.randn(64, 128, device=device)
    B_rect = torch.randn(128, 32, device=device)
    ref_rect = ref_matmul(A_rect, B_rect)
    for name, fn in cheats:
        try:
            out = fn(A_rect, B_rect)
            passes = torch.allclose(ref_rect, out.float(), atol=1e-3, rtol=1e-2)
            err = (ref_rect - out.float()).abs().max().item()
            print(f"  {name:<20} passes={passes}  max_err={err:.6f}")
        except Exception as e:
            print(f"  {name:<20} exception: {e}")

    # Experiment 5: Dtype check (exposes wrong_dtype)
    print("\n" + "=" * 60)
    print("Experiment 5: Output dtype preservation")
    print("=" * 60)
    for name, fn in cheats:
        try:
            out = fn(A, B)
            dtype_ok = out.dtype == A.dtype
            print(f"  {name:<20} input_dtype={A.dtype}  output_dtype={out.dtype}  "
                  f"preserved={dtype_ok}")
        except Exception as e:
            print(f"  {name:<20} exception: {e}")

    # Experiment 6: Adaptive perturbation tolerance
    print("\n" + "=" * 60)
    print("Experiment 6: Adaptive perturbation tolerance")
    print("=" * 60)
    for name, fn in cheats:
        try:
            passed, detail = check_perturbation_tolerance(
                lambda x: fn(A, B),
                lambda x: ref_matmul(A, B),
                A,
            )
            print(f"  {name:<20} passed={passed}  {detail}")
        except Exception as e:
            print(f"  {name:<20} exception: {e}")


if __name__ == "__main__":
    run()