"""Confirm the N % BLOCK_N != 0 attribution.

Source inspection of TritonBench/reference/{flash_attention,
scaled_dot_product_attention}.py: K/V loads are masked to zero for padded
kv positions, but S is never masked, so every padded column contributes
exp(0 - m) to the softmax denominator. causal_flash_attention masks S with
q_idx >= kv_idx, which incidentally excludes all padded columns (padded
kv_idx > every valid q_idx), so it is immune.

This probe emulates the kernel-faithful function on CPU (BLOCK_N=32 padding
to the next multiple, S=0 on padded columns for flash/sdpa) and checks, at
the banked (100,32) GPU inputs:
  (a) out_max of the emulated function matches the banked GPU out_max to
      fp32 accuracy, while the mathematical reference does NOT;
  (b) the Gram-law prediction recomputed from the EMULATED function's
      Jacobian restores meas/pred ~ 1 -- attributing the 100x32 deviation
      entirely to the kernels computing a different function, not to the law.
"""

import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from attn_gram_cpu import jacobian_rows, sim_exact  # noqa: E402

D = os.path.join(HERE, "../data")
BLOCK_N = 32
OPS = ["flash_attention", "scaled_dot_product_attention",
       "causal_flash_attention"]


def kernel_faithful(op, Q, K, V):
    """What the shipped kernel computes: softmax over N_pad columns where
    padded columns carry S = 0 (flash/sdpa). Causal masks them out."""
    N, Dh = Q.shape
    N_pad = math.ceil(N / BLOCK_N) * BLOCK_N
    S = (Q @ K.T) / math.sqrt(Dh)
    pad = Q.new_zeros(N, N_pad - N)
    S_full = torch.cat([S, pad], dim=1)          # padded columns: S = 0
    if op == "causal_flash_attention":
        idx_q = torch.arange(N).unsqueeze(1)
        idx_k = torch.arange(N_pad).unsqueeze(0)
        S_full = S_full.masked_fill(idx_q < idx_k, float("-inf"))
    P = torch.softmax(S_full, dim=-1)
    V_full = torch.cat([V, V.new_zeros(N_pad - N, Dh)], dim=0)
    return P @ V_full


def math_ref(op, Q, K, V):
    S = (Q @ K.T) / math.sqrt(Q.shape[1])
    if op == "causal_flash_attention":
        N = Q.shape[0]
        S = S.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool), 1),
                          float("-inf"))
    return torch.softmax(S, dim=-1) @ V


def main():
    qkv = np.load(os.path.join(D, "attn_gram_qkv.npz"))
    meas = [json.loads(l) for l in open(os.path.join(D, "attn_gram_gpu.jsonl"))
            if '"delta_scale": 0.001' in l]
    torch.manual_seed(2)

    print("=== (a) which function did the GPU actually compute? (out_max, fp32) ===")
    for r in [m for m in meas if m["N"] == 100]:
        op, N, Dh, s = r["op"], r["N"], r["D"], r["seed"]
        Q = torch.from_numpy(qkv[f"{op}_{N}x{Dh}_s{s}_Q"])
        K = torch.from_numpy(qkv[f"{op}_{N}x{Dh}_s{s}_K"])
        V = torch.from_numpy(qkv[f"{op}_{N}x{Dh}_s{s}_V"])
        om_k = kernel_faithful(op, Q, K, V).abs().max().item()
        om_m = math_ref(op, Q, K, V).abs().max().item()
        print(f"{op:30s} s{s} gpu {r['out_max']:.6f}  kernel-faithful {om_k:.6f} "
              f"(rel {abs(om_k-r['out_max'])/r['out_max']:.2e})  "
              f"math {om_m:.6f} (rel {abs(om_m-r['out_max'])/r['out_max']:.2e})")

    print("\n=== (b) Gram law re-predicted from the kernel-faithful Jacobian ===")
    import attn_gram_cpu as agc
    for r in [m for m in meas if m["N"] == 100]:
        op, N, Dh, s = r["op"], r["N"], r["D"], r["seed"]
        Qn = qkv[f"{op}_{N}x{Dh}_s{s}_Q"]
        Kn = qkv[f"{op}_{N}x{Dh}_s{s}_K"]
        Vn = qkv[f"{op}_{N}x{Dh}_s{s}_V"]
        Qt = torch.from_numpy(Qn).double().requires_grad_(True)
        Kt = torch.from_numpy(Kn).double()
        Vt = torch.from_numpy(Vn).double()
        J = torch.autograd.functional.jacobian(
            lambda q: kernel_faithful(op, q, Kt, Vt), Qt).reshape(-1, Qt.numel())
        L = J.norm(dim=1).max().item()
        y_pred, y_sd = sim_exact(J.float(), L, nrep=600)
        y = r["tol"] / (3 * r["sigma"] * L)
        print(f"{op:30s} s{s} y {y:.4f} pred_kernel {y_pred:.4f}+-{y_sd:.4f} "
              f"z {(y-y_pred)/y_sd:+.2f}")


if __name__ == "__main__":
    main()
