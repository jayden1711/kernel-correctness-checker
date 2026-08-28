"""Does the padded-column bug contaminate the banked `last_tile_dropped`
measurements (N=65, the taxonomy's fp32-floor case)?

The suppression argument -- K[-1]=V[-1]=1e4 makes every row's max score huge,
so padded exp(0-m) vanishes -- is only half right: rows whose Q_i . K_last is
NEGATIVE have O(1) row max, and their padded mass is not suppressed. This
probe measures, on the variant's own construction (fp32, 20 seeds):

  1. output difference buggy-vs-true (max rel, and per-row split by the sign
     of the dominant score);
  2. the statistics the taxonomy actually banked: the perturbation response
     s = ||f(x+sigma d) - f(x)||_inf over 40 matched deltas, its min/ulp and
     q95, under the buggy and the true function -- does the fp32-floor
     classification (s at 2-3 ulp) survive correction?
"""

import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(
    HERE, "../../adaptive_tol_theory_2026-08-25/attention_gram/probes"))
from attn_padded_confirm import kernel_faithful, math_ref  # noqa: E402

N, D = 65, 32          # attn_native ran the corpus D=32; spec default is D=64
NS = 40
DELTA_SCALE = 1e-3
OP = "flash_attention"


def variant_inputs(seed):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(N, D, generator=g)
    K = torch.randn(N, D, generator=g)
    V = torch.randn(N, D, generator=g)
    K[-1, :] = 1e4
    V[-1, :] = 1e4
    return Q, K, V


def sens(fn, Q, K, V, seed):
    base = fn(OP, Q, K, V)
    sigma = DELTA_SCALE * Q.std().item()
    g = torch.Generator().manual_seed(10_000 + seed)
    out = []
    for _ in range(NS):
        d = torch.randn(Q.shape, generator=g) * sigma
        out.append((fn(OP, Q + d, K, V) - base).abs().max().item())
    ulp = torch.finfo(torch.float32).eps * base.abs().max().item()
    return np.array(out), ulp, base


rows = []
for seed in range(20):
    Q, K, V = variant_inputs(seed)
    s_bug, ulp_b, out_b = sens(kernel_faithful, Q, K, V, seed)
    s_true, ulp_t, out_t = sens(math_ref, Q, K, V, seed)
    out_rel = ((out_b - out_t).abs().max() / out_t.abs().max()).item()
    # per-row: which rows have a suppressed (positive-dominant) max?
    S = (Q @ K.T) / math.sqrt(D)
    neg_dom = (S[:, -1] < 0).float().mean().item()
    q95b = np.quantile(s_bug, 0.95)
    q95t = np.quantile(s_true, 0.95)
    rows.append((out_rel, neg_dom, s_bug.min() / ulp_b, s_true.min() / ulp_t,
                 q95b, q95t))
    if seed < 6:
        print(f"seed {seed}: out rel diff {out_rel:.3e}  frac rows neg-dominant "
              f"{neg_dom:.2f}  s_min/ulp bug {s_bug.min()/ulp_b:.1f} true "
              f"{s_true.min()/ulp_t:.1f}  q95 bug {q95b:.3e} true {q95t:.3e} "
              f"(ratio {q95b/q95t:.3f})")

r = np.array(rows)
print(f"\n20 seeds: out rel diff max {r[:,0].max():.3e}")
print(f"s_min/ulp: buggy median {np.median(r[:,2]):.1f}  true median {np.median(r[:,3]):.1f}")
print(f"q95 ratio buggy/true: median {np.median(r[:,4]/r[:,5]):.4f} "
      f"range [{(r[:,4]/r[:,5]).min():.4f}, {(r[:,4]/r[:,5]).max():.4f}]")
