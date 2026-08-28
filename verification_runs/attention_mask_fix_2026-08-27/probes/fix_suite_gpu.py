"""GPU verification suite for the flash/sdpa padded-column masking fix.

Runs on a Colab T4 with the FIXED kcc tarball extracted to /content and
flash_attention_buggy.py (the pre-fix kernel, snapshotted from git HEAD)
uploaded alongside. Four stages, one JSONL each:

  A  correctness gate: fixed kernels vs torch math at 9 shapes including all
     previously-affected ones, plus the weights-sum invariant; causal as the
     untouched control.
  B  the predicted x1.2 delta: last_tile_dropped tolerance measured with the
     buggy and the fixed kernel on IDENTICAL inputs and IDENTICAL deltas
     (same generator seed), 10 seeds; plus post-fix fp-floor statistics for
     the taxonomy re-verification.
  C  the three July N=130 proposals replayed against the fixed reference and
     the four (untouched) mutant kernels: reference validity, per-mutant
     caught/naive -- expected classification: valid non-hit, no gap.
  E  post-fix 100x32 measurement, same draw scheme and generator seeds as
     attention_gram's attn_gram_gpu.py, for comparison against the banked
     math-Jacobian predictions (expected meas/pred ~ 1.00, was ~0.89).
"""

import json
import math
import os
import sys

import numpy as np
import torch

assert torch.cuda.is_available()
sys.path.insert(0, "/content")
sys.path.insert(0, "/content/benchmarks/autokernel/files")

from TritonBench.reference.flash_attention import flash_attention
from TritonBench.reference.scaled_dot_product_attention import scaled_dot_product_attention
from TritonBench.reference.causal_flash_attention import causal_flash_attention
from flash_attention_buggy import flash_attention_buggy
from TritonBench.cheating.flash_attention.approx_denom import flash_attention as mut_approx
from TritonBench.cheating.flash_attention.drop_last_tile import flash_attention as mut_drop
from TritonBench.cheating.flash_attention.skip_rescaling import flash_attention as mut_skip
from TritonBench.cheating.flash_attention.wrong_mask import flash_attention as mut_wrongmask

OUT = "/content/fix_suite.jsonl"
fh = open(OUT, "w")


def emit(rec):
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def torch_ref(op, Q, K, V):
    S = (Q @ K.T) / math.sqrt(Q.shape[1])
    if op == "causal":
        N = Q.shape[0]
        S = S.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool,
                                                device=S.device), 1),
                          float("-inf"))
    return torch.softmax(S, dim=-1) @ V


def qlin(xs, q):
    s = sorted(xs)
    n = len(s)
    h = q * (n - 1)
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


# ---------------------------------------------------------------- stage A
print("=== A: correctness gate ===", flush=True)
SHAPES = [(1, 64), (64, 32), (65, 64), (100, 32), (128, 64), (130, 64),
          (192, 64), (256, 16), (333, 64)]
worst = {}
for opname, fn, mathop in [("flash_attention", flash_attention, "flash"),
                           ("scaled_dot_product_attention",
                            scaled_dot_product_attention, "flash"),
                           ("causal_flash_attention", causal_flash_attention,
                            "causal")]:
    for (N, D) in SHAPES:
        rng = np.random.default_rng([N, D, 99])
        Q = torch.from_numpy(rng.normal(size=(N, D)).astype(np.float32)).cuda()
        K = torch.from_numpy(rng.normal(size=(N, D)).astype(np.float32)).cuda()
        V = torch.from_numpy(rng.normal(size=(N, D)).astype(np.float32)).cuda()
        out = fn(Q, K, V)
        ref = torch_ref(mathop, Q, K, V)
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        ws = fn(Q, K, torch.ones_like(V))
        wdev = (ws - torch.ones_like(ws)).abs().max().item()
        worst[opname] = max(worst.get(opname, 0.0), rel)
        emit(dict(stage="A", op=opname, N=N, D=D, rel_err=rel, wsum_dev=wdev))
        print(f"A {opname:30s} ({N:3d},{D:2d}) rel {rel:.2e} wsum_dev {wdev:.2e}",
              flush=True)
print("A worst rel err per op:", worst, flush=True)

# ---------------------------------------------------------------- stage B
print("=== B: last_tile_dropped tol, buggy vs fixed, identical draws ===",
      flush=True)
NS = 40
N, D = 65, 32          # attn_native's construction (corpus D)
for seed in range(10):
    g = torch.Generator(device="cuda").manual_seed(7000 + seed)
    Q = torch.randn(N, D, generator=g, device="cuda")
    K = torch.randn(N, D, generator=g, device="cuda")
    V = torch.randn(N, D, generator=g, device="cuda")
    K[-1, :] = 1e4
    V[-1, :] = 1e4
    sigma = 1e-3 * Q.float().std().item()
    rec = dict(stage="B", seed=seed, sigma=sigma)
    for tag, fn in [("buggy", flash_attention_buggy), ("fixed", flash_attention)]:
        base = fn(Q, K, V)
        g2 = torch.Generator(device="cuda").manual_seed(8000 + seed)  # SAME deltas
        sens = []
        for _ in range(NS):
            d = torch.randn(Q.shape, generator=g2, device="cuda") * sigma
            sens.append((fn(Q + d, K, V) - base).abs().max().item())
        ulp = torch.finfo(torch.float32).eps * base.abs().max().item()
        rec[tag] = dict(tol=3 * qlin(sens, 0.95), s_min_ulp=min(sens) / ulp,
                        s_med_ulp=float(np.median(sens)) / ulp)
    rec["ratio_fixed_over_buggy"] = rec["fixed"]["tol"] / rec["buggy"]["tol"]
    emit(rec)
    print(f"B seed {seed}: tol buggy {rec['buggy']['tol']:.3e} fixed "
          f"{rec['fixed']['tol']:.3e} ratio {rec['ratio_fixed_over_buggy']:.3f} "
          f"| fixed s_med/ulp {rec['fixed']['s_med_ulp']:.1f}", flush=True)

# ---------------------------------------------------------------- stage C
print("=== C: July N=130 proposals vs fixed reference ===", flush=True)
NP_, DP = 130, 64
PROPS = {
    1: dict(Q=("randn", 1.0, None), K=("randn", 1.0, 1e4), V=("randn", 1.0, 1e4)),
    6: dict(Q=("randn", 0.1, None), K=("zeros", 1.0, 10.0), V=("zeros", 1.0, 5.0)),
    9: dict(Q=("ones", 0.01, None), K=("zeros", 1.0, 0.5), V=("zeros", 1.0, 1.0)),
}
MUTS = dict(approx_denom=mut_approx, drop_last_tile=mut_drop,
            skip_rescaling=mut_skip, wrong_mask=mut_wrongmask)
for idx, spec in PROPS.items():
    g = torch.Generator(device="cuda").manual_seed(4242 + idx)

    def build(kind):
        fill, scale, patch = spec[kind]
        if fill == "randn":
            t = torch.randn(NP_, DP, generator=g, device="cuda") * scale
        elif fill == "zeros":
            t = torch.zeros(NP_, DP, device="cuda")
        else:
            t = torch.ones(NP_, DP, device="cuda") * scale
        if patch is not None:
            t[128:, :] = patch
        return t
    Q, K, V = build("Q"), build("K"), build("V")
    ref = flash_attention(Q, K, V)
    ws = flash_attention(Q, K, torch.ones_like(V))
    ref_wsum_dev = (ws - torch.ones_like(ws)).abs().max().item()
    rec = dict(stage="C", idx=idx, ref_wsum_dev=ref_wsum_dev,
               ref_valid=ref_wsum_dev <= 1e-3, muts={})
    for name, fn in MUTS.items():
        mo = fn(Q, K, V)
        finite = bool(torch.isfinite(mo).all())
        mws = fn(Q, K, torch.ones_like(V))
        caught_ws = (not torch.isfinite(mws).all()) or \
            (mws - torch.ones_like(mws)).abs().max().item() > 1e-3
        naive = finite and bool(torch.allclose(mo.float(), ref.float(),
                                               atol=1e-3, rtol=1e-2))
        rec["muts"][name] = dict(caught_wsum=bool(caught_ws), naive_pass=naive)
    gap = any(m["caught_wsum"] and m["naive_pass"] for m in rec["muts"].values())
    rec["classification"] = ("HIT" if rec["ref_valid"] and gap else
                             "valid non-hit" if rec["ref_valid"] else
                             "reference failed")
    emit(rec)
    print(f"C idx {idx}: ref wsum_dev {ref_wsum_dev:.2e} valid={rec['ref_valid']} "
          f"-> {rec['classification']} | " +
          " ".join(f"{k}:c={v['caught_wsum']},n={v['naive_pass']}"
                   for k, v in rec["muts"].items()), flush=True)

# ---------------------------------------------------------------- stage E
print("=== E: post-fix 100x32, attention_gram protocol ===", flush=True)
KERNELS = {"flash_attention": flash_attention,
           "causal_flash_attention": causal_flash_attention,
           "scaled_dot_product_attention": scaled_dot_product_attention}
OP_IDX = {op: i for i, op in enumerate(sorted(KERNELS))}
N, D = 100, 32
for op, fn in KERNELS.items():
    for seed in [0, 1, 2]:
        rng = np.random.default_rng([seed, N, D, OP_IDX[op]])
        Q = torch.from_numpy(rng.normal(size=(N, D)).astype(np.float32)).cuda()
        K = torch.from_numpy(rng.normal(size=(N, D)).astype(np.float32)).cuda()
        V = torch.from_numpy(rng.normal(size=(N, D)).astype(np.float32)).cuda()
        base = fn(Q, K, V)
        sigma = 1e-3 * Q.float().std().item()
        g = torch.Generator(device="cuda").manual_seed(
            1000 + 7919 * seed + 31 * OP_IDX[op] + N + D)
        sens = []
        for _ in range(40):
            d = torch.randn(Q.shape, generator=g, device="cuda") * sigma
            sens.append((fn(Q + d, K, V) - base).abs().max().item())
        tol = 3 * qlin(sens, 0.95)
        emit(dict(stage="E", op=op, N=N, D=D, seed=seed, sigma=sigma, tol=tol))
        print(f"E {op:30s} s{seed} tol {tol:.4e}", flush=True)

fh.close()
print("DONE", flush=True)
