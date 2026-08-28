"""Attention Gram extension, GPU stage — out-of-sample tolerance measurement
on the REAL Triton kernels, at shapes the banked corpus never measured.

Runs on a Colab T4 (same stack as every prior round). For each
(op, shape, seed): draw Q/K/V from numpy default_rng (device-independent, so
the CPU side computes exact-Jacobian predictions for the SAME inputs), run
the shipped Triton kernel, and measure adaptive_tol with the standard NS=40
protocol, plus a delta_scale ladder {1e-4, 1e-3, 1e-2} for the
scale-invariance falsification (y must not move).

Emits /content/attn_gram_gpu.jsonl and /content/attn_gram_qkv.npz (the
banked Q/K/V), with flush+fsync and resume, per house convention.
"""

import json
import math
import os

import numpy as np
import torch

OUT = "/content/attn_gram_gpu.jsonl"
NS = 40
T_LADDER = [0.01, 0.1, 1.0]
DELTA_SCALES = [1e-4, 1e-3, 1e-2]

assert torch.cuda.is_available(), "no CUDA"
import triton  # noqa: E402
print("torch", torch.__version__, "| triton", triton.__version__,
      "|", torch.cuda.get_device_name(0), flush=True)

import sys
sys.path.insert(0, "/content")
try:
    from TritonBench.reference.flash_attention import flash_attention
    from TritonBench.reference.causal_flash_attention import causal_flash_attention
    from TritonBench.reference.scaled_dot_product_attention import scaled_dot_product_attention
except ImportError:
    # flat upload: the three reference files sit directly in /content
    from flash_attention import flash_attention
    from causal_flash_attention import causal_flash_attention
    from scaled_dot_product_attention import scaled_dot_product_attention

KERNELS = {
    "flash_attention": flash_attention,
    "causal_flash_attention": causal_flash_attention,
    "scaled_dot_product_attention": scaled_dot_product_attention,
}
OP_IDX = {op: i for i, op in enumerate(sorted(KERNELS))}

# (N, D): corpus shape as the continuity anchor, then three out-of-sample
# shapes -- bigger N and D, a non-multiple-of-32 N (exercises the masks),
# and a small-D case at the tl.dot minimum.
SHAPES = [(64, 32), (128, 64), (100, 32), (256, 16)]
SEEDS = [0, 1, 2]


def draw_qkv(op, N, D, seed):
    rng = np.random.default_rng([seed, N, D, OP_IDX[op]])
    Q = rng.normal(size=(N, D)).astype(np.float32)
    K = rng.normal(size=(N, D)).astype(np.float32)
    V = rng.normal(size=(N, D)).astype(np.float32)
    return Q, K, V


def qlin(xs, q):
    s = sorted(xs)
    n = len(s)
    h = q * (n - 1)
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def measure(fn, Q, K, V, delta_scale, gen_seed):
    base = fn(Q, K, V)
    sigma = delta_scale * Q.float().std().item()
    g = torch.Generator(device=Q.device).manual_seed(gen_seed)
    sens = []
    for _ in range(NS):
        d = torch.randn(Q.shape, generator=g, device=Q.device,
                        dtype=Q.dtype) * sigma
        sens.append((fn(Q + d, K, V) - base).abs().max().item())
    tol = 3.0 * qlin(sens, 0.95)
    # linearisation ladder along one fresh direction
    g2 = torch.Generator(device=Q.device).manual_seed(gen_seed + 500000)
    d = torch.randn(Q.shape, generator=g2, device=Q.device,
                    dtype=Q.dtype) * sigma
    ladder = {t: (fn(Q + t * d, K, V) - base).abs().max().item()
              for t in T_LADDER}
    s1 = ladder[1.0]
    defect = abs(s1 - ladder[0.1] / 0.1) / s1 if s1 > 0 else None
    return dict(sigma=sigma, tol=tol, sens=sens, defect_t01=defect,
                out_max=base.abs().max().item())


done = set()
if os.path.exists(OUT):
    for ln in open(OUT):
        try:
            j = json.loads(ln)
            done.add((j["op"], j["N"], j["D"], j["seed"], j["delta_scale"]))
        except Exception:
            pass
print("resuming, already done:", len(done), flush=True)
fh = open(OUT, "a")

qkv_bank = {}
for op, fn in KERNELS.items():
    for (N, D) in SHAPES:
        for seed in SEEDS:
            Qn, Kn, Vn = draw_qkv(op, N, D, seed)
            qkv_bank[f"{op}_{N}x{D}_s{seed}_Q"] = Qn
            qkv_bank[f"{op}_{N}x{D}_s{seed}_K"] = Kn
            qkv_bank[f"{op}_{N}x{D}_s{seed}_V"] = Vn
            Q = torch.from_numpy(Qn).cuda()
            K = torch.from_numpy(Kn).cuda()
            V = torch.from_numpy(Vn).cuda()
            for ds in DELTA_SCALES:
                key = (op, N, D, seed, ds)
                if key in done:
                    continue
                r = measure(fn, Q, K, V, ds,
                            gen_seed=1000 + 7919 * seed + 31 * OP_IDX[op] + N + D)
                rec = dict(op=op, N=N, D=D, seed=seed, delta_scale=ds, **r)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                print(f"{op:30s} {N:4d}x{D:<3d} s{seed} ds={ds:g} "
                      f"tol {r['tol']:.4e} defect {r['defect_t01']}",
                      flush=True)

np.savez_compressed("/content/attn_gram_qkv.npz", **qkv_bank)
print("DONE", flush=True)
