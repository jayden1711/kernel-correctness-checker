"""
Design probe for the near-miss mutant family (item 7).

Mechanism: each near-miss mutant is the REFERENCE kernel with its output
multiplied by (1 + delta) inside the kernel. Against the reference, its
perturbation-check error is exactly delta * max|f(x)|, so the margin is

    margin = delta * M / tol,   M = max|f(x)|,  tol = 3 * P95_k ||f(x+d_k)-f(x)||_inf

Both M and tol are functionals of the same input draw, so the margin is a
RATIO of two stable statistics; this probe measures rho = tol / M per
operator over seeds (CPU fp32, the checker's exact delta discipline:
20 deltas, d = 1e-3 * std(x) * randn, P95, scale 3) and derives the delta
that lands each target margin, plus the seed-to-seed margin CV the GPU
validation should expect.

Run:  .venv/bin/python design_deltas.py
"""
import json
import math
import os

import torch

torch.manual_seed(7)
N, D = 64, 128
TARGETS = [0.5, 0.8, 1.0, 1.25, 2.0]


def ln(x):
    m = x.mean(-1, keepdim=True)
    v = ((x - m) ** 2).mean(-1, keepdim=True)
    return (x - m) / torch.sqrt(v + 1e-5)


def softmax(x):
    m = x.max(-1, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(-1, keepdim=True)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


def l2norm(x):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-12)


def sumred(x):
    return x.sum(-1)


OPS = {"layernorm": ln, "softmax": softmax, "gelu": gelu,
       "l2norm": l2norm, "sum_reduction": sumred}


def main():
    out = {}
    print(f"{'op':14s} {'rho=tol/M (median)':>20} {'CV%':>6} "
          f"{'deltas for margins ' + str(TARGETS)}")
    for name, f in OPS.items():
        rhos = []
        for s in range(20):
            torch.manual_seed(s)
            x = torch.randn(N, D)
            base = f(x)
            M = float(base.abs().max())
            ss = []
            for _ in range(20):
                d = torch.randn_like(x) * 1e-3 * float(x.std())
                ss.append(float((f(x + d) - base).abs().max()))
            tol = max(3.0 * torch.quantile(torch.tensor(ss), 0.95).item(), 1e-6)
            rhos.append(tol / M)
        rhos_t = torch.tensor(rhos)
        med = float(rhos_t.median())
        cv = float(rhos_t.std() / rhos_t.mean()) * 100
        deltas = {m: m * med for m in TARGETS}
        out[name] = {"rho_median": med, "cv_pct": cv, "deltas": deltas}
        print(f"{name:14s} {med:20.3e} {cv:6.1f} "
              + "  ".join(f"{m}:{d:.3e}" for m, d in deltas.items()))
    path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "design_deltas.json")
    json.dump(out, open(path, "w"), indent=1)
    print("\nwritten:", os.path.normpath(path))


if __name__ == "__main__":
    main()
