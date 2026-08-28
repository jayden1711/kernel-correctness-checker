"""
B.3 second pass: is the per-invocation scatter of the measured prefix ratio
explained by the parent's own single-draw sampling noise?

For each of the 228 invocations, the structural parent (same exact CDF as
b3_chain.py) is sampled to get the DISTRIBUTION of the joint statistic
q95_20(prefix)/q95_40(full 40) -- mean AND sd -- and the banked measured
ratio is scored as z = (meas - mean)/sd. If the chain is complete, the 228
z-scores are ~N(0,1); systematic residue shows up as |mean z| >> 1/sqrt(228)
or sd(z) >> 1.

Also reports the same for n = 5 (deep prefix, larger noise).
"""

import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "autokernel", "files"))

from tritonbench_registry import OPS, FAMILIES
from verification.layer2_numeric_oracle.structural_l import row_norms

NATIVE = os.path.join(HERE, "..", "..", "adaptive_tol_theory_2026-08-25",
                      "native_run", "gpu_native.jsonl")
DATA = os.path.join(HERE, "..", "data")
NREP = 8000


def qlin_np(a, n):
    sub = np.sort(a[:, :n], axis=1)
    h = 0.95 * (n - 1)
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return sub[:, lo] + (h - lo) * (sub[:, hi] - sub[:, lo])


def qlin(xs, q=0.95):
    s = sorted(xs)
    n = len(s)
    h = q * (n - 1)
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def parent_from_profile(w):
    w = w[w > 0]
    w = w / w.max()
    m = w.size
    t_hi = math.sqrt(2 * math.log(max(2 * m, 4))) + 6.0
    t = np.linspace(0.0, t_hi, 4096)
    tt = torch.from_numpy(t)
    logF = torch.zeros_like(tt)
    for i0 in range(0, m, 2048):
        wi = torch.from_numpy(w[i0:i0 + 2048]).unsqueeze(1)
        c = torch.erf(tt.unsqueeze(0) / wi / math.sqrt(2.0))
        logF += torch.log(c.clamp_min(1e-300)).sum(dim=0)
    return t, torch.exp(logF).numpy()


def replay_inputs():
    rng = np.random.default_rng(0)
    out = {}
    entry = 0
    for spec_key, ref_file, cheat_dir, family, mutant_names in OPS:
        mk_fn = FAMILIES[family][0]
        for _mut in mutant_names:
            for j in range(6):
                out[(entry, j)] = (spec_key, mk_fn(rng))
            entry += 1
    return out


def main():
    inputs = replay_inputs()
    rows = [json.loads(l) for l in open(NATIVE)]
    rows = [r for r in rows if r.get("kind") == "primary" and r.get("sens")]
    rng = np.random.default_rng(31415)
    out = []
    for r in rows:
        key = (int(r["entry"]), int(r["inv"]))
        op, np_args = inputs[key]
        ts = [torch.from_numpy(a).to(torch.float64) if isinstance(a, np.ndarray)
              and a.dtype != np.int64 else
              (torch.from_numpy(a) if isinstance(a, np.ndarray) else a)
              for a in np_args]
        rn = row_norms(op, ts[0], tuple(ts[1:]))
        w = rn.detach().cpu().numpy().astype(np.float64)
        t, F = parent_from_profile(w)
        u = rng.random((NREP, 40))
        s = np.interp(u, F, t)
        rec = {"op": op}
        for n in (5, 20):
            ratio = qlin_np(s, n) / qlin_np(s, 40)
            mu, sd = float(ratio.mean()), float(ratio.std(ddof=1))
            meas = qlin(r["sens"][:n]) / qlin(r["sens"])
            rec[f"z{n}"] = (meas - mu) / sd
            rec[f"sd{n}"] = sd
        out.append(rec)
    for n in (5, 20):
        zs = np.array([o[f"z{n}"] for o in out])
        sds = np.array([o[f"sd{n}"] for o in out])
        print(f"n={n}: mean z = {zs.mean():+.3f} (expected sem "
              f"{1 / math.sqrt(len(zs)):.3f}), sd(z) = {zs.std(ddof=1):.3f}, "
              f"worst |z| = {abs(zs).max():.2f}, "
              f"median predicted sd = {np.median(sds) * 100:.2f}%")
        # worst ops
        import collections
        byop = collections.defaultdict(list)
        for o in out:
            byop[o["op"]].append(o[f"z{n}"])
        wo = sorted(((abs(np.mean(v)), o) for o, v in byop.items()),
                    reverse=True)[:4]
        print("   op-mean |z| worst:",
              ", ".join(f"{o} {np.mean(byop[o]):+.2f}" for _a, o in wo))
    json.dump(out, open(os.path.join(DATA, "b3_zscores.json"), "w"))


if __name__ == "__main__":
    main()
