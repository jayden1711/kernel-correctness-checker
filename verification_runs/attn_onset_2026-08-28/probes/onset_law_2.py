"""
Follow-ups to onset_law.py:

F1  MECHANISM TEST for the residual per-record structure (T4 found
    corr(median log r, g_min) = +0.55..+0.75 in the saturated regime,
    falsifying strict g-independence of the medians). Hypothesis: the
    residual comes from CROSS-ROW MAX COMPETITION -- the inf-norms take a
    max over rows, and |1-e^{-dg}| amplifies favorable dg draws in the
    numerator more than |dg| does in the denominator. Falsifiable
    prediction: restricting BOTH norms to the single dominant row must
    (a) kill the g_min correlation and (b) shrink the T1 residual.
    If it doesn't, the mechanism is something else and we say so.

F2  CAUSAL kappa sweep -- the banked floor-flag at kappa=20 was a CAUSAL
    record (1/5), and causal rows see fewer effective keys, so its floor
    onset should sit earlier than sdpa's. Anchor: P(floor | kappa=20)
    should be consistent with the banked 1-of-5 (and 0-of-5 for sdpa).

Run:  .venv/bin/python onset_law_2.py
"""
import json
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from verification.layer2_numeric_oracle import math_refs

torch.manual_seed(20260829)

N, D = 64, 32
DELTA_SCALE = 1e-3


def phi(u):
    return 1.0 if u == 0 else (1.0 - math.exp(-u)) / u


def f32(op, q_, k_, v_):
    s = (q_ @ k_.T) / math.sqrt(q_.shape[1])
    if op == "causal_flash_attention":
        s = s.masked_fill(torch.triu(torch.ones(s.shape, dtype=torch.bool),
                                     diagonal=1), float("-inf"))
    return torch.softmax(s, dim=-1) @ v_


def ulp_at(m):
    if m == 0 or not math.isfinite(m):
        return float("nan")
    return 2.0 ** (math.floor(math.log2(abs(m))) - 23)


def record(op, kappa, n_deltas, want_single_row=False):
    Q0, K0, V = torch.randn(N, D), torch.randn(N, D), torch.randn(N, D)
    x, kk = (kappa * Q0).float(), (kappa * K0).float()
    fn = math_refs.get(op)
    f = lambda t: fn(t, kk.double(), V.double())
    base = f32(op, x, kk, V)
    x_std = float(x.std())
    u_out = ulp_at(float(base.abs().max()))

    L = (x.double() @ kk.double().T) / math.sqrt(D)
    if op == "causal_flash_attention":
        L = L.masked_fill(torch.triu(torch.ones(L.shape, dtype=torch.bool),
                                     diagonal=1), float("-inf"))
    top2 = torch.topk(L, k=2, dim=-1)
    gaps = top2.values[:, 0] - top2.values[:, 1]
    finite = torch.isfinite(gaps)
    g_min = float(gaps[finite].min()) if finite.any() else float("nan")

    rs, rs_single, s_all = [], [], []
    for _ in range(n_deltas):
        d = torch.randn(N, D) * DELTA_SCALE * x_std
        diff = f32(op, x + d, kk, V) - base
        s_meas = float(diff.abs().max())
        s_all.append(s_meas)
        _, jd = torch.func.jvp(f, (x.double(),), (d.double(),))
        s_lin = float(jd.abs().max())
        if s_meas <= 0 or s_lin <= 0:
            continue
        rs.append(s_meas / s_lin)
        if want_single_row:
            i_star = int(jd.abs().max(dim=-1).values.argmax())
            s1 = float(diff[i_star].abs().max())
            l1 = float(jd[i_star].abs().max())
            if s1 > 0 and l1 > 0:
                rs_single.append(s1 / l1)
    med = sorted(rs)[len(rs) // 2] if len(rs) >= 5 else None
    med1 = (sorted(rs_single)[len(rs_single) // 2]
            if len(rs_single) >= 5 else None)
    s_all.sort()
    sulp = s_all[len(s_all) // 2] / u_out if u_out == u_out else float("nan")
    return {"median_r": med, "median_r_single": med1, "g_min": g_min,
            "sulp": sulp}


def corr(pairs):
    n = len(pairs)
    if n < 3:
        return float("nan")
    ma = sum(a for a, _ in pairs) / n
    mb = sum(b for _, b in pairs) / n
    cov = sum((a - ma) * (b - mb) for a, b in pairs)
    sa = math.sqrt(sum((a - ma) ** 2 for a, _ in pairs))
    sb = math.sqrt(sum((b - mb) ** 2 for _, b in pairs))
    return cov / (sa * sb) if sa > 0 and sb > 0 else float("nan")


def main():
    out = {}
    print("F1 mechanism test: full-output max vs single-dominant-row ratio")
    for op in ("scaled_dot_product_attention", "causal_flash_attention"):
        recs = [record(op, 20.0, 20, want_single_row=True) for _ in range(150)]
        sat = [r for r in recs if r["median_r"] and r["median_r_single"]
               and r["g_min"] > 5]
        pf = [(math.log10(r["median_r"]), r["g_min"]) for r in sat]
        ps = [(math.log10(r["median_r_single"]), r["g_min"]) for r in sat]
        # medians above/below 1, full vs single-row
        up_full = sum(1 for r in sat if r["median_r"] > 1) / max(1, len(sat))
        up_sing = sum(1 for r in sat
                      if r["median_r_single"] > 1) / max(1, len(sat))
        print(f"  {op[:30]:30s} n(sat)={len(sat)}  "
              f"corr(med,g_min): full={corr(pf):+.3f} single={corr(ps):+.3f}  "
              f"P(med>1): full={up_full:.2f} single={up_sing:.2f}")
        out[f"F1_{op}"] = {"n": len(sat), "corr_full": corr(pf),
                           "corr_single": corr(ps), "up_full": up_full,
                           "up_single": up_sing}

    print("\nF2 causal vs sdpa floor onset (n=200 per point):")
    print(f"  {'kappa':>6} | {'sdpa floor%':>12} {'causal floor%':>14}")
    for kappa in (15, 20, 25, 30, 40, 50):
        row = {}
        for op in ("scaled_dot_product_attention", "causal_flash_attention"):
            recs = [record(op, float(kappa), 10) for _ in range(200)]
            sulps = [r["sulp"] for r in recs if r["sulp"] == r["sulp"]]
            row[op] = 100 * sum(1 for s in sulps if s < 32) / len(sulps)
        print(f"  {kappa:6d} | {row['scaled_dot_product_attention']:11.1f}% "
              f"{row['causal_flash_attention']:13.1f}%")
        out[f"F2_k{kappa}"] = row
    print("  (banked at kappa=20: causal 1/5 floor-flagged, sdpa 0/5)")

    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "data",
                                     "onset_law_2.json"), "w"), indent=1)
    print("\ndone; data/onset_law_2.json written")


if __name__ == "__main__":
    main()
