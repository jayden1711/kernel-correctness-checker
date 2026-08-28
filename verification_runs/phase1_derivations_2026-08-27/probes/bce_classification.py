"""Is bce_loss's sandwich pass real, or is a floor masking a broken linearisation?

Data-analysis only -- reads banked records from this round and from
adaptive_tol_theory_2026-08-25. No GPU, no new measurement.

Three questions, in the order that can falsify fastest:
  A. Is tol sitting on the 1e-6 absolute clamp? (the equal_attention_weights
     mechanism)
  B. Is the q95 SAMPLE on the fp32 quantisation floor? (the last_tile_dropped /
     skip_rescaling mechanism). NOTE the diagnostic is q95, not min -- for m=1
     the min sample lands on the floor by construction and GPU_NATIVE.md 3b
     already records that as benign.
  C. If neither: what does y = tol/(3 sigma L) actually measure at m=1?
"""
import json, math, statistics as st

NEW = [json.loads(l) for l in open(
    "verification_runs/phase1_derivations_2026-08-27/native_run/phase1_native.jsonl")]
OLD = [json.loads(l) for l in open(
    "verification_runs/adaptive_tol_theory_2026-08-25/generalization/data/gpu_native.jsonl")]
HALF_NORMAL_Q95 = 1.959964          # q95(|Z|)
LO_R = 0.6744898                    # q50(|Z|)
NS, ETA = 40, 0.05


def qlin(xs, q):
    s = sorted(xs); n = len(s); h = q * (n - 1)
    lo = math.floor(h); hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def shape_stats(rs):
    """q95/RMS of the scalar response, its skew, and the median defect."""
    qs, sk, df = [], [], []
    for r in rs:
        s = r["sens"]
        rms = math.sqrt(sum(v * v for v in s) / len(s))
        if rms <= 0:
            continue
        qs.append(qlin(s, 0.95) / rms)
        mu, sd = st.mean(s), st.pstdev(s)
        if sd > 0:
            sk.append(sum(((v - mu) / sd) ** 3 for v in s) / len(s))
        if r.get("defect_t01") is not None:
            df.append(r["defect_t01"])
    return (st.median(qs), st.median(sk) if sk else float("nan"),
            st.median(df) if df else float("nan"))


bce = [r for r in NEW if r["op"] == "bce_loss" and r.get("kind") == "primary"]

# --- A ----------------------------------------------------------------------
raws = [3.0 * qlin(r["sens"], 0.95) for r in bce]
print("A. floor clamp:  any invocation with raw 3*q95 < 1e-6?",
      any(x < 1e-6 for x in raws),
      " | min raw = %.4e = %.1fx the floor" % (min(raws), min(raws) / 1e-6))

# --- B ----------------------------------------------------------------------
r0 = bce[0]
ulp = min(r0["sens"]) / r0["s_over_ulp"]
print("B. fp32 floor:   q95 sample = %.1f ulp (floor cases were 2-3 ulp);"
      " defect varies 44-79%%, not the constant 900%% signature"
      % (qlin(r0["sens"], 0.95) / ulp))

# --- C ----------------------------------------------------------------------
print("\nC. at m=1, L = sqrt(E[s^2])/sigma and tol = 3 q95(s) are statistics of")
print("   the SAME scalar response, so y = q95(s)/sqrt(E[s^2]) -- a shape ratio.")
pts = []
for op in ("nll_loss", "kldiv_loss", "huber_loss", "mse_loss", "bce_loss"):
    q, sk, d = shape_stats([r for r in NEW if r["op"] == op and r.get("kind") == "primary"])
    pts.append((op, d, q, sk))
q, sk, d = shape_stats([r for r in OLD if r["op"] == "cross_entropy"
                        and r.get("kind") == "primary" and "sens" in r])
pts.append(("cross_entropy", d, q, sk))

print(f"   {'operator':16s} {'defect':>9s} {'q95/RMS':>9s} {'skew':>7s}")
for op, d, qq, s_ in sorted(pts, key=lambda t: t[1]):
    print(f"   {op:16s} {100*d:8.2f}% {qq:9.4f} {s_:7.3f}")
print(f"   {'exact linearity':16s} {'0.00%':>9s} {HALF_NORMAL_Q95:9.4f} {0.995:7.3f}")

xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
slope = sum((a-mx)*(b-my) for a, b in zip(xs, ys)) / sum((a-mx)**2 for a in xs)
inter = my - slope*mx
ss = sum((b-my)**2 for b in ys)
rs = sum((b-(inter+slope*a))**2 for a, b in zip(xs, ys))
HI = math.sqrt(2*math.log(2)) + math.sqrt(2*math.log(NS/ETA))
print(f"\n   fit: q95/RMS = {inter:.4f} {slope:+.4f}*defect,  R^2 = {1-rs/ss:.3f}")
print(f"   zero-defect intercept {inter:.4f} vs half-normal {HALF_NORMAL_Q95:.4f}"
      f"  ({100*(inter/HALF_NORMAL_Q95-1):+.1f}%)")
print(f"   m=1 window [{LO_R:.4f}, {HI:.4f}];  lower bound reached at defect"
      f" = {100*(LO_R-inter)/slope:.0f}%  (bce_loss is at 53%)")
