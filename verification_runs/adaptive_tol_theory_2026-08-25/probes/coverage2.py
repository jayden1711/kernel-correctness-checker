"""
Full 29-operator coverage run, on the CORPUS's own inputs (replayed exactly
from np.random.default_rng(0), same draws the banked T4 run used).

Per operator:
  1. REPLAY   adaptive_tol on CPU vs the banked Triton-on-T4 value, same input.
  2. C1       linearisation: || f(x+d)-f(x) ||_inf  vs  || J d ||_inf (jvp).
  3. SANDWICH 2.023 sigma L <= tol <= 3 sigma L (sqrt(2 ln 2m) + sqrt(2 ln(n/eta))).
"""
import gzip, json, math, statistics as st
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp

DT = torch.float32
NS, K, ETA = 40, 400, 0.05
DELTA_SCALE = 1e-3
SQRT2INV = 0.7071067811865476

exec(open(__file__.rsplit('/', 1)[0] + '/_inputs.py').read())   # FAMILIES, OPS


def R(op):
    if op == "argmax":      return lambda x: x.argmax(dim=-1)
    if op == "argmin":      return lambda x: x.argmin(dim=-1)
    if op == "softmax":     return lambda x: torch.softmax(x, -1)
    if op == "log_softmax": return lambda x: torch.log_softmax(x, -1)
    if op == "gelu":        return lambda x: x * 0.5 * (1.0 + torch.erf(x * SQRT2INV))
    if op == "swish":       return lambda x: x * torch.sigmoid(x)
    if op == "l1norm":      return lambda x: x / (x.abs().sum(-1, keepdim=True) + 1e-12)
    if op == "l2norm":      return lambda x: x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-12)
    if op == "frobenius_norm": return lambda x: x / (torch.sqrt((x * x).sum()) + 1e-12)
    if op == "sum_reduction":  return lambda x: x.sum(-1)
    if op == "mean_reduction": return lambda x: x.mean(-1)
    if op == "max_reduction":  return lambda x: x.max(-1).values
    if op == "min_reduction":  return lambda x: x.min(-1).values
    if op == "matmul":      return lambda x, B: x @ B
    if op == "rmsnorm":
        return lambda x, g: x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * g
    if op == "layernorm":
        def f(x, g, b):
            m = x.mean(-1, keepdim=True)
            v = ((x - m) ** 2).mean(-1, keepdim=True)
            return (x - m) * torch.rsqrt(v + 1e-5) * g + b
        return f
    if op == "instancenorm":
        def f(x, w, b):
            N, C = x.shape[0], x.shape[1]; sp = x.shape[2:]
            x2 = x.contiguous().view(N * C, -1)
            m = x2.mean(-1, keepdim=True); v = ((x2 - m) ** 2).mean(-1, keepdim=True)
            ch = torch.arange(N * C) % C
            y = (x2 - m) * torch.rsqrt(v + 1e-5) * w[ch].unsqueeze(-1) + b[ch].unsqueeze(-1)
            return y.view(N, C, *sp)
        return f
    if op == "groupnorm":
        def f(x, ng, w, b):
            N, C = x.shape[0], x.shape[1]; sp = x.shape[2:]
            ssz = 1
            for dd in sp: ssz *= dd
            cpg = C // ng; gsz = cpg * ssz
            x2 = x.contiguous().view(N * ng, gsz)
            m = x2.mean(-1, keepdim=True); v = ((x2 - m) ** 2).mean(-1, keepdim=True)
            def ex(t):
                t = t.view(ng, cpg).unsqueeze(-1).expand(ng, cpg, ssz).reshape(ng, gsz)
                return t.unsqueeze(0).expand(N, ng, gsz).reshape(N * ng, gsz)
            y = (x2 - m) * torch.rsqrt(v + 1e-5) * ex(w) + ex(b)
            return y.view(N, C, *sp)
        return f
    if op == "batchnorm":
        def f(x, rm, rv, w, b):
            sh = (1, -1) + (1,) * (x.dim() - 2)
            return (x - rm.view(sh)) * torch.rsqrt(rv.view(sh) + 1e-5) * w.view(sh) + b.view(sh)
        return f
    if op == "cross_entropy":
        # host wrapper returns per_sample_loss.mean() -- a SCALAR (m = 1)
        return lambda x, t: (-torch.log_softmax(x, -1)
                             .gather(1, t.unsqueeze(1)).squeeze(1)).mean()
    if op in ("flash_attention", "scaled_dot_product_attention"):
        def f(Q, Kk, V):
            S = Q @ Kk.transpose(-2, -1) * (1.0 / math.sqrt(Q.shape[-1]))
            return torch.softmax(S, -1) @ V
        return f
    if op == "causal_flash_attention":
        def f(Q, Kk, V):
            N = Q.shape[0]
            S = Q @ Kk.transpose(-2, -1) * (1.0 / math.sqrt(Q.shape[-1]))
            i = torch.arange(N).unsqueeze(1); j = torch.arange(N).unsqueeze(0)
            return torch.softmax(S.masked_fill(j > i, float("-inf")), -1) @ V
        return f
    if op.startswith("avg_pool"):
        fn = {"1d": F.avg_pool1d, "2d": F.avg_pool2d, "3d": F.avg_pool3d}[op[-2:]]
        return lambda x, k, s, p: fn(x, k, s, p, count_include_pad=True)
    if op.startswith("max_pool"):
        fn = {"1d": F.max_pool1d, "2d": F.max_pool2d, "3d": F.max_pool3d}[op[-2:]]
        return lambda x, k, s, p: fn(x, k, s, p)
    raise KeyError(op)


def qlin(xs, q):
    s = sorted(xs); n = len(s); h = q * (n - 1)
    lo = math.floor(h); hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


# ---- banked GPU primary records -------------------------------------------
d = json.load(gzip.open(
    'verification_runs/n_samples_curve_2026-08-25/arms/CURVE_n40.json.gz'))

def primary_sens(records):
    for r in records or []:
        if r.get("name") == "perturbation_tolerance":
            for sc in (r.get("subchecks") or []):
                if isinstance(sc, dict) and sc.get("kind") == "perturbation_sensitivities":
                    return sc
    return None

banked = {}
for i, e in enumerate(d['entries']):
    s = primary_sens(e['mutant']['records'])
    if s: banked[(i, 0)] = s
    for j, ref in enumerate(e.get('refs') or []):
        s = primary_sens(ref['records'])
        if s: banked[(i, j + 1)] = s

# ---- replay ---------------------------------------------------------------
rng = np.random.default_rng(0)
entries = [(op, fam, m) for op, fam, muts in OPS for m in muts]

res = {}
for i, (op, fam, mut) in enumerate(entries):
    mk = FAMILIES[fam]; fn = R(op)
    for j in range(6):
        args = mk(rng)
        if (i, j) not in banked:
            continue
        t = [torch.from_numpy(a) if isinstance(a, np.ndarray) else a for a in args]
        x, rest = t[0], t[1:]
        base = fn(x, *rest)
        m_out = base.numel()
        xstd = x.float().std().item() or 1.0
        sigma = DELTA_SCALE * xstd

        g = torch.Generator().manual_seed(1000 + 7 * i + j)
        deltas = [torch.randn(x.shape, generator=g, dtype=DT) * sigma for _ in range(NS)]
        sens = [(fn(x + dd, *rest) - base).abs().max().item() for dd in deltas]
        tol = max(3.0 * qlin(sens, 0.95), 1e-6)

        # --- C1 / linearisation via jvp (float outputs only) ---
        rel = None
        if base.is_floating_point():
            f1 = lambda tt: fn(tt, *rest)
            rr = []
            for dd in deltas[:10]:
                try:
                    _, jd = jvp(f1, (x,), (dd,))
                except Exception:
                    rr = None; break
                sl = jd.abs().max().item()
                sa = (fn(x + dd, *rest) - base).abs().max().item()
                if sa > 0:
                    rr.append(abs(sa - sl) / sa)
            rel = st.median(rr) if rr else None

        # --- L via MC row norms ---
        L = None
        if base.is_floating_point():
            g2 = torch.Generator().manual_seed(123)
            acc = torch.zeros_like(base, dtype=torch.float64)
            for _ in range(K):
                dd = torch.randn(x.shape, generator=g2, dtype=DT) * sigma
                acc += ((fn(x + dd, *rest) - base).double()) ** 2
            L = ((acc / K).sqrt().max() / sigma).item()

        gpu_tol = max(3.0 * qlin(banked[(i, j)]['sensitivities'], 0.95), 1e-6)
        gcv_v = banked[(i, j)]['sensitivities']
        gcv = st.stdev(gcv_v) / st.fmean(gcv_v) if st.fmean(gcv_v) > 0 else None
        res.setdefault(op, []).append(
            dict(tol=tol, gpu_tol=gpu_tol, m=m_out, L=L, rel=rel, sigma=sigma,
                 cv=(st.stdev(sens) / st.fmean(sens)) if st.fmean(sens) > 0 else None,
                 gcv=gcv, floor=(3.0 * qlin(sens, 0.95) <= 1e-6),
                 gfloor=(3.0 * qlin(gcv_v, 0.95) <= 1e-6)))

# ---- report ---------------------------------------------------------------
print("=" * 118)
print("FULL 29-OPERATOR TABLE -- corpus inputs replayed exactly from default_rng(0)")
print("=" * 118)
print("%-30s %3s %7s %10s %10s %7s %9s %8s %6s" %
      ("op", "n", "m", "GPU tol", "CPU tol", "tol r", "C1 relerr", "GPUtol/3sL", "GPU sand"))
summary = []
for op, _, _ in OPS:
    v = res.get(op)
    if not v:
        print("%-30s  -- no primary records --" % op); continue
    gt = st.median([a['gpu_tol'] for a in v]); ct = st.median([a['tol'] for a in v])
    m = v[0]['m']
    rels = [a['rel'] for a in v if a['rel'] is not None]
    Ls = [a['L'] for a in v if a['L'] is not None]
    relm = st.median(rels) if rels else None
    # sandwich, evaluated per invocation
    ok_lo = ok_hi = n_ev = g_lo = g_hi = 0
    ratios = []
    gratios = []
    for a in v:
        if a['L'] is None or a['L'] <= 0 or a['floor']:
            continue
        lo = 3 * 0.6744898 * a['sigma'] * a['L']
        hi = 3 * a['sigma'] * a['L'] * (math.sqrt(2 * math.log(2 * a['m'])) +
                                        math.sqrt(2 * math.log(NS / ETA)))
        n_ev += 1
        ok_lo += (a['tol'] >= lo); ok_hi += (a['tol'] <= hi)
        g_lo += (a['gpu_tol'] >= lo); g_hi += (a['gpu_tol'] <= hi)
        gratios.append(a['gpu_tol'] / (3 * a['sigma'] * a['L']))
        ratios.append(a['tol'] / (3 * a['sigma'] * a['L']))
    sand = ("%d/%d" % (min(ok_lo, ok_hi), n_ev)) if n_ev else "n/a"
    gsand = ("%d/%d" % (min(g_lo, g_hi), n_ev)) if n_ev else "n/a"
    print("%-30s %3d %7d %10.3e %10.3e %7.3f %9s %8s %6s"
          % (op, len(v), m, gt, ct, ct / gt if gt > 0 else float('nan'),
             ("%.3f%%" % (100 * relm)) if relm is not None else "n/a",
             ("%.3f" % st.median(gratios)) if gratios else "n/a", gsand))
    summary.append((op, len(v), m, gt, ct, ct / gt if gt > 0 else float('nan'),
                    relm, sand, n_ev,
                    st.median([a['gcv'] for a in v if a['gcv'] is not None])
                    if any(a['gcv'] is not None for a in v) else None,
                    all(a['gfloor'] for a in v)))

print()
print("SANDWICH EVALUATED AGAINST THE BANKED TRITON-ON-T4 TOLERANCE:")
print("  total evaluable primary invocations: %d" % sum(x[8] for x in summary))
print("  passing both sides: %d" % sum(int(x[7].split('/')[0]) for x in summary if '/' in x[7]))
print()
tr = [s[5] for s in summary if s[5] == s[5]]
print("replay tolerance ratio CPU/GPU across %d ops: min %.3f  med %.3f  max %.3f"
      % (len(tr), min(tr), st.median(tr), max(tr)))
