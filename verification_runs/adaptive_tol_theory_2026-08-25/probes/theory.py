"""
Does adaptive_tol = 3.0 * q95(sensitivities) admit a derivation?

Replicates check_perturbation_tolerance's sensitivity mechanism EXACTLY
(same x_std, same per-sample randn_like, same float32 cast, same linear-
interpolation quantile) on math-equivalent torch references, on CPU.

Three tests:
  T1  LINEARISATION.  Is  s = ||f(x+d) - f(x)||_inf  equal to  ||J d||_inf ?
      Every bound below rests on this. Measured against torch.func.jvp.
  T2  HOMOGENEITY.    Is adaptive_tol exactly proportional to delta_scale?
      If yes, `scale` and `delta_scale` are NOT separately identifiable.
  T3  SANDWICH.       2.023*sigma*L  <=  adaptive_tol  <=  3*sigma*L*(sqrt(2 ln 2m) + sqrt(2 ln(n/eta)))
      with L = max_i ||J_i||_2 computed exactly by jacrev on small shapes.
"""
import math, json, statistics as st
import torch
from torch.func import jvp, jacrev

torch.manual_seed(0)
DT = torch.float32


def qlin(xs, q):
    s = sorted(xs); n = len(s); h = q * (n - 1)
    lo = math.floor(h); hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


# ---- math-equivalent references (perturbed arg is always the first one) ----
def r_softmax(x):      return torch.softmax(x, dim=-1)
def r_log_softmax(x):  return torch.log_softmax(x, dim=-1)
def r_gelu(x):         return torch.nn.functional.gelu(x)
def r_swish(x):        return x * torch.sigmoid(x)
def r_l2norm(x):       return x / x.norm(dim=-1, keepdim=True)
def r_l1norm(x):       return x / x.abs().sum(dim=-1, keepdim=True)
def r_frobenius(x):    return x / x.norm()
def r_sum(x):          return x.sum(dim=-1)
def r_mean(x):         return x.mean(dim=-1)
def r_maxred(x):       return x.max(dim=-1).values
def r_rmsnorm(x, g):   return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * g
def r_layernorm(x, g, b):
    m = x.mean(-1, keepdim=True); v = x.var(-1, unbiased=False, keepdim=True)
    return (x - m) * torch.rsqrt(v + 1e-5) * g + b
def r_matmul(A, B):    return A @ B
def r_sdpa(Q, K, V):   return torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1]), -1) @ V


# (name, fn, primary shape, extra-arg shapes)   shapes taken from verification/specs/*
CASES = [
    ("softmax",      r_softmax,   (1, 512),    []),
    ("softmax",      r_softmax,   (256, 1024), []),
    ("log_softmax",  r_log_softmax, (1, 512),  []),
    ("gelu",         r_gelu,      (333,),      []),
    ("gelu",         r_gelu,      (4096,),     []),
    ("swish",        r_swish,     (333,),      []),
    ("l2norm",       r_l2norm,    (1, 512),    []),
    ("l1norm",       r_l1norm,    (1, 512),    []),
    ("frobenius_norm", r_frobenius, (20, 20),  []),
    ("frobenius_norm", r_frobenius, (37, 53),  []),
    ("sum_reduction", r_sum,      (512, 512),  []),
    ("mean_reduction", r_mean,    (512, 512),  []),
    ("max_reduction", r_maxred,   (512, 512),  []),
    ("rmsnorm",      r_rmsnorm,   (1, 512),    [(512,)]),
    ("layernorm",    r_layernorm, (1, 512),    [(512,), (512,)]),
    ("layernorm",    r_layernorm, (512, 512),  [(512,), (512,)]),
    ("matmul",       r_matmul,    (1, 512),    [(512, 512)]),
    ("matmul",       r_matmul,    (333, 257),  [(257, 129)]),
    ("sdpa",         r_sdpa,      (128, 64),   [(128, 64), (128, 64)]),
]

N = 40
SEED = 1


def sens_vector(fn, x, extras, n=N, delta_scale=1e-3, seed=SEED):
    """Byte-for-byte the mechanism in perturbation.py (non-batched path)."""
    g = torch.Generator().manual_seed(seed)
    ref_base = fn(x, *extras)
    x_std = x.float().std().item()
    if x_std == 0:
        x_std = 1.0
    out = []
    for _ in range(n):
        d = torch.randn(x.shape, generator=g, dtype=x.dtype) * delta_scale * x_std
        out.append(((fn(x + d, *extras) - ref_base).abs().max(), d))
    sv = torch.stack([o[0] for o in out]).to(torch.float32)
    return sv, [o[1] for o in out], x_std


print("=" * 100)
print("T1  LINEARISATION:  s = ||f(x+d)-f(x)||_inf   vs   s_lin = ||J d||_inf")
print("=" * 100)
print("%-16s %-14s %8s %12s %12s %10s" %
      ("op", "shape", "m_out", "med s", "med s_lin", "rel err"))
T1 = {}
for name, fn, shp, extra_shapes in CASES:
    torch.manual_seed(7)
    x = torch.randn(*shp, dtype=DT)
    extras = [torch.randn(*s, dtype=DT) for s in extra_shapes]
    if name == "layernorm":
        extras = [torch.ones(*extra_shapes[0], dtype=DT),
                  torch.zeros(*extra_shapes[1], dtype=DT)]
    if name == "rmsnorm":
        extras = [torch.ones(*extra_shapes[0], dtype=DT)]
    sv, deltas, x_std = sens_vector(fn, x, extras)
    m_out = fn(x, *extras).numel()

    f1 = lambda t: fn(t, *extras)
    rel = []
    slin = []
    for d in deltas[:12]:
        _, jd = jvp(f1, (x,), (d,))
        s_lin = jd.abs().max().item()
        s_act = (fn(x + d, *extras) - fn(x, *extras)).abs().max().item()
        slin.append(s_lin)
        if s_act > 0:
            rel.append(abs(s_act - s_lin) / s_act)
    T1[(name, shp)] = (st.median(sv.tolist()), st.median(slin),
                       st.median(rel) if rel else float('nan'), m_out)
    print("%-16s %-14s %8d %12.4e %12.4e %9.2f%%" %
          (name, str(shp), m_out, st.median(sv.tolist()), st.median(slin),
           100 * (st.median(rel) if rel else float('nan'))))

print()
print("=" * 100)
print("T2  HOMOGENEITY:  log-log slope of adaptive_tol vs delta_scale")
print("     linearisation => slope exactly 1.0 => `scale`(3.0) and `delta_scale`(1e-3)")
print("     are ONE parameter (their product), not two.")
print("=" * 100)
SCALES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
print("%-16s %-14s %s   %8s" % ("op", "shape",
      "  ".join("tol@%.0e" % s for s in SCALES), "slope"))
for name, fn, shp, extra_shapes in CASES:
    torch.manual_seed(7)
    x = torch.randn(*shp, dtype=DT)
    extras = [torch.randn(*s, dtype=DT) for s in extra_shapes]
    if name == "layernorm":
        extras = [torch.ones(*extra_shapes[0], dtype=DT), torch.zeros(*extra_shapes[1], dtype=DT)]
    if name == "rmsnorm":
        extras = [torch.ones(*extra_shapes[0], dtype=DT)]
    tols = []
    for ds in SCALES:
        sv, _, _ = sens_vector(fn, x, extras, delta_scale=ds)
        tols.append(3.0 * qlin(sv.tolist(), 0.95))
    # slope over the three central decades (avoid fp floor at 1e-5 and
    # curvature at 1e-1)
    lx = [math.log10(s) for s in SCALES[1:4]]
    ly = [math.log10(t) if t > 0 else float('nan') for t in tols[1:4]]
    slope = (ly[-1] - ly[0]) / (lx[-1] - lx[0])
    print("%-16s %-14s %s   %8.4f" % (name, str(shp),
          "  ".join("%9.2e" % t for t in tols), slope))
