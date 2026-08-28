"""
T3: the sandwich.   Let sigma = delta_scale * std(x),  L = max_i ||J_i||_2
(the 2->inf operator norm of the reference's Jacobian at x), m = output dim,
n = n_samples.

Under linearisation (verified to <0.1% in T1),  s_k = ||J d_k||_inf  with
d_k = sigma * g_k, g_k ~ N(0, I) i.i.d.  Then

  LOWER   adaptive_tol >= 3 * 0.6745 * sigma * L = 2.023 sigma L
          w.p. >= 1 - (n+1)/2^n     [q95_n >= X_(n-1:n) >= parent median,
                                     and parent median >= median|<J_i*,d>|]
  UPPER   adaptive_tol <= 3 sigma L (sqrt(2 ln 2m) + sqrt(2 ln(n/eta)))
          w.p. >= 1 - eta           [E s <= sigma L sqrt(2 ln 2m) (max of 2m
                                     gaussians) + Borell-TIS + union over n]

L is estimated per-row by Monte Carlo:  E[(J d)_i^2] = sigma^2 ||J_i||^2, so
Lhat = max_i rms_k((f(x+d_k)-f(x))_i) / sigma.  Validated against exact
row norms from torch.func.jacrev on the small cases.
"""
import math
import torch
from torch.func import jacrev

DT = torch.float32
K = 400          # MC samples for row norms
N = 40           # n_samples for the tolerance itself
ETA = 0.05


def qlin(xs, q):
    s = sorted(xs); n = len(s); h = q * (n - 1)
    lo = math.floor(h); hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


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
def r_sdpa(Q, K_, V):  return torch.softmax(Q @ K_.transpose(-2, -1) / math.sqrt(Q.shape[-1]), -1) @ V


CASES = [
    ("softmax",        r_softmax,     (1, 512),    []),
    ("softmax",        r_softmax,     (256, 1024), []),
    ("log_softmax",    r_log_softmax, (1, 512),    []),
    ("gelu",           r_gelu,        (333,),      []),
    ("gelu",           r_gelu,        (4096,),     []),
    ("swish",          r_swish,       (333,),      []),
    ("l2norm",         r_l2norm,      (1, 512),    []),
    ("l1norm",         r_l1norm,      (1, 512),    []),
    ("frobenius_norm", r_frobenius,   (20, 20),    []),
    ("frobenius_norm", r_frobenius,   (37, 53),    []),
    ("sum_reduction",  r_sum,         (512, 512),  []),
    ("mean_reduction", r_mean,        (512, 512),  []),
    ("max_reduction",  r_maxred,      (512, 512),  []),
    ("rmsnorm",        r_rmsnorm,     (1, 512),    [(512,)]),
    ("layernorm",      r_layernorm,   (1, 512),    [(512,), (512,)]),
    ("layernorm",      r_layernorm,   (512, 512),  [(512,), (512,)]),
    ("matmul",         r_matmul,      (1, 512),    [(512, 512)]),
    ("matmul",         r_matmul,      (333, 257),  [(257, 129)]),
    ("sdpa",           r_sdpa,        (128, 64),   [(128, 64), (128, 64)]),
]


def build(name, extra_shapes):
    if name == "layernorm":
        return [torch.ones(*extra_shapes[0], dtype=DT), torch.zeros(*extra_shapes[1], dtype=DT)]
    if name == "rmsnorm":
        return [torch.ones(*extra_shapes[0], dtype=DT)]
    return [torch.randn(*s, dtype=DT) for s in extra_shapes]


print("%-16s %-13s %8s %10s %10s %10s %8s %8s %8s" %
      ("op", "shape", "m", "sigma", "L_hat", "adap_tol", "tol/3sL",
       "sq2ln2m", "UB/tol"))
print("-" * 110)

viol_lo = viol_hi = 0
rows = []
for name, fn, shp, es in CASES:
    torch.manual_seed(7)
    x = torch.randn(*shp, dtype=DT)
    extras = build(name, es)
    base = fn(x, *extras)
    m = base.numel()
    x_std = x.float().std().item()
    sigma = 1e-3 * x_std

    # --- MC row norms -> L_hat ---
    g = torch.Generator().manual_seed(123)
    acc = torch.zeros_like(base, dtype=torch.float64)
    for _ in range(K):
        dd = torch.randn(x.shape, generator=g, dtype=DT) * sigma
        acc += ((fn(x + dd, *extras) - base).double()) ** 2
    row_rms = (acc / K).sqrt()
    L_hat = (row_rms.max() / sigma).item()

    # --- the actual shipped statistic, same RNG discipline as perturbation.py ---
    g2 = torch.Generator().manual_seed(1)
    sens = []
    for _ in range(N):
        dd = torch.randn(x.shape, generator=g2, dtype=DT) * sigma
        sens.append((fn(x + dd, *extras) - base).abs().max().item())
    tol = max(3.0 * qlin(sens, 0.95), 1e-6)

    lo = 3 * 0.6744898 * sigma * L_hat
    hi = 3 * sigma * L_hat * (math.sqrt(2 * math.log(2 * m)) +
                              math.sqrt(2 * math.log(N / ETA)))
    ratio = tol / (3 * sigma * L_hat)
    if tol < lo: viol_lo += 1
    if tol > hi: viol_hi += 1
    rows.append((name, shp, m, ratio, math.sqrt(2 * math.log(2 * m)), hi / tol))
    print("%-16s %-13s %8d %10.3e %10.3e %10.3e %8.3f %8.3f %8.2f" %
          (name, str(shp), m, sigma, L_hat, tol, ratio,
           math.sqrt(2 * math.log(2 * m)), hi / tol))

print("-" * 110)
print("lower-bound violations: %d/%d      upper-bound violations: %d/%d"
      % (viol_lo, len(CASES), viol_hi, len(CASES)))
print()
print("tol/(3 sigma L)  vs  sqrt(2 ln 2m):   the upper bound's leading term.")
print("  ratio of the two (1.0 = the sqrt(2 ln 2m) term is exactly right):")
for name, shp, m, ratio, s2, ub in rows:
    print("   %-16s %-13s m=%-7d  %.3f" % (name, str(shp), m, ratio / s2))

# --- validate L_hat against exact jacrev row norms on two small cases -------
print()
print("VALIDATION of the Monte-Carlo L_hat against exact jacrev row norms:")
for name, fn, shp, es in [c for c in CASES if c[0] in ("softmax", "gelu", "l2norm")
                          and math.prod(c[2]) <= 512]:
    torch.manual_seed(7)
    x = torch.randn(*shp, dtype=DT)
    extras = build(name, es)
    J = jacrev(lambda t: fn(t, *extras))(x)
    J = J.reshape(fn(x, *extras).numel(), x.numel())
    L_exact = J.norm(dim=1).max().item()
    x_std = x.float().std().item(); sigma = 1e-3 * x_std
    g = torch.Generator().manual_seed(123)
    base = fn(x, *extras)
    acc = torch.zeros_like(base, dtype=torch.float64)
    for _ in range(K):
        dd = torch.randn(x.shape, generator=g, dtype=DT) * sigma
        acc += ((fn(x + dd, *extras) - base).double()) ** 2
    L_hat = ((acc / K).sqrt().max() / sigma).item()
    print("   %-14s %-12s  L_exact %.6f   L_hat %.6f   err %.2f%%"
          % (name, str(shp), L_exact, L_hat, 100 * abs(L_hat - L_exact) / L_exact))
