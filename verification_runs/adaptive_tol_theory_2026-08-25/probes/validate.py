"""Does the CPU torch surrogate reproduce the banked GPU/Triton sensitivity
statistics?  If yes, the linearisation and sandwich results measured on the
surrogate transfer to the shipped kernels.

Compares the *shape* statistic CV = sd/mean of the 40-sample sensitivity
vector, which is scale-free (independent of sigma, L and the kernel's ulp
noise) and is exactly the quantity the Gumbel/max-of-gaussians model predicts.
"""
import math, statistics as st, gzip, json
import torch

DT = torch.float32
N = 40

BANKED = {}
d = json.load(gzip.open('verification_runs/n_samples_curve_2026-08-25/arms/CURVE_n40.json.gz'))
def walk(o):
    if isinstance(o, dict):
        if o.get('kind') == 'perturbation_sensitivities': yield o
        for v in o.values(): yield from walk(v)
    elif isinstance(o, list):
        for v in o: yield from walk(v)
acc = {}
for e in d['entries']:
    op = e['op']
    for r in list(walk(e['mutant'])) + [x for ref in (e.get('refs') or []) for x in walk(ref)]:
        s = r['sensitivities']
        if max(s) <= 0: continue
        acc.setdefault(op, []).append(st.stdev(s) / st.fmean(s))
for k, v in acc.items():
    BANKED[k] = st.median(v)


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
def r_minred(x):       return x.min(dim=-1).values
def r_rmsnorm(x, g):   return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * g
def r_layernorm(x, g, b):
    m = x.mean(-1, keepdim=True); v = x.var(-1, unbiased=False, keepdim=True)
    return (x - m) * torch.rsqrt(v + 1e-5) * g + b
def r_matmul(A, B):    return A @ B
def r_sdpa(Q, K_, V):  return torch.softmax(Q @ K_.transpose(-2, -1) / math.sqrt(Q.shape[-1]), -1) @ V

ROWSHAPES = [(512, 512), (256, 1024), (1, 512), (1000, 333), (2048, 128)]
OPS = {
    "softmax":        (r_softmax,     ROWSHAPES, 0),
    "log_softmax":    (r_log_softmax, ROWSHAPES, 0),
    "l2norm":         (r_l2norm,      ROWSHAPES, 0),
    "l1norm":         (r_l1norm,      ROWSHAPES, 0),
    "sum_reduction":  (r_sum,         ROWSHAPES, 0),
    "mean_reduction": (r_mean,        ROWSHAPES, 0),
    "max_reduction":  (r_maxred,      ROWSHAPES, 0),
    "min_reduction":  (r_minred,      ROWSHAPES, 0),
    "gelu":           (r_gelu,        [(4096,), (1024,), (100000,), (333,)], 0),
    "swish":          (r_swish,       [(4096,), (1024,), (100000,), (333,)], 0),
    "frobenius_norm": (r_frobenius,   [(37, 53), (20, 20), (64, 64), (1, 100), (100, 1)], 0),
    "rmsnorm":        (r_rmsnorm,     ROWSHAPES, 1),
    "layernorm":      (r_layernorm,   ROWSHAPES, 2),
    "matmul":         (r_matmul,      [(512, 512, 512), (256, 512, 1024), (1, 512, 512), (333, 257, 129)], -1),
    "flash_attention": (r_sdpa,       [(128, 64), (64, 64), (256, 64), (65, 64), (192, 64)], -2),
}

print("%-16s %10s %10s %8s" % ("op", "banked GPU", "CPU torch", "ratio"))
print("-" * 48)
for name, (fn, shapes, kind) in OPS.items():
    cvs = []
    for shp in shapes:
        torch.manual_seed(7)
        if kind == -1:      # matmul (M,K,N)
            M, K, Nn = shp
            x = torch.randn(M, K, dtype=DT); extras = [torch.randn(K, Nn, dtype=DT)]
        elif kind == -2:    # attention (S,D)
            S, Dd = shp
            x = torch.randn(S, Dd, dtype=DT)
            extras = [torch.randn(S, Dd, dtype=DT), torch.randn(S, Dd, dtype=DT)]
        else:
            x = torch.randn(*shp, dtype=DT)
            if kind == 1:   extras = [torch.ones(shp[-1], dtype=DT)]
            elif kind == 2: extras = [torch.ones(shp[-1], dtype=DT), torch.zeros(shp[-1], dtype=DT)]
            else:           extras = []
        base = fn(x, *extras)
        x_std = x.float().std().item() or 1.0
        g = torch.Generator().manual_seed(1)
        sv = []
        for _ in range(N):
            dd = torch.randn(x.shape, generator=g, dtype=DT) * 1e-3 * x_std
            sv.append((fn(x + dd, *extras) - base).abs().max().item())
        if max(sv) > 0 and st.fmean(sv) > 0:
            cvs.append(st.stdev(sv) / st.fmean(sv))
    if not cvs:
        continue
    cpu = st.median(cvs)
    b = BANKED.get(name)
    print("%-16s %10s %10.4f %8s"
          % (name, ("%.4f" % b) if b else "-", cpu,
             ("%.2f" % (cpu / b)) if b else "-"))
