"""
Diagnose the CPU/GPU sensitivity discrepancy by REPLAYING THE EXACT INPUTS.

The banked GPU run (probe_redundancy.py) builds inputs from
`np.random.default_rng(0)` via entry["input_fn"](rng).  Per corpus entry it
makes exactly 6 *counted* draws -- 1 mutant call then 5 reference calls -- the
2 warm calls being rolled back by the snapshot/restore around them.  So entry
k's inputs are draws 6k .. 6k+5, and the whole sequence is reproducible on any
machine, bit for bit, because numpy's Generator is device-independent.

The perturbation deltas are NOT reproducible (torch CUDA RNG), so per-sample
s_k cannot match.  But the SCALE of the sensitivity vector (mean, q95,
adaptive_tol) is what the sandwich bound is about, and with n=40 draws its own
sampling error is only ~CV/sqrt(40) ~ 2-3%.  Agreement at that level on the
SAME input is decisive; disagreement points at the kernel.

Compare: banked Triton-on-T4  vs  math-equivalent torch-on-CPU, same input.
"""
import gzip, json, math, statistics as st
import numpy as np
import torch
import torch.nn.functional as F

DT = torch.float32
NS = 40
DELTA_SCALE = 1e-3

# --------------------------------------------------------------------------
# input generators -- copied verbatim from benchmarks/autokernel/files/
# tritonbench_registry.py so the numpy draw sequence matches exactly
# --------------------------------------------------------------------------
def _mk_single(rng):
    return (rng.normal(size=(64, 128)).astype(np.float32),)

def _mk_triple(shape_x, shape_w):
    def _mk(rng):
        x = rng.normal(size=shape_x).astype(np.float32)
        w1 = rng.normal(size=shape_w).astype(np.float32)
        w2 = rng.normal(size=shape_w).astype(np.float32)
        return (x, w1, w2)
    return _mk

def _mk_rmsnorm(rng):
    x = rng.normal(size=(64, 128)).astype(np.float32)
    gamma = rng.normal(size=(128,)).astype(np.float32)
    return (x, gamma)

def _mk_matmul(rng):
    A = rng.normal(size=(32, 16)).astype(np.float32)
    B = rng.normal(size=(16, 32)).astype(np.float32)
    return (A, B)

def _mk_attention(rng):
    N, D = 64, 32
    Q = rng.normal(size=(N, D)).astype(np.float32)
    K = rng.normal(size=(N, D)).astype(np.float32)
    V = rng.normal(size=(N, D)).astype(np.float32)
    return (Q, K, V)

def _mk_groupnorm(rng):
    N, C, H, W = 2, 8, 4, 4
    num_groups = 2
    x = rng.normal(size=(N, C, H, W)).astype(np.float32)
    weight = rng.normal(size=(C,)).astype(np.float32)
    bias = rng.normal(size=(C,)).astype(np.float32)
    return (x, num_groups, weight, bias)

def _mk_batchnorm(rng):
    N, C, H, W = 2, 8, 4, 4
    x = rng.normal(size=(N, C, H, W)).astype(np.float32)
    running_mean = rng.normal(size=(C,)).astype(np.float32)
    running_var = rng.uniform(0.5, 2.0, size=(C,)).astype(np.float32)
    weight = rng.normal(size=(C,)).astype(np.float32)
    bias = rng.normal(size=(C,)).astype(np.float32)
    return (x, running_mean, running_var, weight, bias)

def _mk_cross_entropy(rng):
    n_rows, n_cols = 64, 32
    logits = rng.normal(size=(n_rows, n_cols)).astype(np.float32)
    targets = rng.integers(0, n_cols, size=(n_rows,)).astype(np.int64)
    return (logits, targets)

def _mk_pool(shape_x, kernel_size, stride, padding):
    def _mk(rng):
        x = rng.normal(size=shape_x).astype(np.float32)
        return (x, kernel_size, stride, padding)
    return _mk

FAMILIES = {
    "single":        _mk_single,
    "layernorm":     _mk_triple((64, 128), (128,)),
    "instancenorm":  _mk_triple((2, 4, 4, 4), (4,)),
    "rmsnorm":       _mk_rmsnorm,
    "matmul":        _mk_matmul,
    "attention":     _mk_attention,
    "groupnorm":     _mk_groupnorm,
    "batchnorm":     _mk_batchnorm,
    "cross_entropy": _mk_cross_entropy,
    "pool1d":        _mk_pool((2, 3, 32), 4, 4, 0),
    "pool2d":        _mk_pool((2, 3, 16, 16), 4, 4, 0),
    "pool3d":        _mk_pool((2, 3, 8, 8, 8), 2, 2, 0),
}

OPS = [
    ("argmax", "single", ["tiebreak"]),
    ("argmin", "single", ["tiebreak"]),
    ("avg_pool1d", "pool1d", ["wrong_divisor"]),
    ("avg_pool2d", "pool2d", ["wrong_divisor"]),
    ("avg_pool3d", "pool3d", ["wrong_divisor"]),
    ("batchnorm", "batchnorm", ["wrong_running_stats_broadcast"]),
    ("causal_flash_attention", "attention", ["wrong_causal_mask"]),
    ("cross_entropy", "cross_entropy", ["missing_max_subtraction"]),
    ("flash_attention", "attention", ["approx_denom", "drop_last_tile",
                                      "skip_rescaling", "wrong_mask"]),
    ("frobenius_norm", "single", ["wrong_norm"]),
    ("gelu", "single", ["sigmoid_approx"]),
    ("groupnorm", "groupnorm", ["ignore_affine"]),
    ("instancenorm", "instancenorm", ["skip_eps"]),
    ("l1norm", "single", ["partial_reduction"]),
    ("l2norm", "single", ["wrong_norm"]),
    ("layernorm", "layernorm", ["ignore_gamma_beta", "skip_mean_subtract",
                                "wrong_variance_estimate"]),
    ("log_softmax", "single", ["skip_max_subtraction"]),
    ("matmul", "matmul", ["partial_k_reduct", "skip_boundary_tiles",
                          "swapped_strides", "wrong_dtype"]),
    ("max_pool1d", "pool1d", ["wrong_padding"]),
    ("max_pool2d", "pool2d", ["wrong_padding"]),
    ("max_pool3d", "pool3d", ["wrong_padding"]),
    ("max_reduction", "single", ["wrong_padding"]),
    ("mean_reduction", "single", ["partial_reduction"]),
    ("min_reduction", "single", ["wrong_padding"]),
    ("rmsnorm", "rmsnorm", ["ignore_gamma", "partial_reduction", "wrong_norm"]),
    ("scaled_dot_product_attention", "attention", ["wrong_mask"]),
    ("softmax", "single", ["first_tile", "wrong_reduction"]),
    ("sum_reduction", "single", ["partial_reduction"]),
    ("swish", "single", ["linear_sigmoid_approx"]),
]

# --------------------------------------------------------------------------
# math-equivalent torch references, matched line-by-line to
# TritonBench/reference/*.py (eps values, reduction axes, gelu = exact erf,
# avg_pool count_include_pad=True, causal mask j<=i, instancenorm channel =
# row_idx % C, groupnorm gamma expanded per (batch, group) row).
# `x` is always the PRIMARY (perturbed) tensor; rest are held fixed.
# --------------------------------------------------------------------------
SQRT2INV = 0.7071067811865476

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
            N, C = x.shape[0], x.shape[1]
            sp = x.shape[2:]
            x2 = x.contiguous().view(N * C, -1)
            m = x2.mean(-1, keepdim=True)
            v = ((x2 - m) ** 2).mean(-1, keepdim=True)
            ch = torch.arange(N * C) % C
            y = (x2 - m) * torch.rsqrt(v + 1e-5) * w[ch].unsqueeze(-1) + b[ch].unsqueeze(-1)
            return y.view(N, C, *sp)
        return f
    if op == "groupnorm":
        def f(x, num_groups, w, b):
            N, C = x.shape[0], x.shape[1]
            sp = x.shape[2:]
            ssz = 1
            for d in sp: ssz *= d
            cpg = C // num_groups
            gsz = cpg * ssz
            x2 = x.contiguous().view(N * num_groups, gsz)
            m = x2.mean(-1, keepdim=True)
            v = ((x2 - m) ** 2).mean(-1, keepdim=True)
            g2 = w.view(num_groups, cpg).unsqueeze(-1).expand(num_groups, cpg, ssz)
            g2 = g2.reshape(num_groups, gsz).unsqueeze(0).expand(N, num_groups, gsz)
            g2 = g2.reshape(N * num_groups, gsz)
            b2 = b.view(num_groups, cpg).unsqueeze(-1).expand(num_groups, cpg, ssz)
            b2 = b2.reshape(num_groups, gsz).unsqueeze(0).expand(N, num_groups, gsz)
            b2 = b2.reshape(N * num_groups, gsz)
            y = (x2 - m) * torch.rsqrt(v + 1e-5) * g2 + b2
            return y.view(N, C, *sp)
        return f
    if op == "batchnorm":
        def f(x, rm, rv, w, b):
            sh = (1, -1) + (1,) * (x.dim() - 2)
            return (x - rm.view(sh)) * torch.rsqrt(rv.view(sh) + 1e-5) * w.view(sh) + b.view(sh)
        return f
    if op == "cross_entropy":
        return lambda x, t: -torch.log_softmax(x, -1).gather(1, t.unsqueeze(1)).squeeze(1)
    if op in ("flash_attention", "scaled_dot_product_attention"):
        def f(Q, K, V):
            S = Q @ K.transpose(-2, -1) * (1.0 / math.sqrt(Q.shape[-1]))
            return torch.softmax(S, -1) @ V
        return f
    if op == "causal_flash_attention":
        def f(Q, K, V):
            N = Q.shape[0]
            S = Q @ K.transpose(-2, -1) * (1.0 / math.sqrt(Q.shape[-1]))
            i = torch.arange(N).unsqueeze(1); j = torch.arange(N).unsqueeze(0)
            S = S.masked_fill(j > i, float("-inf"))
            return torch.softmax(S, -1) @ V
        return f
    if op.startswith("avg_pool"):
        d = op[-2:]
        fn = {"1d": F.avg_pool1d, "2d": F.avg_pool2d, "3d": F.avg_pool3d}[d]
        return lambda x, k, s, p: fn(x, k, s, p, count_include_pad=True)
    if op.startswith("max_pool"):
        d = op[-2:]
        fn = {"1d": F.max_pool1d, "2d": F.max_pool2d, "3d": F.max_pool3d}[d]
        return lambda x, k, s, p: fn(x, k, s, p)
    raise KeyError(op)


def qlin(xs, q):
    s = sorted(xs); n = len(s); h = q * (n - 1)
    lo = math.floor(h); hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def sensitivities(fn, x, rest, seed):
    """Exactly perturbation.py's non-batched path."""
    base = fn(x, *rest)
    xs = x.float().std().item()
    if xs == 0: xs = 1.0
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(NS):
        d = torch.randn(x.shape, generator=g, dtype=x.dtype) * DELTA_SCALE * xs
        out.append((fn(x + d, *rest) - base).abs().max().item())
    return out


# --------------------------------------------------------------------------
# banked GPU records, keyed by (entry_index, invocation_index)
# invocation 0 = the mutant run, 1..5 = the five reference runs
# --------------------------------------------------------------------------
d = json.load(gzip.open(
    'verification_runs/n_samples_curve_2026-08-25/arms/CURVE_n40.json.gz'))

def primary_sens(records):
    """The perturbation_tolerance record = the PRIMARY (unmodified) input."""
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

print("banked PRIMARY perturbation records: %d  (expect 6 per entry x 40 = 240)"
      % len(banked))

# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------
rng = np.random.default_rng(0)
entries = [(op, fam, m) for op, fam, muts in OPS for m in muts]
print("corpus entries reconstructed: %d  (banked: %d)  order match: %s\n"
      % (len(entries), len(d['entries']),
         all(entries[i][0] == d['entries'][i]['op'] and
             entries[i][2] == d['entries'][i]['mutant']['name']
             for i in range(min(len(entries), len(d['entries']))))))

rows = []
for i, (op, fam, mut) in enumerate(entries):
    mk = FAMILIES[fam]
    fn = R(op)
    for j in range(6):
        args = mk(rng)                      # <- the exact banked draw
        key = (i, j)
        if key not in banked:
            continue
        t = []
        for a in args:
            t.append(torch.from_numpy(a) if isinstance(a, np.ndarray) else a)
        x, rest = t[0], t[1:]
        try:
            cpu = sensitivities(fn, x, rest, seed=1000 + 7 * i + j)
        except Exception as ex:
            rows.append((op, mut, j, None, None, str(ex)[:40]))
            continue
        gpu = banked[key]['sensitivities']
        rows.append((op, mut, j, cpu, gpu, None))

print("=" * 100)
print("SAME INPUT, Triton-on-T4 vs torch-on-CPU: ratio CPU/GPU of the")
print("statistic the bound is about (adaptive_tol = 3*q95, floored at 1e-6)")
print("=" * 100)
print("%-30s %5s %12s %12s %8s %8s" %
      ("op", "n", "GPU tol", "CPU tol", "tol r", "CV r"))
byop = {}
for op, mut, j, cpu, gpu, err in rows:
    if cpu is None or gpu is None:
        continue
    gt = max(3 * qlin(gpu, .95), 1e-6)
    ct = max(3 * qlin(cpu, .95), 1e-6)
    gcv = st.stdev(gpu) / st.fmean(gpu) if st.fmean(gpu) > 0 else float('nan')
    ccv = st.stdev(cpu) / st.fmean(cpu) if st.fmean(cpu) > 0 else float('nan')
    byop.setdefault(op, []).append((gt, ct, gcv, ccv))
worst = []
for op in [o for o, _, _ in OPS]:
    v = byop.get(op)
    if not v:
        continue
    gt = st.median([a[0] for a in v]); ct = st.median([a[1] for a in v])
    gcv = st.median([a[2] for a in v]); ccv = st.median([a[3] for a in v])
    tr = ct / gt if gt > 0 else float('nan')
    cr = ccv / gcv if gcv and gcv == gcv and gcv > 0 else float('nan')
    print("%-30s %5d %12.4e %12.4e %8.3f %8.3f" % (op, len(v), gt, ct, tr, cr))
    worst.append((abs(math.log10(tr)) if tr > 0 else 99, op, tr))
worst.sort(reverse=True)
print("\nlargest |log10| tolerance ratios:")
for w, op, tr in worst[:8]:
    print("   %-30s CPU/GPU = %.4f" % (op, tr))
