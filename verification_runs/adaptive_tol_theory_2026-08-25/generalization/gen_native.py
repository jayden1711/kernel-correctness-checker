"""
STRUCTURAL L and a PREDICTIVE tol FORMULA -- both evaluated natively.

Part A.  L_struct = max_i ||J_i||_2 written in closed form from each kernel's
arithmetic (no probing), compared against the SAME Monte-Carlo estimator the
native run used, re-run on the REAL TRITON KERNELS at K = 400 / 4000 / 20000.
Raising K separates "the closed form is wrong" from "the estimator is biased":
L_MC is a max over m noisy per-row estimates and is biased HIGH by
O(sqrt(2 ln m) / sqrt(2K)).

Part B.  y = tol/(3 sigma L).  Model M3 predicts y with NO fitted constant, by
simulating q95_40(max_i (||J_i||/L)|z_i|) from the closed-form row-norm profile
under an orthogonal-rows assumption.  Where rows are genuinely correlated
(softmax denominators, layernorm moments, attention) M3 must over-predict.

Closed forms -- see the report for derivations.  Attention is included exactly:
    f_id = sum_j p_ij V_jd,  and f_id depends only on Q_i, so
    grad_{Q_i} f_id = (1/sqrt(D)) K^T ( p_i * (V[:,d] - f_id) )
"""
import json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, "/content")
sys.path.insert(0, "/content/benchmarks/autokernel/files")

OUT = "/content/gen_native.jsonl"
NS = 40
DELTA_SCALE = 1e-3
SQRT2INV = 0.7071067811865476
K_LIST = [400, 4000, 20000]
NSIM = 3000
EXCLUDE = {"argmax", "argmin"}
STATIC = {"sum_reduction", "mean_reduction", "max_reduction", "min_reduction",
          "max_pool1d", "max_pool2d", "max_pool3d",
          "avg_pool1d", "avg_pool2d", "avg_pool3d"}

from tritonbench_registry import build_corpus
CORPUS = build_corpus()
dev = "cuda"
print(torch.cuda.get_device_name(0), flush=True)


def split(inputs):
    if isinstance(inputs, tuple):
        return inputs[0], list(inputs[1:])
    return inputs, []


def qlin_t(a, q, dim=-1):
    s, _ = torch.sort(a, dim=dim)
    n = s.shape[dim]
    h = q * (n - 1)
    lo = int(math.floor(h)); hi = min(lo + 1, n - 1)
    return s.select(dim, lo) + (h - lo) * (s.select(dim, hi) - s.select(dim, lo))


# ------------------------------------------------- closed-form row norms
def rownorms(op, x, rest):
    if op == "sum_reduction":
        return torch.full((x.shape[0],), math.sqrt(x.shape[-1]), device=dev)
    if op == "mean_reduction":
        return torch.full((x.shape[0],), math.sqrt(x.shape[-1]) / x.shape[-1], device=dev)
    if op in ("max_reduction", "min_reduction"):
        return torch.ones(x.shape[0], device=dev)
    if op.startswith("max_pool"):
        return torch.ones(_out_numel(op, x, rest), device=dev)
    if op.startswith("avg_pool"):
        W = rest[0] ** int(op[-2])
        return torch.full((_out_numel(op, x, rest),), math.sqrt(W) / W, device=dev)
    if op == "matmul":
        return rest[0].norm(dim=0).repeat(x.shape[0])
    if op == "gelu":
        pdf = torch.exp(-x * x / 2) / math.sqrt(2 * math.pi)
        return (0.5 * (1 + torch.erf(x * SQRT2INV)) + x * pdf).abs().flatten()
    if op == "swish":
        s = torch.sigmoid(x)
        return (s * (1 + x * (1 - s))).abs().flatten()
    if op in ("softmax", "log_softmax"):
        p = torch.softmax(x, -1)
        s2 = (p * p).sum(-1, keepdim=True)
        base = torch.sqrt((1 - 2 * p + s2).clamp_min(0))
        return (p * base).flatten() if op == "softmax" else base.flatten()
    if op == "l2norm":
        nrm = torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-12)
        u = x / nrm
        return (torch.sqrt((1 - u * u).clamp_min(0)) / nrm).flatten()
    if op == "l1norm":
        S = x.abs().sum(-1, keepdim=True) + 1e-12
        f = x / S; n = x.shape[-1]
        return (torch.sqrt((1 - 2 * f.abs() + n * f * f).clamp_min(0)) / S).flatten()
    if op == "frobenius_norm":
        nrm = torch.sqrt((x * x).sum()) + 1e-12
        u = x / nrm
        return (torch.sqrt((1 - u * u).clamp_min(0)) / nrm).flatten()
    if op == "layernorm":
        g = rest[0]; n = x.shape[-1]
        m = x.mean(-1, keepdim=True); v = ((x - m) ** 2).mean(-1, keepdim=True)
        z = (x - m) * torch.rsqrt(v + 1e-5)
        return (g.abs() * torch.rsqrt(v + 1e-5)
                * torch.sqrt((1 - 1.0 / n - z * z / n).clamp_min(0))).flatten()
    if op == "rmsnorm":
        g = rest[0]; n = x.shape[-1]
        r = torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)
        a = x * x / (n * r * r)
        c = (x * x).mean(-1, keepdim=True) / (r * r)
        return ((g.abs() / r) * torch.sqrt((1 - 2 * a + a * c).clamp_min(0))).flatten()
    if op == "batchnorm":
        rv, w = rest[1], rest[2]
        sh = (1, -1) + (1,) * (x.dim() - 2)
        return (w.view(sh).abs() * torch.rsqrt(rv.view(sh) + 1e-5)).expand_as(x).flatten().contiguous()
    if op == "instancenorm":
        w = rest[0]; N, C = x.shape[0], x.shape[1]
        x2 = x.contiguous().view(N * C, -1); n = x2.shape[-1]
        m = x2.mean(-1, keepdim=True); v = ((x2 - m) ** 2).mean(-1, keepdim=True)
        z = (x2 - m) * torch.rsqrt(v + 1e-5)
        ch = torch.arange(N * C, device=dev) % C
        return (w[ch].abs().unsqueeze(-1) * torch.rsqrt(v + 1e-5)
                * torch.sqrt((1 - 1.0 / n - z * z / n).clamp_min(0))).flatten()
    if op == "groupnorm":
        ng, w = rest[0], rest[1]
        N, C = x.shape[0], x.shape[1]; sp = x.shape[2:]
        ssz = 1
        for dd in sp: ssz *= dd
        cpg = C // ng; gsz = cpg * ssz
        x2 = x.contiguous().view(N * ng, gsz); n = gsz
        m = x2.mean(-1, keepdim=True); v = ((x2 - m) ** 2).mean(-1, keepdim=True)
        z = (x2 - m) * torch.rsqrt(v + 1e-5)
        g2 = w.view(ng, cpg).unsqueeze(-1).expand(ng, cpg, ssz).reshape(ng, gsz)
        g2 = g2.unsqueeze(0).expand(N, ng, gsz).reshape(N * ng, gsz)
        return (g2.abs() * torch.rsqrt(v + 1e-5)
                * torch.sqrt((1 - 1.0 / n - z * z / n).clamp_min(0))).flatten()
    if op == "cross_entropy":
        t = rest[0]
        p = torch.softmax(x, -1).clone()
        p[torch.arange(x.shape[0], device=dev), t] -= 1.0
        return torch.tensor([p.norm().item() / x.shape[0]], device=dev)
    if "attention" in op:
        Kk, V = rest[0], rest[1]
        N, D = x.shape
        S = x @ Kk.transpose(-2, -1) * (1.0 / math.sqrt(D))
        if op == "causal_flash_attention":
            ii = torch.arange(N, device=dev).unsqueeze(1)
            jj = torch.arange(N, device=dev).unsqueeze(0)
            S = S.masked_fill(jj > ii, float("-inf"))
        p = torch.softmax(S, -1)                       # (N, N)
        f = p @ V                                      # (N, D)
        outs = []
        for i in range(N):
            W = p[i].unsqueeze(1) * (V - f[i].unsqueeze(0))     # (N, D)
            G = (Kk.transpose(0, 1) @ W) / math.sqrt(D)         # (D, D) col d = grad wrt Q_i
            outs.append(G.norm(dim=0))
        return torch.cat(outs)
    return None


def _out_numel(op, x, rest):
    import torch.nn.functional as F
    k, s, p = rest[0], rest[1], rest[2]
    fn = {"avg_pool1d": F.avg_pool1d, "avg_pool2d": F.avg_pool2d,
          "avg_pool3d": F.avg_pool3d, "max_pool1d": F.max_pool1d,
          "max_pool2d": F.max_pool2d, "max_pool3d": F.max_pool3d}[op]
    return fn(x, k, s, p).numel()


def y_independent(rn, nsim=NSIM, seed=0):
    rn = rn[rn > 0]
    L = rn.max()
    w = (rn / L)
    if w.numel() > 30000:
        top = w[w > 1e-3]
        rest_ = w[w <= 1e-3]
        if rest_.numel():
            top = torch.cat([top, rest_[:: max(1, rest_.numel() // 8000)]])
        w = top
    g = torch.Generator(device=dev).manual_seed(seed)
    acc = 0.0
    CH = 200
    done = 0
    while done < nsim:
        b = min(CH, nsim - done)
        z = torch.randn(b, NS, w.numel(), generator=g, device=dev).abs()
        s = (z * w).max(dim=2).values                 # (b, NS)
        acc += qlin_t(s, 0.95, dim=1).sum().item()
        done += b
    return acc / nsim


fh = open(OUT, "w")
def emit(r):
    fh.write(json.dumps(r) + "\n"); fh.flush(); os.fsync(fh.fileno())


rng = np.random.default_rng(0)
for i, entry in enumerate(CORPUS):
    op = entry["op"]
    ref = entry["torch_ref_fn"]
    for j in range(6):
        np_args = entry["input_fn"](rng)
        if op in EXCLUDE or j > 0:
            continue
        x, rest = split(entry["to_torch"](np_args))
        base = ref(x, *rest)
        m = base.numel()
        xs = x.float().std().item() or 1.0
        sigma = DELTA_SCALE * xs

        rn = rownorms(op, x, rest)
        Ls = rn.max().item()

        # native MC estimator on the real kernel, at increasing K
        Lmc = {}
        g2 = torch.Generator(device=dev).manual_seed(123)
        acc = torch.zeros_like(base, dtype=torch.float64)
        done = 0
        for K in K_LIST:
            while done < K:
                d = torch.randn(x.shape, generator=g2, device=dev, dtype=x.dtype) * sigma
                acc += ((ref(x + d, *rest) - base).double()) ** 2
                done += 1
            Lmc[K] = ((acc / done).sqrt().max() / sigma).item()

        emit(dict(op=op, m=m, sigma=sigma, L_struct=Ls,
                  L_mc={str(k): v for k, v in Lmc.items()},
                  static=op in STATIC,
                  spread=float((rn.max() / rn.median()).item()),
                  n_rows=int((rn > 0).sum().item()),
                  y_M3=y_independent(rn)))
        print("%-30s L_struct %.4e  L_mc400 %.4e  L_mc20k %.4e"
              % (op, Ls, Lmc[400], Lmc[20000]), flush=True)
fh.close()
print("DONE", flush=True)
