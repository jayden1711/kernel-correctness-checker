"""Every Phase-1 Triton kernel vs its torch reference, on the T4.

Runs BEFORE any measurement. A wrong kernel would produce a confident, wrong
sandwich result, and that is the failure mode this project has been most
careful about.
"""
import sys, math, json
sys.path.insert(0, "/content")
import torch, torch.nn.functional as F
from phase1_kernels import KERNELS

torch.manual_seed(0)
D = "cuda"

def rope_ref(x, c, s):
    h = x.shape[-1] // 2
    return torch.cat([x[:, :h] * c - x[:, h:] * s, x[:, :h] * s + x[:, h:] * c], -1)

CASES = {
 "relu":        (lambda: (torch.randn(4096, device=D),), lambda x: F.relu(x)),
 "leaky_relu":  (lambda: (torch.randn(4096, device=D), 0.2), lambda x, s: F.leaky_relu(x, s)),
 "sigmoid":     (lambda: (torch.randn(4096, device=D),), torch.sigmoid),
 "tanh":        (lambda: (torch.randn(4096, device=D),), torch.tanh),
 "selu":        (lambda: (torch.randn(4096, device=D),), F.selu),
 "elu":         (lambda: (torch.randn(4096, device=D), 2.0), lambda x, a: F.elu(x, a)),
 "softplus":    (lambda: (torch.randn(4096, device=D), 2.0), lambda x, b: F.softplus(x, beta=b)),
 "hardsigmoid": (lambda: (torch.randn(4096, device=D) * 2,), F.hardsigmoid),
 "new_gelu":    (lambda: (torch.randn(4096, device=D),), lambda x: F.gelu(x, approximate="tanh")),

 "cumsum":           (lambda: (torch.randn(64, 512, device=D),), lambda x: torch.cumsum(x, -1)),
 "cumsum_reverse":   (lambda: (torch.randn(64, 512, device=D),), lambda x: torch.cumsum(x.flip(-1), -1).flip(-1)),
 "cumsum_exclusive": (lambda: (torch.randn(64, 512, device=D),), lambda x: torch.cumsum(x, -1) - x),
 "masked_cumsum":    (lambda: (torch.randn(64, 512, device=D),
                               torch.randint(0, 2, (64, 512), device=D).float()),
                      lambda x, m: torch.cumsum(x * m, -1)),

 "matvec":            (lambda: (torch.randn(512, 512, device=D), torch.randn(512, device=D)),
                       lambda A, v: A @ v),
 "batched_matmul":    (lambda: (torch.randn(4, 128, 128, device=D), torch.randn(4, 128, 128, device=D)),
                       lambda A, B: torch.bmm(A, B)),
 "diagonal_matmul":   (lambda: (torch.randn(512, device=D), torch.randn(512, 512, device=D)),
                       lambda d, B: torch.diag(d) @ B),
 "triangular_matmul": (lambda: (torch.randn(512, 512, device=D), torch.randn(512, 512, device=D)),
                       lambda A, B: torch.tril(A @ B)),

 "mse_loss":   (lambda: (torch.randn(512, 512, device=D), torch.randn(512, 512, device=D)),
                lambda x, t: F.mse_loss(x, t)),
 "huber_loss": (lambda: (torch.randn(512, 512, device=D), torch.randn(512, 512, device=D)),
                lambda x, t: F.smooth_l1_loss(x, t, beta=1.0)),
 "bce_loss":   (lambda: (torch.rand(512, 512, device=D) * 0.98 + 0.01,
                         torch.randint(0, 2, (512, 512), device=D).float()),
                lambda p, t: F.binary_cross_entropy(p, t)),
 "kldiv_loss": (lambda: (torch.log_softmax(torch.randn(512, 512, device=D), -1),
                         torch.softmax(torch.randn(512, 512, device=D), -1)),
                lambda lq, p: F.kl_div(lq, p, reduction="batchmean")),
 "nll_loss":   (lambda: (torch.log_softmax(torch.randn(512, 512, device=D), -1),
                         torch.randint(0, 512, (512,), device=D)),
                lambda lp, t: F.nll_loss(lp, t)),

 "rope":          (lambda: (lambda th: (torch.randn(512, 128, device=D), torch.cos(th), torch.sin(th)))(
                       torch.randn(512, 64, device=D)), rope_ref),
 "swiglu":        (lambda: (torch.randn(512, 512, device=D),),
                   lambda x: F.silu(x[:, :256]) * x[:, 256:]),
 "logsumexp":     (lambda: (torch.randn(512, 512, device=D),), lambda x: torch.logsumexp(x, -1)),
 "std_reduction": (lambda: (torch.randn(512, 512, device=D),), lambda x: x.std(-1)),
 "var_reduction": (lambda: (torch.randn(512, 512, device=D),), lambda x: x.var(-1)),
}

print(f"{'operator':20s} {'max_err':>12s} {'rel':>10s}  verdict")
print("-" * 62)
bad = []
for op, (mk, ref) in CASES.items():
    try:
        args = mk()
        got = KERNELS[op](*args)
        exp = ref(*args)
        if got.shape != exp.shape:
            print(f"{op:20s} {'--':>12s} {'--':>10s}  SHAPE {tuple(got.shape)} vs {tuple(exp.shape)}")
            bad.append(op); continue
        err = (got.float() - exp.float()).abs().max().item()
        den = exp.float().abs().max().item()
        rel = err / den if den > 0 else err
        ok = rel < 2e-5 or err < 1e-5
        print(f"{op:20s} {err:12.3e} {rel:10.2e}  {'OK' if ok else 'MISMATCH'}")
        if not ok: bad.append(op)
    except Exception as e:
        print(f"{op:20s} {'--':>12s} {'--':>10s}  ERROR {type(e).__name__}: {str(e)[:60]}")
        bad.append(op)
print("-" * 62)
print(f"{len(CASES)-len(bad)}/{len(CASES)} kernels correct")
if bad: print("FAILING:", bad)
