"""Pass 2: K-convergence, the saturating regime, and the two declined operators.

Q1  K-CONVERGENCE. Pass 1 measured L at K=400, where the original round showed
    the probe is biased HIGH by ~8% and only converges onto the closed form at
    K=20000. Repeat at K in {400, 4000, 20000} so "the closed forms are right"
    is separable from "the estimator is biased", exactly as GPU_NATIVE 2 did.

Q2  THE SATURATING REGIME. Pass 1 used ordinary make_inputs draws. The spread
    claim under test (sigmoid/tanh/swiglu ~5e10/7e9/8e9) was measured on
    SATURATING input, so pass 1 did not test it. This runs each spec's own
    adversarial variants natively and reports spread + sandwich there.

Q3  Why row_norms declined for diagonal_matmul / triangular_matmul.
"""
import importlib, json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, "/content")
from phase1_kernels import KERNELS
from verification.layer2_numeric_oracle.structural_l import row_norms, y_profile

OUT = "/content/pass2.jsonl"
NS, ETA, DELTA_SCALE = 40, 0.05, 1e-3
K_LADDER = [400, 4000, 20000]
K_SUB = ["relu", "sigmoid", "tanh", "swiglu", "cumsum", "cumsum_reverse",
         "logsumexp", "rope", "matvec", "std_reduction", "mse_loss", "new_gelu"]

def qlin(xs, q):
    s = sorted(xs); n = len(s); h = q*(n-1); lo = math.floor(h); hi = min(lo+1, n-1)
    return s[lo] + (h-lo)*(s[hi]-s[lo])

def split(i):
    return (i[0], list(i[1:])) if isinstance(i, tuple) else (i, [])

fh = open(OUT, "a")
def emit(r):
    fh.write(json.dumps(r)+"\n"); fh.flush(); os.fsync(fh.fileno())

t0 = time.time()

# ---- Q3 first, it is cheap -------------------------------------------------
for op in ("diagonal_matmul", "triangular_matmul"):
    spec = importlib.import_module(f"verification.specs.{op}").get_spec()
    torch.manual_seed(1)
    inp = spec.make_inputs(spec.valid_shapes[0], "cuda", torch.float32)
    x, rest = split(inp)
    try:
        rn = row_norms(op, x, list(rest))
        info = dict(op=op, rn_none=rn is None,
                    numel=int(rn.numel()) if rn is not None else None,
                    finite=bool(torch.isfinite(rn).all()) if rn is not None else None,
                    n_positive=int((rn > 0).sum().item()) if rn is not None else None,
                    out_numel=int(KERNELS[op](*inp).numel()))
        if rn is not None:
            y = y_profile(rn.float(), NS)
            info["y_profile"] = y
            info["y_is_none"] = y is None
    except Exception as e:
        info = dict(op=op, error=repr(e)[:200])
    info["kind"] = "declined_diag"
    emit(info); print("Q3", info, flush=True)

# ---- Q1 K-convergence ------------------------------------------------------
print("=== Q1 K-convergence ===", flush=True)
for op in K_SUB:
    spec = importlib.import_module(f"verification.specs.{op}").get_spec()
    torch.manual_seed(9000)
    inp = spec.make_inputs(spec.valid_shapes[0], "cuda", torch.float32)
    x, rest = split(inp)
    fn = KERNELS[op]
    base = fn(x, *rest)
    xs = x.float().std().item()
    sigma = DELTA_SCALE * (xs if math.isfinite(xs) and xs > 0 else 1.0)
    rn = row_norms(op, x, list(rest))
    Lc = rn.max().item()
    rec = dict(op=op, kind="kconv", L_closed=Lc, sigma=sigma)
    g = torch.Generator(device=x.device).manual_seed(123)
    acc = torch.zeros_like(base, dtype=torch.float64)
    seen = 0
    for K in K_LADDER:
        while seen < K:
            d = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)*sigma
            acc += ((fn(x+d, *rest) - base).double())**2
            seen += 1
        rec[f"L_mc_{K}"] = ((acc/seen).sqrt().max()/sigma).item()
        rec[f"ratio_{K}"] = rec[f"L_mc_{K}"]/Lc
    emit(rec)
    print("  %-18s closed %.4e  r400 %.3f  r4k %.3f  r20k %.3f  [%.0fs]" %
          (op, Lc, rec["ratio_400"], rec["ratio_4000"], rec["ratio_20000"],
           time.time()-t0), flush=True)

# ---- Q2 saturating / adversarial regime ------------------------------------
print("=== Q2 adversarial regime ===", flush=True)
for op in KERNELS:
    spec = importlib.import_module(f"verification.specs.{op}").get_spec()
    torch.manual_seed(4242)
    inp = spec.make_inputs(spec.valid_shapes[0], "cuda", torch.float32)
    fn = KERNELS[op]
    try:
        variants = [("primary", inp)] + list(spec.get_adversarial_inputs(inp))
    except Exception as e:
        emit(dict(op=op, kind="adv", error=repr(e)[:200])); continue
    for name, vin in variants:
        x, rest = split(vin)
        try:
            base = fn(x, *rest)
            if not torch.isfinite(base).all():
                emit(dict(op=op, kind="adv", variant=name,
                          note="non-finite kernel output on this variant")); continue
            xs = x.float().std().item()
            sigma = DELTA_SCALE * (xs if math.isfinite(xs) and xs > 0 else 1.0)
            g = torch.Generator(device=x.device).manual_seed(777)
            deltas = [torch.randn(x.shape, generator=g, device=x.device,
                                  dtype=x.dtype)*sigma for _ in range(NS)]
            sens = [(fn(x+d, *rest)-base).abs().max().item() for d in deltas]
            tol = max(3.0*qlin(sens, 0.95), 1e-6)
            g2 = torch.Generator(device=x.device).manual_seed(123)
            acc = torch.zeros_like(base, dtype=torch.float64)
            for _ in range(400):
                d = torch.randn(x.shape, generator=g2, device=x.device, dtype=x.dtype)*sigma
                acc += ((fn(x+d, *rest)-base).double())**2
            prof = (acc/400).sqrt().flatten()/sigma
            L = prof.max().item()
            pos = prof[prof > 0]
            spread = (pos.max()/pos.median()).item() if pos.numel() else float("nan")
            zf = float((prof == 0).double().mean().item())
            if L <= 0 or not math.isfinite(L):
                emit(dict(op=op, kind="adv", variant=name, L=L,
                          note="L<=0, sandwich vacuous")); continue
            lo = 3*0.6744898*sigma*L
            hi = 3*sigma*L*(math.sqrt(2*math.log(2*base.numel()))+math.sqrt(2*math.log(NS/ETA)))
            # linearisation defect at t=0.1
            s1 = [(fn(x+d, *rest)-base).abs().max().item() for d in deltas[:10]]
            s01 = [(fn(x+0.1*d, *rest)-base).abs().max().item() for d in deltas[:10]]
            dfc = [abs(a-b/0.1)/a for a, b in zip(s1, s01) if a > 0]
            ulp = torch.finfo(base.dtype).eps*max(base.abs().max().item(), 1e-30)
            emit(dict(op=op, kind="adv", variant=name, m=base.numel(), sigma=sigma,
                      L=L, tol=tol, lo=lo, hi=hi, ok_lo=tol >= lo, ok_hi=tol <= hi,
                      ratio=tol/(3*sigma*L), spread=spread, zero_frac=zf,
                      defect_t01=float(np.median(dfc)) if dfc else None,
                      s_over_ulp=min(sens)/ulp if ulp > 0 else None,
                      cv=float(np.std(sens, ddof=1)/np.mean(sens)) if np.mean(sens) > 0 else None))
        except Exception as e:
            emit(dict(op=op, kind="adv", variant=name, error=repr(e)[:200]))
    print("  %-20s done [%.0fs]" % (op, time.time()-t0), flush=True)

fh.close()
print("PASS2 DONE %.1fs" % (time.time()-t0), flush=True)
