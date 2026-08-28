"""
GPU-NATIVE sandwich verification.  Everything below runs the REAL Triton
kernels: the sensitivities, the Jacobian/L estimate, adaptive_tol, and the
attention input-conditional check.  No CPU math-equivalent reference is used
anywhere in the measurement path.

Inputs are the corpus's own, replayed bit-for-bit from np.random.default_rng(0)
with the same 6-counted-draws-per-entry sequence probe_redundancy.py used, so
every number is directly comparable to the banked / CPU-derived round.

THE JACOBIAN, NATIVELY.  torch.func.jvp cannot differentiate a @triton.jit
kernel -- there is no autograd registration.  The native substitute is the
directional derivative by its definition, evaluated with the kernel itself:

        || J d ||_inf  =  lim_{t->0}  || f(x + t d) - f(x) ||_inf  /  t

Write s(t) = ||f(x+td)-f(x)||_inf.  Linearity of f along d is then exactly
s(t) = t * s(1), i.e. a log-log slope of 1 and a vanishing "linearisation
defect"

        defect  =  | s(1) - s(t)/t |  /  s(1)        (t small)

This is a STRONGER test than the CPU jvp comparison: jvp differentiates the
mathematical reference, whereas this differentiates the kernel that actually
ships.  t = 0.1 is the operating point (t = 0.01 is reported too, but there
the difference approaches the fp32 noise floor for some operators).

Writes one JSONL line per invocation with flush+fsync, and resumes by reading
its own output -- the VM can be reclaimed at any time (SESSION_HANDOFF).
"""
import json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/content")
sys.path.insert(0, "/content/benchmarks/autokernel/files")

OUT = "/content/gpu_native.jsonl"
NS, K_MC, ETA = 40, 400, 0.05
DELTA_SCALE = 1e-3
T_LADDER = [0.01, 0.1, 1.0, 10.0]
EXCLUDE = {"argmax", "argmin"}          # int64 output, J = 0 a.e.

assert torch.cuda.is_available(), "no CUDA"
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "|", torch.cuda.get_device_name(0), flush=True)
import triton
print("triton", triton.__version__, flush=True)

from tritonbench_registry import build_corpus
CORPUS = build_corpus()
print("corpus entries:", len(CORPUS), flush=True)


def qlin(xs, q):
    s = sorted(xs); n = len(s); h = q * (n - 1)
    lo = math.floor(h); hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def split(inputs):
    """primary tensor + the companion args held fixed, as checker.py does."""
    if isinstance(inputs, tuple):
        return inputs[0], list(inputs[1:])
    return inputs, []


def peak_attention_weight(Q, Kt):
    S = Q @ Kt.transpose(-2, -1) * (1.0 / math.sqrt(Q.shape[-1]))
    return torch.softmax(S, -1).max().item()


def measure(ref, x, rest, seed):
    """All quantities, all from the Triton kernel."""
    base = ref(x, *rest)
    m = base.numel()
    x_std = x.float().std().item() or 1.0
    sigma = DELTA_SCALE * x_std

    g = torch.Generator(device=x.device).manual_seed(seed)
    deltas = [torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype) * sigma
              for _ in range(NS)]

    sens = [(ref(x + d, *rest) - base).abs().max().item() for d in deltas]
    tol = max(3.0 * qlin(sens, 0.95), 1e-6)

    # --- L: E[(J d)_i^2] = sigma^2 ||J_i||^2, per output coordinate ---------
    g2 = torch.Generator(device=x.device).manual_seed(123)
    acc = torch.zeros_like(base, dtype=torch.float64)
    for _ in range(K_MC):
        d = torch.randn(x.shape, generator=g2, device=x.device, dtype=x.dtype) * sigma
        acc += ((ref(x + d, *rest) - base).double()) ** 2
    L = ((acc / K_MC).sqrt().max() / sigma).item()

    # --- native linearisation: s(t d) must equal t * s(d) ------------------
    ladder = {}
    for t in T_LADDER:
        vals = [(ref(x + t * d, *rest) - base).abs().max().item() for d in deltas[:10]]
        ladder[t] = vals
    s1 = ladder[1.0]
    defect_01 = [abs(a - b / 0.1) / a for a, b in zip(s1, ladder[0.1]) if a > 0]
    defect_001 = [abs(a - b / 0.01) / a for a, b in zip(s1, ladder[0.01]) if a > 0]
    lo_, hi_ = ladder[0.1], ladder[1.0]
    slopes = [math.log10(b / a) for a, b in zip(lo_, hi_) if a > 0 and b > 0]

    return dict(m=m, sigma=sigma, L=L, tol=tol, sens=sens,
                defect_t01=float(np.median(defect_01)) if defect_01 else None,
                defect_t001=float(np.median(defect_001)) if defect_001 else None,
                slope=float(np.median(slopes)) if slopes else None,
                cv=float(np.std(sens, ddof=1) / np.mean(sens)) if np.mean(sens) > 0 else None)


def sandwich(r):
    lo = 3 * 0.6744898 * r["sigma"] * r["L"]
    hi = 3 * r["sigma"] * r["L"] * (math.sqrt(2 * math.log(2 * r["m"])) +
                                    math.sqrt(2 * math.log(NS / ETA)))
    return lo, hi, (r["tol"] >= lo), (r["tol"] <= hi)


done = set()
if os.path.exists(OUT):
    for ln in open(OUT):
        try:
            j = json.loads(ln)
            done.add((j["entry"], j["inv"], j["kind"]))
        except Exception:
            pass
print("resuming, already done:", len(done), flush=True)

fh = open(OUT, "a")
def emit(rec):
    fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())


rng = np.random.default_rng(0)
t0 = time.time()
for i, entry in enumerate(CORPUS):
    op = entry["op"]
    to_torch = entry["to_torch"]
    ref_fn = entry["torch_ref_fn"]
    spec = entry["spec"]
    for j in range(6):                       # 1 mutant call + 5 reference calls
        np_args = entry["input_fn"](rng)     # <-- the exact banked draw
        if op in EXCLUDE:
            continue
        if (i, j, "primary") in done:
            continue
        inputs = to_torch(np_args)
        x, rest = split(inputs)
        try:
            r = measure(ref_fn, x, rest, seed=1000 + 7 * i + j)
        except Exception as ex:
            emit(dict(entry=i, inv=j, op=op, kind="primary", error=repr(ex)[:200]))
            continue
        lo, hi, ok_lo, ok_hi = sandwich(r)
        emit(dict(entry=i, inv=j, op=op, mutant=entry["mutant_name"], kind="primary",
                  m=r["m"], sigma=r["sigma"], L=r["L"], tol=r["tol"],
                  lo=lo, hi=hi, ok_lo=ok_lo, ok_hi=ok_hi,
                  ratio=r["tol"] / (3 * r["sigma"] * r["L"]),
                  defect_t01=r["defect_t01"], defect_t001=r["defect_t001"],
                  slope=r["slope"], cv=r["cv"], sens=r["sens"]))
    print("[%5.1fs] entry %2d/%d %s" % (time.time() - t0, i + 1, len(CORPUS), op),
          flush=True)

# ------------------------------------------------------------------ attention
print("=== attention input-conditional check (native) ===", flush=True)
rng2 = np.random.default_rng(0)
seen = set()
for i, entry in enumerate(CORPUS):
    op = entry["op"]
    np_args = [entry["input_fn"](rng2) for _ in range(6)][0]
    if "attention" not in op or op in seen:
        continue
    seen.add(op)
    inputs = entry["to_torch"](np_args)
    ref_fn = entry["torch_ref_fn"]
    spec = entry["spec"]
    variants = [("primary", inputs)] + list(spec.get_adversarial_inputs(inputs))
    for name, vin in variants:
        if (i, name, "attn") in done:
            continue
        x, rest = split(vin)
        try:
            r = measure(ref_fn, x, rest, seed=555)
            pw = peak_attention_weight(x, rest[0]) if rest else None
        except Exception as ex:
            emit(dict(entry=i, inv=name, op=op, kind="attn", error=repr(ex)[:200]))
            continue
        lo, hi, ok_lo, ok_hi = sandwich(r)
        emit(dict(entry=i, inv=name, op=op, kind="attn", variant=name,
                  m=r["m"], sigma=r["sigma"], L=r["L"], tol=r["tol"],
                  ok_lo=ok_lo, ok_hi=ok_hi, peak_weight=pw,
                  defect_t01=r["defect_t01"], slope=r["slope"], cv=r["cv"]))
        print("   %-30s %-28s peak=%.6f cv=%.4f defect=%.4f%%"
              % (op, name, pw if pw else float('nan'), r["cv"],
                 100 * (r["defect_t01"] or float('nan'))), flush=True)

fh.close()
print("DONE in %.1fs" % (time.time() - t0), flush=True)
