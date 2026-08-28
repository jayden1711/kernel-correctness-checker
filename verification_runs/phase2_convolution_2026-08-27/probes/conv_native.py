"""GPU-NATIVE sandwich verification for the 27 Phase-1 operators.

Same methodology as verification_runs/adaptive_tol_theory_2026-08-25/probes/
gpu_native.py, constants unchanged (NS=40, K_MC=400, ETA=0.05,
DELTA_SCALE=1e-3, T_LADDER). Everything in the measurement path is the REAL
Triton kernel from phase1_kernels.py -- torch is used only to build inputs.

THE JACOBIAN, NATIVELY. torch.func.jvp cannot differentiate a @triton.jit
kernel. The native substitute is the directional derivative by definition,
evaluated with the kernel itself:
        s(t) = || f(x + t d) - f(x) ||_inf ,   linear in t iff s(t) = t s(1)
        defect = | s(1) - s(t)/t | / s(1)

ADDITIONS beyond the original probe, both needed by this round's questions:
  * `prof_spread` -- max/median of the NATIVE per-coordinate row-norm vector
    sqrt(E[(Jd)_i^2])/sigma. The CPU round measured this from the closed form
    and found sigmoid/tanh/swiglu at ~5e10/7e9/8e9 against a previous corpus
    max of 38.7. This measures the same quantity from live kernel execution.
  * `L_closed` / `y_M3` -- the closed form and the M3 prediction on the SAME
    input, so closed-vs-native L and the M3 re-fit both come out of one run.

Inputs come from each operator's own KernelSpec.make_inputs, seeded per
(op, invocation) so the run is reproducible; the Phase-1 operators are not in
the TritonBench corpus and so have no banked draw to replay.

Checkpointing per SESSION_HANDOFF: one JSONL line per invocation, flush+fsync,
resumes by reading its own output.
"""
import importlib, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/content")
OUT = "/content/conv_native.jsonl"
NS, K_MC, ETA = 40, 400, 0.05
DELTA_SCALE = 1e-3
T_LADDER = [0.01, 0.1, 1.0]
N_INV = 6

assert torch.cuda.is_available(), "no CUDA"
import triton
print("torch", torch.__version__, "| triton", triton.__version__,
      "|", torch.cuda.get_device_name(0), flush=True)

from conv_kernels import KERNELS
from verification.layer2_numeric_oracle.structural_l import row_norms, y_profile

OPS = list(KERNELS.keys())
print("operators:", len(OPS), flush=True)


def qlin(xs, q):
    s = sorted(xs); n = len(s); h = q * (n - 1)
    lo = math.floor(h); hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def split(inputs):
    if isinstance(inputs, tuple):
        return inputs[0], list(inputs[1:])
    return inputs, []


def measure(fn, x, rest, seed):
    """Every quantity below comes from the Triton kernel."""
    base = fn(x, *rest)
    m = base.numel()
    x_std = x.float().std().item()
    if not math.isfinite(x_std) or x_std == 0:
        x_std = 1.0
    sigma = DELTA_SCALE * x_std

    g = torch.Generator(device=x.device).manual_seed(seed)
    deltas = [torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype) * sigma
              for _ in range(NS)]
    sens = [(fn(x + d, *rest) - base).abs().max().item() for d in deltas]
    tol = max(3.0 * qlin(sens, 0.95), 1e-6)

    # L and the NATIVE row-norm profile: E[(J d)_i^2] = sigma^2 ||J_i||^2
    g2 = torch.Generator(device=x.device).manual_seed(123)
    acc = torch.zeros_like(base, dtype=torch.float64)
    for _ in range(K_MC):
        d = torch.randn(x.shape, generator=g2, device=x.device, dtype=x.dtype) * sigma
        acc += ((fn(x + d, *rest) - base).double()) ** 2
    prof = (acc / K_MC).sqrt().flatten() / sigma
    L = prof.max().item()
    pos = prof[prof > 0]
    prof_spread = (pos.max() / pos.median()).item() if pos.numel() else float("nan")
    prof_zero_frac = float((prof == 0).double().mean().item())

    # native linearisation ladder
    ladder = {}
    for t in T_LADDER:
        ladder[t] = [(fn(x + t * d, *rest) - base).abs().max().item() for d in deltas[:10]]
    s1 = ladder[1.0]
    d01 = [abs(a - b / 0.1) / a for a, b in zip(s1, ladder[0.1]) if a > 0]
    d001 = [abs(a - b / 0.01) / a for a, b in zip(s1, ladder[0.01]) if a > 0]
    slopes = [math.log10(b / a) for a, b in zip(ladder[0.1], ladder[1.0]) if a > 0 and b > 0]

    # float-floor diagnostic (GPU_NATIVE.md 3b / 4 mechanism ii)
    ulp = torch.finfo(base.dtype).eps * max(base.abs().max().item(), 1e-30)
    s_over_ulp = min(sens) / ulp if ulp > 0 else float("inf")

    # determinism over repeats
    det = 0.0
    for _ in range(4):
        det = max(det, (fn(x, *rest) - base).abs().max().item())

    return dict(m=m, sigma=sigma, L=L, tol=tol, sens=sens,
                prof_spread=prof_spread, prof_zero_frac=prof_zero_frac,
                defect_t01=float(np.median(d01)) if d01 else None,
                defect_t001=float(np.median(d001)) if d001 else None,
                slope=float(np.median(slopes)) if slopes else None,
                cv=float(np.std(sens, ddof=1) / np.mean(sens)) if np.mean(sens) > 0 else None,
                s_over_ulp=s_over_ulp, det_floor=det)


def sandwich(r):
    lo = 3 * 0.6744898 * r["sigma"] * r["L"]
    hi = 3 * r["sigma"] * r["L"] * (math.sqrt(2 * math.log(2 * r["m"])) +
                                    math.sqrt(2 * math.log(NS / ETA)))
    return lo, hi, (r["tol"] >= lo), (r["tol"] <= hi)


done = set()
if os.path.exists(OUT):
    for ln in open(OUT):
        try:
            j = json.loads(ln); done.add((j["op"], j["inv"]))
        except Exception:
            pass
print("resuming, already done:", len(done), flush=True)
fh = open(OUT, "a")
def emit(rec):
    fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())

t0 = time.time()
for oi, op in enumerate(OPS):
    spec = importlib.import_module(f"verification.specs.{op}").get_spec()
    for inv in range(N_INV):
        if (op, inv) in done:
            continue
        torch.manual_seed(9000 + 31 * oi + inv)
        # Conv sweeps ALL FIVE configs rather than repeating valid_shapes[0].
        # For Phase-1 operators the config list was the cross_shape edge sweep
        # and the right choice was one realistic shape; here every entry is a
        # realistic, DISTINCT hyperparameter regime (stride, padding, dilation,
        # groups, 1x1), and the closed form has to hold across all of them --
        # that variation is the thing under test, not noise.
        #
        # NOT valid_shapes[inv % len]. That list is the cross_shape sweep and
        # deliberately contains degenerate edge shapes -- (1,) and (4, 1) --
        # which are the right thing to test for shape generalisation and the
        # WRONG thing for a sensitivity study: a 1-element tensor makes
        # std() undefined (NaN, and NaN is truthy so `or 1.0` does not catch
        # it), and cumsum_exclusive on a width-1 row has an identically zero
        # output, hence J = 0 and L = 0. Both crashed the first pass.
        # The original probe drew 6 samples from ONE corpus input generator;
        # this mirrors that.
        inputs = spec.make_inputs(spec.valid_shapes[inv % len(spec.valid_shapes)],
                                  "cuda", torch.float32)
        x, rest = split(inputs)
        try:
        # Route through spec.run_candidate rather than calling KERNELS[op]
        # directly. depthwise_conv2d and pointwise_conv2d have SHORTER
        # signatures than the generic fn(x, *rest) call -- depthwise takes no
        # `groups` (it is implied by C_in) and pointwise takes none of the four
        # hyperparameters. Calling the kernel directly raised TypeError on all
        # 12 of their invocations. The spec already encodes each operator's
        # calling convention, and using it here is also what the checker does,
        # so the measured path and the shipped path stay identical.
            _fn = (lambda t, *_ignored, _s=spec, _k=KERNELS[op], _i=inputs:
                   _s.run_candidate(_k, (t,) + tuple(_i[1:])))
            r = measure(_fn, x, rest, seed=1000 + 7 * oi + inv)
        except Exception as ex:
            emit(dict(op=op, inv=inv, error=repr(ex)[:300])); continue

        # closed form + M3 prediction on the SAME input
        Lc, yM3 = None, None
        try:
            rn = row_norms(op, x, list(rest))
            if rn is not None and rn.numel() and torch.isfinite(rn).all():
                Lc = rn.max().item()
                yM3 = y_profile(rn.float(), NS)
        except Exception:
            pass

        # L == 0 means the kernel's output does not depend on x at all on this
        # input. Recorded as a defined outcome rather than crashing on the
        # ratio -- the sandwich is vacuous there, not violated.
        if not math.isfinite(r["L"]) or r["L"] <= 0:
            emit(dict(op=op, inv=inv, kind="degenerate", m=r["m"],
                      sigma=r["sigma"], L=r["L"], tol=r["tol"],
                      note="L<=0: output independent of primary on this input"))
            continue
        lo, hi, ok_lo, ok_hi = sandwich(r)
        emit(dict(op=op, inv=inv, kind="primary", m=r["m"], sigma=r["sigma"],
                  L=r["L"], L_closed=Lc, tol=r["tol"], lo=lo, hi=hi,
                  ok_lo=ok_lo, ok_hi=ok_hi,
                  ratio=r["tol"] / (3 * r["sigma"] * r["L"]),
                  y_M3=yM3,
                  prof_spread=r["prof_spread"], prof_zero_frac=r["prof_zero_frac"],
                  defect_t01=r["defect_t01"], defect_t001=r["defect_t001"],
                  slope=r["slope"], cv=r["cv"],
                  s_over_ulp=r["s_over_ulp"], det_floor=r["det_floor"],
                  sens=r["sens"]))
    print("[%6.1fs] %2d/%d %s" % (time.time() - t0, oi + 1, len(OPS), op), flush=True)

fh.close()
print("DONE in %.1fs" % (time.time() - t0), flush=True)
