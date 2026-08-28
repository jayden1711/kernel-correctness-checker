"""
Native diagnosis of the attention adversarial variants.

Three competing explanations for a flat s(t):
  (a) fp32 QUANTISATION FLOOR -- the output has huge magnitude (V[-1]=1e4), so
      one ulp there swamps the perturbation response.
  (b) KERNEL NONDETERMINISM   -- ref(x) != ref(x); a fixed noise floor.
  (c) GENUINE SATURATION      -- softmax has collapsed to a hard select.
They are distinguished by measuring, natively: the output magnitude and its
ulp, the kernel's own repeat-to-repeat spread, and the s(t) ladder.

Adversarial inputs are SEEDED here (spec _make_qkv uses bare torch.randn), so
this is reproducible, unlike the corpus run.
"""
import json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/content")
sys.path.insert(0, "/content/benchmarks/autokernel/files")

OUT = "/content/attn_native.jsonl"
NS = 40
DELTA_SCALE = 1e-3
T_LADDER = [0.01, 0.1, 1.0, 10.0]
REPEATS = 12
N_SEEDS = 5          # independent draws of the adversarial input

from tritonbench_registry import build_corpus
CORPUS = build_corpus()
print(torch.cuda.get_device_name(0), flush=True)


def qlin(xs, q):
    s = sorted(xs); n = len(s); h = q * (n - 1)
    lo = math.floor(h); hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def split(inputs):
    if isinstance(inputs, tuple):
        return inputs[0], list(inputs[1:])
    return inputs, []


fh = open(OUT, "w")
def emit(r):
    fh.write(json.dumps(r) + "\n"); fh.flush(); os.fsync(fh.fileno())


seen = set()
rng = np.random.default_rng(0)
for i, entry in enumerate(CORPUS):
    op = entry["op"]
    np_args = entry["input_fn"](rng)
    if "attention" not in op or op in seen:
        continue
    seen.add(op)
    ref = entry["torch_ref_fn"]
    spec = entry["spec"]

    for sd in range(N_SEEDS):
        torch.manual_seed(4000 + sd)
        inputs = entry["to_torch"](np_args)
        variants = [("primary", inputs)] + list(spec.get_adversarial_inputs(inputs))
        for name, vin in variants:
            x, rest = split(vin)
            base = ref(x, *rest)

            finite = bool(torch.isfinite(base).all().item())
            omax = base.abs().max().item()
            ulp = float(np.spacing(np.float32(omax))) if omax > 0 else 0.0

            # (b) kernel determinism floor
            reps = [ref(x, *rest) for _ in range(REPEATS)]
            det = max((r - base).abs().max().item() for r in reps)

            # (c) saturation predictor
            S = x @ rest[0].transpose(-2, -1) * (1.0 / math.sqrt(x.shape[-1]))
            pw = torch.softmax(S, -1).max().item()

            xs = x.float().std().item() or 1.0
            sigma = DELTA_SCALE * xs
            g = torch.Generator(device=x.device).manual_seed(777 + sd)
            deltas = [torch.randn(x.shape, generator=g, device=x.device,
                                  dtype=x.dtype) * sigma for _ in range(NS)]
            sens = [(ref(x + d, *rest) - base).abs().max().item() for d in deltas]
            ladder = {}
            for t in T_LADDER:
                ladder[t] = [(ref(x + t * d, *rest) - base).abs().max().item()
                             for d in deltas[:10]]
            s1 = ladder[1.0]
            defect = [abs(a - b / 0.1) / a for a, b in zip(s1, ladder[0.1]) if a > 0]
            mean = float(np.mean(sens))
            emit(dict(op=op, variant=name, seed=sd, finite=finite,
                      out_max=omax, ulp=ulp, det_floor=det, peak_weight=pw,
                      sigma=sigma, m=base.numel(),
                      cv=float(np.std(sens, ddof=1) / mean) if mean > 0 else None,
                      s_med=float(np.median(sens)),
                      tol=max(3 * qlin(sens, 0.95), 1e-6),
                      ladder={str(t): float(np.median(v)) for t, v in ladder.items()},
                      defect=float(np.median(defect)) if defect else None,
                      sens_over_ulp=(float(np.median(sens)) / ulp) if ulp > 0 else None,
                      sens_over_det=(float(np.median(sens)) / det) if det > 0 else None))
        print("  seed %d done for %s" % (sd, op), flush=True)

fh.close()
print("DONE", flush=True)
