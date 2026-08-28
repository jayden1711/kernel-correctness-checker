"""
The attention probe found a failure mode the prior round missed: for some
inputs the measured "sensitivity" is only 2-3 ulp of the output, so it is
reporting float32 granularity rather than || J d ||_inf.

This checks the 27 IN-SCOPE operators' own corpus inputs against that mode,
natively, and also confirms the kernels are deterministic (so that s is a
function of the input at all).
"""
import json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, "/content")
sys.path.insert(0, "/content/benchmarks/autokernel/files")

OUT = "/content/floor_native.jsonl"
NS, REPEATS = 40, 12
DELTA_SCALE = 1e-3
EXCLUDE = {"argmax", "argmin"}

from tritonbench_registry import build_corpus
CORPUS = build_corpus()


def split(inputs):
    if isinstance(inputs, tuple):
        return inputs[0], list(inputs[1:])
    return inputs, []


fh = open(OUT, "w")
def emit(r):
    fh.write(json.dumps(r) + "\n"); fh.flush(); os.fsync(fh.fileno())


rng = np.random.default_rng(0)
for i, entry in enumerate(CORPUS):
    op = entry["op"]
    ref = entry["torch_ref_fn"]
    for j in range(6):
        np_args = entry["input_fn"](rng)
        if op in EXCLUDE:
            continue
        x, rest = split(entry["to_torch"](np_args))
        base = ref(x, *rest)
        omax = base.abs().max().item()
        ulp = float(np.spacing(np.float32(omax))) if omax > 0 else 0.0
        det = max((ref(x, *rest) - base).abs().max().item() for _ in range(REPEATS))
        xs = x.float().std().item() or 1.0
        sigma = DELTA_SCALE * xs
        g = torch.Generator(device=x.device).manual_seed(1000 + 7 * i + j)
        sens = [(ref(x + torch.randn(x.shape, generator=g, device=x.device,
                                     dtype=x.dtype) * sigma, *rest) - base
                 ).abs().max().item() for _ in range(NS)]
        smin = min(sens)
        emit(dict(op=op, entry=i, inv=j, finite=bool(torch.isfinite(base).all().item()),
                  out_max=omax, ulp=ulp, det_floor=det,
                  s_min=smin, s_med=float(np.median(sens)),
                  min_over_ulp=(smin / ulp) if ulp > 0 else None))
    print("entry %d %s" % (i, op), flush=True)
fh.close()
print("DONE", flush=True)
