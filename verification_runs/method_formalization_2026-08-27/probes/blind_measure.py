"""
BLIND GENERALIZATION TEST, stage 2 (GPU): measure the shipped CUDA kernel.

Runs on the Colab VM. Loads the banked inputs and the banked 40 deltas per
config (stage 1 drew them; nothing is drawn here -- the GPU stage has NO
RNG), applies the checker's exact measurement protocol to
`torch.logcumsumexp` on cuda float32, and writes one JSON line per config:

    s_meas[k] = || f32(x + d_k) - f32(x) ||_inf        (40 values)
    y_meas    = q95_40(s_meas) / (sigma * L_pred)      (L from stage 1)

The kernel under test is ATen's CUDA logcumsumexp -- an implementation this
project did not write and has never measured. Comparison happens offline in
blind_compare.py against the pre-registered predictions.
"""

import json

import numpy as np
import torch

NPZ = "/content/blind_inputs.npz"
PRED = "/content/blind_predictions.json"
OUT = "/content/blind_gpu.jsonl"

assert torch.cuda.is_available()
bank = np.load(NPZ)
preds = json.load(open(PRED))["preds"]

with open(OUT, "w") as f:
    for p in preds:
        key = p["key"]
        x = torch.from_numpy(bank[key + "_x"]).cuda()
        deltas = torch.from_numpy(bank[key + "_d"]).cuda()
        base = torch.logcumsumexp(x, dim=1)
        s = []
        for k in range(deltas.shape[0]):
            out = torch.logcumsumexp(x + deltas[k], dim=1)
            s.append(float((out - base).abs().max()))
        f.write(json.dumps({"key": key, "s_meas": s,
                            "out_absmax": float(base.abs().max()),
                            "dtype": str(x.dtype),
                            "dev": torch.cuda.get_device_name(0)}) + "\n")
        print(key, "done", flush=True)
print("wrote", OUT)
