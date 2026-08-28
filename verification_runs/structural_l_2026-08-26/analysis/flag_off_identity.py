"""
FLAG OFF MUST BE BIT-IDENTICAL.

The structural path was added by extracting the Monte-Carlo estimator into a
helper. Extraction is only behaviour-preserving if the RNG consumption and the
arithmetic order are unchanged -- and RNG order is exactly the defect class
this project has been bitten by before (SESSION_HANDOFF Section 7, and the
batched-draw note in perturbation.py itself).

So this compares the shipped path against a standalone transcription of the
PRE-CHANGE function body, from the same seed, and demands EXACT equality of
both the returned verdict and the tolerance embedded in its message.
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
assert os.environ.get("KCC_STRUCTURAL_L") != "1", "run this with the flag OFF"

import torch
import torch.nn.functional as F
from verification.layer2_numeric_oracle.perturbation import check_perturbation_tolerance


def legacy(candidate_fn, reference_fn, x, n_samples=20, quantile=0.95,
           scale=3.0, delta_scale=1e-3):
    """Verbatim pre-change body, unbatched path."""
    x = x.detach().clone()
    ref_base = reference_fn(x)
    x_std = x.float().std().item()
    if x_std == 0:
        x_std = 1.0
    deltas = [torch.randn_like(x) * delta_scale * x_std for _ in range(n_samples)]
    sens = []
    for d in deltas:
        sens.append((reference_fn(x + d) - ref_base).abs().max())
    st = torch.stack(sens).to(device="cpu", dtype=torch.float32)
    tol = scale * torch.quantile(st, quantile).item()
    tol = max(tol, 1e-6)
    out = candidate_fn(x)
    err = (out.float() - ref_base.float()).abs().max().item()
    return (err <= tol), tol, err


CASES = [
    ("softmax",    lambda x: torch.softmax(x, -1)),
    ("gelu",       lambda x: F.gelu(x)),
    ("layernorm",  lambda x: F.layer_norm(x, (128,))),
    ("l2norm",     lambda x: x / x.norm(dim=-1, keepdim=True)),
    ("sum_red",    lambda x: x.sum(-1)),
]

bad = 0
for nm, ref in CASES:
    for mutate in (False, True):
        cand = ref if not mutate else (lambda t, r=ref: r(t) * 1.05)
        torch.manual_seed(7)
        x = torch.randn(64, 128)
        torch.manual_seed(11)
        ok_new, msg = check_perturbation_tolerance(cand, ref, x)[:2]
        torch.manual_seed(11)
        ok_old, tol_old, err_old = legacy(cand, ref, x)
        tol_new = float(msg.split("adaptive_tol=")[1].split()[0].rstrip(".,"))
        same = (ok_new == ok_old) and (tol_new == round(tol_old, 6))
        tag = "OK " if same else "MISMATCH"
        if not same:
            bad += 1
        print(f"{tag} {nm:<12} mutated={int(mutate)}  verdict {ok_old}->{ok_new}"
              f"  tol {tol_old:.9f} vs {tol_new:.6f}")

print()
print("FLAG-OFF IDENTITY:", "PASS" if bad == 0 else f"FAIL ({bad})")
sys.exit(1 if bad else 0)
