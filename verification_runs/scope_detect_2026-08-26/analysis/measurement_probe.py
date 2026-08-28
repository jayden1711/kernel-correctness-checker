"""
Do the SIGNAL FUNCTIONS actually measure what the thresholds assume?

test_scope_detect.py replays banked (defect, s/ulp) pairs through classify(),
which settles the RULE. This settles the other half that can be done without a
GPU: whether `measure_defect` and `sulp_median`, run end to end against live
references, produce the values the rule expects for each mechanism.

CPU, torch references, NOT Triton kernels. So this validates the measurement
CODE. It does not and cannot reproduce the banked Triton numbers -- that is the
GPU step, and it is not attempted here.

Three constructed regimes, one per mechanism the detector claims to separate:

  in-scope     ordinary gelu / layernorm            -> expect silence
  saturation   softmax on logits x 1e3 (hard select) -> expect a large defect
  fp floor     output at magnitude 1e4 with a perturbation below its ulp
                                                    -> expect s/ulp of order 1
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
os.environ["KCC_SCOPE_DETECT"] = "1"

import torch
import torch.nn.functional as F
from verification.layer2_numeric_oracle import scope_detect as S
from verification.layer2_numeric_oracle.perturbation import check_perturbation_tolerance

torch.manual_seed(0)
DELTA, NS = 1e-3, 20


def run(name, ref, x, expect):
    torch.manual_seed(5)
    out = check_perturbation_tolerance(ref, ref, x, op_name=name)
    recs = out[2] if len(out) > 2 else []
    scope = [r for r in recs if r.get("kind") == "scope_divergence"]
    if not scope:
        got, d, u = "silent", None, None
    else:
        r = scope[0]
        got = ",".join(x["reason"] for x in r["reasons"] if x["severity"] != "advisory") or "advisory-only"
        d, u = r["defect_pct"], r["sulp_median"]
    ok = "OK " if got == expect else "MISMATCH"
    ds = f"{d:9.2f}" if d is not None else "        -"
    us = f"{u:12.2f}" if u is not None else "           -"
    print(f"{ok} {name:<28} defect%={ds}  s/ulp={us}   -> {got}")
    return got == expect


ok = True

# --- in scope -------------------------------------------------------------
ok &= run("gelu", lambda t: F.gelu(t), torch.randn(64, 128), "silent")
g, b = torch.ones(128), torch.zeros(128)
ok &= run("layernorm", lambda t: F.layer_norm(t, (128,), g, b),
          torch.randn(64, 128), "silent")
ok &= run("softmax", lambda t: torch.softmax(t, -1), torch.randn(64, 128), "silent")

# --- mechanism (i): saturation -------------------------------------------
# Logits scaled so softmax is a hard select: the response to a perturbation
# stops being proportional to it.
ok &= run("softmax_saturated", lambda t: torch.softmax(t * 1e3, -1),
          torch.randn(64, 128), "saturation")

# --- mechanism (ii): float32 quantisation floor ---------------------------
# Output pinned at magnitude ~1e4, where one float32 ulp is 9.77e-04. Same
# construction as the banked `last_tile_dropped` variant (V[-1] = 1e4).
#
# TWO SHAPES, because they exercise different halves of the pairing:
#
#   (a) response of a couple of ulp -- the banked shape (s/ulp 2-3). The
#       response is pinned to the quantisation step regardless of perturbation
#       size, so s(t) is constant and the defect blows up. BOTH screens fire,
#       which is the banked signature.
#   (b) response driven to exactly zero. `measure_defect` returns None here by
#       design -- |s(1) - s(0.1)/0.1| / s(1) is undefined at s(1) = 0, and
#       reporting 0.0 would read as "perfectly linear", the opposite of the
#       truth. So the DEFECT SCREEN GOES BLIND and only s/ulp fires.
#
# (b) is the concrete case that answers "why pair the two screens rather than
# use the defect alone": on its own the defect would have said nothing here.
ok &= run("fp_floor_few_ulp", lambda t: (1e4 + t * 2.0).float(),
          torch.randn(64, 128), "quantisation_floor,saturation")
ok &= run("fp_floor_zero_response", lambda t: (t * 1e-9 + 1e4).float(),
          torch.randn(64, 128), "quantisation_floor")

# --- structural exclusion -------------------------------------------------
ok &= run("argmax", lambda t: t.argmax(-1), torch.randn(64, 128),
          "structural_exclusion")

print()
print("MEASUREMENT PROBE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
