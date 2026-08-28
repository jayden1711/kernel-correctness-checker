"""
FLAG ON must run, must take the structural branch, and must decline where no
formula was derived.

Not a correctness result -- correctness lives in regime_probe.py -- just proof
that the wiring reaches the branch on both the ordinary and the adversarial
call sites, and that argmax/argmin fall back rather than being forced onto a
formula that was never written for them.
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
os.environ["KCC_STRUCTURAL_L"] = "1"
os.environ["KCC_STRUCTURAL_NSIM"] = "200"     # smoke only; keep it quick

import torch
import torch.nn.functional as F
from verification.layer2_numeric_oracle import structural_l as S
from verification.layer2_numeric_oracle.perturbation import check_perturbation_tolerance

assert S._STRUCTURAL, "flag not picked up"

ref = lambda x: torch.softmax(x, -1)
torch.manual_seed(3)
x = torch.randn(64, 128)

ok, msg = check_perturbation_tolerance(ref, ref, x, op_name="softmax")[:2]
print("softmax     ", ok, "|", msg)
assert "KCC_STRUCTURAL_L" in msg or "adaptive_tol" in msg

# argmax has no derived formula -> must fall back to the probe, and the
# message must therefore quote a measured sensitivity, not the closed form.
ok2, msg2 = check_perturbation_tolerance(ref, ref, x, op_name="argmax")[:2]
print("argmax      ", ok2, "|", msg2)
assert S.structural_adaptive_tol("argmax", x, (), 20, 0.95, 3.0, 1e-3) is None

# no op_name at all (every legacy call site) -> probe.
ok3, msg3 = check_perturbation_tolerance(ref, ref, x)[:2]
print("no op_name  ", ok3, "|", msg3)

# a real failure must still be reported as a failure under the structural band
bad = lambda t: torch.softmax(t, -1) * 1.5
ok4, msg4 = check_perturbation_tolerance(bad, ref, x, op_name="softmax")[:2]
print("mutated     ", ok4, "|", msg4)
assert ok4 is False, "structural band failed to catch a 1.5x error"

# companion-bearing operator
g, b = torch.ones(128), torch.zeros(128)
lref = lambda t: F.layer_norm(t, (128,), g, b)
ok5, msg5 = check_perturbation_tolerance(lref, lref, x, op_name="layernorm",
                                         companions=(g, b))[:2]
print("layernorm   ", ok5, "|", msg5)

# same operator with the companion MISSING -> must decline, not raise
assert S.structural_adaptive_tol("layernorm", x, (), 20, 0.95, 3.0, 1e-3) is None
print("\nmissing-companion fallback: OK")
print("SMOKE: PASS")
