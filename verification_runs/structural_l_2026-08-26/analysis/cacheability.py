"""
Can M3 be CACHED? -- the one thing that could rescue the structural path.

The cost probe measures M3 at 1128x the Monte-Carlo path. But M3's only input
is the NORMALISED profile `w = ||J_i|| / L`, and if `w`'s shape is stable
across the random inputs a given (operator, shape) sees, then `y` is a
per-(op, shape) CONSTANT and the simulation is paid once per corpus run rather
than once per call. That would take the structural path from 1128x slower to
effectively free, and it is the strongest form of the proposal.

For the 9 shape-only operators this is true by construction -- `w` is all-ones
or all-(sqrt(W)/W), independent of the input. The open question is the other
18, where `w` is a function of the actual tensor.

This measures it: `y` across 8 independent random draws per operator, same
shape each time. A tight spread means cacheable; a wide one means the
simulation has to be re-run per call and the 1128x stands.

Dismissing the cached variant without measuring it would be the same error
the previous round made with its `n_samples` projection -- reasoning about a
number instead of taking it.
"""
import os, sys, statistics as st
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn.functional as F
from verification.layer2_numeric_oracle import structural_l as S

NS, NSIM, DRAWS = 20, 3000, 8

CASES = {
    "gelu":           (lambda: torch.randn(64, 128), ()),
    "swish":          (lambda: torch.randn(64, 128), ()),
    "softmax":        (lambda: torch.randn(64, 128), ()),
    "log_softmax":    (lambda: torch.randn(64, 128), ()),
    "l1norm":         (lambda: torch.randn(64, 128), ()),
    "l2norm":         (lambda: torch.randn(64, 128), ()),
    "frobenius_norm": (lambda: torch.randn(64, 128), ()),
    "layernorm":      (lambda: torch.randn(64, 128), (torch.ones(128), torch.zeros(128))),
    "rmsnorm":        (lambda: torch.randn(64, 128), (torch.ones(128),)),
    "matmul":         (lambda: torch.randn(16, 64), (torch.randn(64, 64),)),
    "sum_reduction":  (lambda: torch.randn(64, 128), ()),
    "max_reduction":  (lambda: torch.randn(64, 128), ()),
}

print(f"{'operator':<18}{'y min':>10}{'y med':>10}{'y max':>10}{'spread':>9}"
      f"{'CV %':>8}   cacheable?")
for op, (mk, rest) in CASES.items():
    ys = []
    for d in range(DRAWS):
        torch.manual_seed(100 + d)
        x = mk()
        rn = S.row_norms(op, x, list(rest))
        # seed pinned so the SIMULATION noise is identical across draws and
        # any spread seen is genuinely the profile moving, not the simulator.
        ys.append(S.y_profile(rn.float(), NS, nsim=NSIM, seed=0))
    lo, hi, med = min(ys), max(ys), st.median(ys)
    cv = st.pstdev(ys) / med * 100
    verdict = "YES (<1%)" if cv < 1 else ("marginal" if cv < 3 else "NO")
    print(f"{op:<18}{lo:>10.4f}{med:>10.4f}{hi:>10.4f}{hi/lo:>9.4f}{cv:>8.2f}   {verdict}")
