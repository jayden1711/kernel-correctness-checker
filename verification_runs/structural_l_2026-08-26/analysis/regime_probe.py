"""
Does the closed form still predict `adaptive_tol` on the inputs the checker
ACTUALLY spends its time on?

The generalization round validated the closed forms on the corpus's ORDINARY
random inputs and said so explicitly:

    "Attention's closed form is verified only on the corpus's ordinary inputs.
     The saturating and fp-floor adversarial inputs [...] are outside the linear
     regime, so a Jacobian-based prediction is not expected to hold there and
     was not tested."

From the banked n=20 arm, 634 of the 844 perturbation-routed calls -- 76.5% of
the probing time -- are those adversarial variants. So "not tested" covers
three quarters of where the saving would have to come from. This probe tests
it, on CPU, using each spec's OWN `get_adversarial_inputs` so the variants are
the ones the checker really generates rather than a stand-in.

Reported per (operator, variant): tol_struct / tol_mc. 1.0 means the closed
form reproduces the probe. Anything far from 1.0 on a variant means enabling
KCC_STRUCTURAL_L moves the pass/fail band for that call by that factor.

CPU, torch references, seeded. Absolute times are not the point here -- the
ratio is.
"""
import importlib, math, os, sys, json
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn.functional as F
from verification.layer2_numeric_oracle import structural_l as S

torch.manual_seed(0)
NS = 20
NSIM = 3000
DELTA = 1e-3
SCALE = 3.0
Q = 0.95

# (spec module, torch reference, inputs factory).  Inputs match the spec's own
# make_inputs contract; the reference is the torch semantics the corpus uses.
def rms(x, g):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * g

CASES = {
 "sum_reduction":  (lambda x: x.sum(-1),               lambda: torch.randn(64, 128)),
 "mean_reduction": (lambda x: x.mean(-1),              lambda: torch.randn(64, 128)),
 "max_reduction":  (lambda x: x.max(-1).values,        lambda: torch.randn(64, 128)),
 "min_reduction":  (lambda x: x.min(-1).values,        lambda: torch.randn(64, 128)),
 "gelu":           (lambda x: F.gelu(x),               lambda: torch.randn(64, 128)),
 "swish":          (lambda x: F.silu(x),               lambda: torch.randn(64, 128)),
 "softmax":        (lambda x: torch.softmax(x, -1),    lambda: torch.randn(64, 128)),
 "log_softmax":    (lambda x: torch.log_softmax(x, -1),lambda: torch.randn(64, 128)),
 "l1norm":         (lambda x: x / x.abs().sum(-1, keepdim=True), lambda: torch.randn(64, 128)),
 "l2norm":         (lambda x: x / x.norm(dim=-1, keepdim=True),  lambda: torch.randn(64, 128)),
 "frobenius_norm": (lambda x: x / x.norm(),            lambda: torch.randn(64, 128)),
}


def mc_tol(ref, x):
    xs = x.float().std().item() or 1.0
    base = ref(x)
    sens = []
    for _ in range(NS):
        d = torch.randn_like(x) * DELTA * xs
        sens.append((ref(x + d) - base).abs().max())
    st = torch.stack(sens).to(device="cpu", dtype=torch.float32)
    return max(SCALE * torch.quantile(st, Q).item(), 1e-6)


def load_spec(op):
    """Each spec module exposes get_spec(); that is the only supported entry
    point. An earlier version of this probe duck-typed for the first object
    with get_adversarial_inputs and picked up the abstract base, which raised
    NotImplementedError and silently reported zero adversarial variants --
    i.e. it would have concluded the untested regime was untestable."""
    mod = importlib.import_module(f"verification.specs.{op}")
    return mod.get_spec()


def main():
    rows = []
    print(f"{'operator':<18}{'variant':<30}{'tol_mc':>12}{'tol_struct':>13}{'ratio':>9}")
    for op, (ref, mk) in CASES.items():
        spec = load_spec(op)
        x0 = mk()
        variants = [("<ordinary>", x0)]
        if spec is not None:
            try:
                for nm, inp in spec.get_adversarial_inputs(x0):
                    t = inp[0] if isinstance(inp, tuple) else inp
                    variants.append((nm, t))
            except Exception as e:
                print(f"  ! {op}: adversarial gen failed: {type(e).__name__}: {e}")
        for nm, x in variants:
            if not torch.is_floating_point(x) or x.numel() < 2:
                continue
            try:
                tm = mc_tol(ref, x)
                ts = S.structural_adaptive_tol(op, x, (), NS, Q, SCALE, DELTA,
                                               nsim=NSIM)
            except Exception as e:
                print(f"{op:<18}{nm:<30}  ERROR {type(e).__name__}: {e}")
                continue
            if ts is None:
                print(f"{op:<18}{nm:<30}{tm:>12.3e}{'declined':>13}{'--':>9}")
                continue
            r = ts / tm
            flag = "" if 0.9 <= r <= 1.1 else ("  <<<" if (r < 0.5 or r > 2) else "  <")
            rows.append(dict(op=op, variant=nm, tol_mc=tm, tol_struct=ts, ratio=r,
                             ordinary=(nm == "<ordinary>")))
            print(f"{op:<18}{nm:<30}{tm:>12.3e}{ts:>13.3e}{r:>9.3f}{flag}")

    print()
    ordn = [r["ratio"] for r in rows if r["ordinary"]]
    advn = [r["ratio"] for r in rows if not r["ordinary"]]

    def summ(nm, v):
        if not v:
            print(f"  {nm}: none"); return
        v = sorted(v)
        med = v[len(v) // 2]
        w10 = sum(1 for r in v if 0.9 <= r <= 1.1)
        w2x = sum(1 for r in v if 0.5 <= r <= 2.0)
        print(f"  {nm:<34} n={len(v):>3}  min {v[0]:8.3f}  med {med:7.3f}  "
              f"max {v[-1]:10.3f}   within +-10%: {w10}/{len(v)}   within 2x: {w2x}/{len(v)}")

    print("tol_struct / tol_mc:")
    summ("ORDINARY inputs (validated regime)", ordn)
    summ("ADVERSARIAL inputs (untested)", advn)
    json.dump(rows, open("verification_runs/structural_l_2026-08-26/analysis/regime.json", "w"), indent=1)


if __name__ == "__main__":
    main()
