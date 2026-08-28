"""Layernorm padded-lane blast radius — the three emulation questions.

E1  The recorded `cross_shape` catch of `wrong_variance_estimate` fails only
    at (1000, 333) (max_err=0.0249 banked). The reference carries the
    padded-lane bug there; this mutant (one-pass variance over 0-padded
    loads) does NOT. Question: under a CORRECTED (masked) reference, does
    that sub-shape still fail — i.e. is the recorded catching check
    bug-manufactured?
E2  f322abe4 counterfactual (corrected reference + today's fixed checker):
    valid? any mutant caught WITH a naive gap (hit) or not?
E3  Magnitude: how wrong is the shipped reference at (1000, 333) on
    corpus-distribution inputs (kernel-faithful vs ideal)?

fp32 CPU emulation, 10 seeds everywhere (no banked GPU input draws exist for
cross_shape inputs — spec.make_inputs uses device RNG). Mutant emulations
follow the cheating sources exactly:
  ignore_gamma_beta   : unmasked diff (shares the ref bug), drops affine
  skip_mean_subtract  : unmasked diff variance, does NOT subtract mean in
                        the output numerator
  wrong_variance      : one-pass E[x^2]-E[x]^2 over 0-padded loads (no
                        padded-lane term), same /n denominator
"""

import math
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
sys.path.insert(0, ROOT)
from verification.layer3_properties import layernorm_properties as lnp  # noqa

EPS = 1e-5
CS_ATOL = CS_RTOL = 1e-4          # _check_cross_shape tolerances
NAIVE = dict(atol=1e-3, rtol=1e-2)


def npow2(n):
    return 1 << (n - 1).bit_length()


def _stats_buggy(x):
    n = x.shape[-1]
    pad = npow2(n) - n
    mean = x.sum(-1, keepdim=True) / n
    var = ((x - mean) ** 2).sum(-1, keepdim=True) / n + pad * mean ** 2 / n
    return mean, var


def ln_buggy(x, g, b):
    mean, var = _stats_buggy(x)
    return (x - mean) / torch.sqrt(var + EPS) * g + b


def ln_ideal(x, g, b):
    n = x.shape[-1]
    mean = x.sum(-1, keepdim=True) / n
    var = ((x - mean) ** 2).sum(-1, keepdim=True) / n
    return (x - mean) / torch.sqrt(var + EPS) * g + b


def mut_ignore_gb(x, g, b):
    mean, var = _stats_buggy(x)          # shares the unmasked diff
    return (x - mean) / torch.sqrt(var + EPS)


def mut_skip_mean(x, g, b):
    mean, var = _stats_buggy(x)          # shares the unmasked diff
    return x / torch.sqrt(var + EPS) * g + b


def mut_wrong_var(x, g, b):
    n = x.shape[-1]
    mean = x.sum(-1, keepdim=True) / n
    var = (x * x).sum(-1, keepdim=True) / n - mean ** 2   # 0-pads add nothing
    return (x - mean) / torch.sqrt(var + EPS) * g + b


def allclose_cs(a, b):
    return torch.allclose(a.float(), b.float(), atol=CS_ATOL, rtol=CS_RTOL)


def main():
    print("=== E1: cross_shape sub (1000,333), wrong_variance_estimate ===")
    fails_buggy = fails_ideal = 0
    errs_b, errs_i = [], []
    for sd in range(10):
        g_ = torch.Generator().manual_seed(100 + sd)
        x = torch.randn(1000, 333, generator=g_)
        gam, bet = torch.ones(333), torch.zeros(333)
        mut = mut_wrong_var(x, gam, bet)
        rb = ln_buggy(x, gam, bet)
        ri = ln_ideal(x, gam, bet)
        eb = (mut - rb).abs().max().item()
        ei = (mut - ri).abs().max().item()
        errs_b.append(eb)
        errs_i.append(ei)
        fails_buggy += not allclose_cs(mut, rb)
        fails_ideal += not allclose_cs(mut, ri)
    print(f"vs BUGGY ref:     fails {fails_buggy}/10, max_err "
          f"{min(errs_b):.4f}..{max(errs_b):.4f}  (banked: fail, 0.0249)")
    print(f"vs CORRECTED ref: fails {fails_ideal}/10, max_err "
          f"{min(errs_i):.2e}..{max(errs_i):.2e}  (atol {CS_ATOL})")
    # the four pow2 shapes must be pad-free hence verdict-identical
    for shape in [(512, 512), (256, 1024), (1, 512), (2048, 128)]:
        g_ = torch.Generator().manual_seed(7)
        x = torch.randn(*shape, generator=g_)
        gam, bet = torch.ones(shape[-1]), torch.zeros(shape[-1])
        assert torch.equal(ln_buggy(x, gam, bet), ln_ideal(x, gam, bet))
    print("pow2 shapes: buggy == corrected BITWISE (pad term structurally zero)")

    print("\n=== E3: reference error at (1000,333), corpus inputs ===")
    rels = []
    for sd in range(10):
        g_ = torch.Generator().manual_seed(200 + sd)
        x = torch.randn(1000, 333, generator=g_)
        gam, bet = torch.ones(333), torch.zeros(333)
        rb, ri = ln_buggy(x, gam, bet), ln_ideal(x, gam, bet)
        rels.append(((rb - ri).abs().max() / ri.abs().max()).item())
    print(f"kernel vs ideal max rel err: {min(rels):.4f}..{max(rels):.4f} "
          f"(silent, ~0 at pow2 widths)")

    print("\n=== E2: f322abe4 counterfactual (corrected ref, today's checker) ===")
    hits = 0
    for sd in range(10):
        g_ = torch.Generator().manual_seed(300 + sd)
        x = torch.randn(512, 333, generator=g_) + 10.0
        gam = torch.ones(333) * 2
        bet = torch.ones(333) * 3
        ref = ln_ideal(x, gam, bet)
        # reference validity under today's checker (fixed wrappers)
        o_id = ln_ideal(x, torch.ones(333), torch.zeros(333))
        checks = dict(
            zero_mean=lnp.check_zero_mean(o_id)[0],
            unit_variance=lnp.check_unit_variance(o_id)[0],
            scale_invariance=lnp.check_scale_invariance(
                lambda xi: ln_ideal(xi, gam, bet), x)[0],
            affine_correctness=lnp.check_affine_correctness(ln_ideal, x)[0],
            precision_coercion=lnp.check_precision_coercion(
                lambda xi: ln_ideal(xi, torch.ones(333), torch.zeros(333)), x)[0],
        )
        valid = all(checks.values())
        gap = False
        muts = {}
        for name, fn in [("ignore_gamma_beta", mut_ignore_gb),
                         ("skip_mean_subtract", mut_skip_mean),
                         ("wrong_variance", mut_wrong_var)]:
            mo = fn(x, gam, bet)
            naive = torch.allclose(mo.float(), ref.float(), **NAIVE)
            # caught = fails any of the same property checks (affine check is
            # the natural catcher for the first two; wrong_variance is
            # near-correct math and passes them)
            caught = not lnp.check_affine_correctness(fn, x)[0]
            muts[name] = (caught, naive)
            gap |= (caught and naive)
        hits += (valid and gap)
        if sd == 0:
            print(f"seed0: ref checks {checks}")
            print(f"seed0: mutants (caught, naive_pass): {muts}")
    print(f"counterfactual is_hit: {hits}/10 seeds  "
          f"(recorded: invalid-input non-hit)")


if __name__ == "__main__":
    main()
