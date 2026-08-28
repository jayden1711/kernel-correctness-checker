"""
Validity-domain boundaries for the l1norm/l2norm/frobenius_norm Layer-3
property checks, derived then verified numerically.

Kernel eps placements (TritonBench/reference, eps = 1e-12, fp32):
    l1norm:          y = x / (S1 + eps),        S1 = sum|x|
    l2norm:          y = x / sqrt(S2 + eps),    S2 = sum x^2
    frobenius_norm:  y = x / (sqrt(S2) + eps)   (note: eps OUTSIDE the sqrt)

Exact-math deviations of the unit-norm checks (atol = 1e-3):
    unit_l1:    dev = eps/(S1+eps)          -> invalid iff S1 < eps(1-a)/a ~ 1.0e-9
    unit_l2:    dev = 1-sqrt(S2/(S2+eps))
                    ~ eps/(2 S2)            -> invalid iff S2 < eps/(2a) = 5.0e-10
                                               (||x||_2 < 2.24e-5)
    unit_frob:  dev = eps/(sqrt(S2)+eps)    -> invalid iff ||x||_F < eps(1-a)/a ~ 1.0e-9

positive_scale_invariance (c, atol=1e-3, rtol=1e-3): comparing f(x) with
f(cx) leaves a residual eps-shift; per element j the deviation is
out_j * E where
    l1:   E = (eps/S1)(1-1/c)
    l2:   E = (eps/(2 S2))(1-1/c^2)
    frob: E = (eps/sqrt(S2))(1-1/c)
and allclose fails iff E > rtol + atol/max|out| -- i.e. the boundary
depends on the input's peak-to-norm ratio, sharper for peaked inputs.

fp32 ABSORPTION: computed in fp32, S + eps == S exactly once
S >= 2^24 * eps = 1.678e-5 (for l2: S2 >= 1.678e-5, ||x||_2 >= 4.1e-3;
for frobenius: sqrt(S2) >= 1.678e-5), so above absorption the entire eps
term vanishes and the checks measure pure rounding (~1e-7). The eps
deviation is only VISIBLE in a window below absorption, and only BINDING
below the atol boundary. Between the two the check passes with margin.

This probe verifies each derived boundary by sweeping the input scale in
faithful fp32 emulation and locating the measured pass->fail crossing.

Run:  .venv/bin/python norm_domains.py
"""
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from verification.layer3_properties.norm_properties import (
    check_unit_l1_norm, check_unit_l2_norm, check_unit_frobenius_norm,
    check_positive_scale_invariance)

EPS = 1e-12
ATOL = 1e-3
N_ROWS, N_COLS = 64, 128
torch.manual_seed(3)


def l1_f32(x):
    x = x.float()
    s = x.abs().sum(dim=-1, keepdim=True)
    return x / (s + torch.tensor(EPS, dtype=torch.float32))


def l2_f32(x):
    x = x.float()
    s = (x * x).sum(dim=-1, keepdim=True)
    return x / torch.sqrt(s + torch.tensor(EPS, dtype=torch.float32))


def frob_f32(x):
    x = x.float()
    s = (x * x).sum()
    return x / (torch.sqrt(s) + torch.tensor(EPS, dtype=torch.float32))


def crossing(check, lo, hi, n=400, fill="randn"):
    """Largest scale at which the check FAILS (measured boundary)."""
    worst = None
    for i in range(n):
        sig = lo * (hi / lo) ** (i / (n - 1))
        torch.manual_seed(11)
        x = (torch.randn(N_ROWS, N_COLS) if fill == "randn"
             else torch.ones(N_ROWS, N_COLS)) * sig
        if not check(x):
            worst = sig
    return worst


def main():
    print("unit-norm checks: measured fail boundary vs derived (fill=randn,"
          f" shape ({N_ROWS},{N_COLS}))")
    # derived boundaries converted to input scale sigma:
    # E[S1] = n_cols*sigma*sqrt(2/pi); S2 = n_cols*sigma^2 (per row);
    # frobenius S2 over the whole matrix.
    n = N_COLS
    d_l1 = 1e-9 / (n * math.sqrt(2 / math.pi))
    d_l2 = math.sqrt(5.0e-10 / n)
    d_fr = 1e-9 / math.sqrt(N_ROWS * n)
    tests = [
        ("unit_l1", lambda x: check_unit_l1_norm(l1_f32(x))[0], d_l1),
        ("unit_l2", lambda x: check_unit_l2_norm(l2_f32(x))[0], d_l2),
        ("unit_frobenius", lambda x: check_unit_frobenius_norm(frob_f32(x))[0], d_fr),
    ]
    for name, chk, pred in tests:
        meas = crossing(chk, pred / 30, pred * 30)
        print(f"  {name:16s} predicted sigma* = {pred:.3e}   measured "
              f"largest-fail = {'none in window' if meas is None else f'{meas:.3e}'}"
              f"{'' if meas is None else f'   ratio {meas/pred:.2f}'}")

    print("\npositive_scale_invariance (c=4.2): measured vs derived, "
          "randn (dispersed) and one-hot-ish (peaked) fills")
    c = 4.2
    rtol = 1e-3

    def peaked(sig):
        x = torch.zeros(N_ROWS, N_COLS)
        x[:, 0] = sig
        x[:, 1] = sig * 1e-3   # keep norm dominated by one entry
        return x

    # boundary: E > rtol + atol/out_max
    # randn: out_max ~ 3.5/(n sqrt(2/pi)) ~ 0.034 -> thresh = rtol + atol/0.034
    for kind, fillfn, outmax in (
            ("randn", lambda s: torch.randn(N_ROWS, N_COLS) * s,
             3.5 / (n * math.sqrt(2 / math.pi))),
            ("peaked", peaked, 1.0)):
        thresh = rtol + ATOL / outmax
        d1 = EPS * (1 - 1 / c) / thresh / (n * math.sqrt(2 / math.pi)) \
            if kind == "randn" else EPS * (1 - 1 / c) / thresh
        d2 = (math.sqrt(EPS * (1 - 1 / c ** 2) / (2 * thresh) / n)
              if kind == "randn"
              else math.sqrt(EPS * (1 - 1 / c ** 2) / (2 * thresh)))
        for name, fn, pred in (("l1", l1_f32, d1), ("l2", l2_f32, d2)):
            def chk(x, fn=fn):
                return check_positive_scale_invariance(fn, x, scale=c)[0]
            worst = None
            for i in range(300):
                sig = pred / 30 * (900) ** (i / 299)
                torch.manual_seed(13)
                x = fillfn(sig)
                if not chk(x):
                    worst = sig
            print(f"  {name}/{kind:6s} predicted sigma* = {pred:.3e}   "
                  f"measured largest-fail = "
                  f"{'none in window' if worst is None else f'{worst:.3e}'}"
                  f"{'' if worst is None else f'   ratio {worst/pred:.2f}'}")

    print("\nfp32 absorption cliffs (deviation collapses to rounding above):")
    for name, fn, chk, sstar in (
            ("l1", l1_f32, check_unit_l1_norm, 2 ** 24 * EPS / (n * math.sqrt(2/math.pi))),
            ("l2", l2_f32, check_unit_l2_norm, math.sqrt(2 ** 24 * EPS / n)),
            ("frob", frob_f32, check_unit_frobenius_norm,
             2 ** 24 * EPS / math.sqrt(N_ROWS * n))):
        for mult in (0.3, 3.0):
            torch.manual_seed(17)
            x = torch.randn(N_ROWS, N_COLS) * sstar * mult
            dev = chk(fn(x))[1]
            side = "below" if mult < 1 else "above"
            print(f"  {name:5s} {side} absorption (sigma={sstar*mult:.2e}): {dev}")


if __name__ == "__main__":
    main()
