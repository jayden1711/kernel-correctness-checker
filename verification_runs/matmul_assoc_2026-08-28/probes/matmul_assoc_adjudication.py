"""
Adjudication of the LAST 2 reference-suspect records: matmul
`scalar_associativity` failures 50fb4e31 ((64,256)@(256,32)) and eab75b02
((65,65)@(65,65)), both plain randn scale-1 inputs, recorded 2026-07-23.

NORM_ADJUDICATION standard, adapted to a METAMORPHIC check (the check
compares the kernel against itself under a scale transform, so "ideal
math" has two arms):

  faithful  fp32 emulation of the reference kernel's own tile order
            (BLOCK_K = 32 chunked accumulation) through the RUN-ERA check
            (git HEAD: atol=1e-4, torch.allclose default rtol=1e-5) --
            must FAIL to validate the record;
  correct-implementation contrast
            torch.mm fp32 (a different, independently correct fp32
            matmul) through the same run-era check -- if it fails too,
            the failure is a property of fp32 matmul itself, not of this
            kernel;
  ideal     float64 through the same check -- if it passes, the failure
            is fp precision, not algebra.

  check-domain false alarm <=> faithful fails AND torch.mm-fp32 fails
                               AND fp64 passes, unanimously over seeds.

THE (65,65) HYPOTHESIS is tested directly: if the failure were the
boundary-tile mechanism (the flash/layernorm bug family), it would (a)
not appear at pow2 shapes with no partial tiles and (b) show up as
kernel-vs-torch.mm divergence concentrated at 65-boundary shapes. Both
are measured at control shapes {(64,64), (65,65), (64,256)x(256,32),
(128,128), (96,96)}.

Also derives the run-era check's validity domain (item-3 style) and
verifies the shipped (working-tree, atol=2e-3/rtol=1e-3) check passes
the same faithful arithmetic 10/10.

No recorded seeds (randn fills) -> 10-seed unanimity required.

Run:  .venv/bin/python matmul_assoc_adjudication.py
"""
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from verification.layer3_properties.matmul_properties import (
    check_scalar_associativity as check_current)

BLOCK_K = 32
C = 100.0


def kernel_f32(A, B):
    """The reference kernel's arithmetic: fp32, K accumulated in BLOCK_K
    chunks (masked boundary loads contribute exact zeros, so the chunked
    sum IS the boundary-tile behaviour)."""
    A, B = A.float(), B.float()
    M, K = A.shape
    acc = torch.zeros(M, B.shape[1], dtype=torch.float32)
    for k in range(0, K, BLOCK_K):
        acc = acc + A[:, k:k + BLOCK_K] @ B[k:k + BLOCK_K, :]
    return acc


def check_run_era(kernel_fn, A, B, atol=1e-4):
    lhs = kernel_fn(C * A, B).float()
    rhs = (C * kernel_fn(A, B)).float()
    ok = torch.allclose(lhs, rhs, atol=atol)          # rtol = default 1e-5
    return ok, float((lhs - rhs).abs().max())


def main():
    records = [("50fb4e31", (64, 256), (256, 32)),
               ("eab75b02", (65, 65), (65, 65))]

    print("== adjudication (10 seeds each; run-era check atol=1e-4)")
    for pid, sa, sb in records:
        f_fail = t_fail = i_fail = cur_fail = 0
        errs_f, errs_t, errs_i = [], [], []
        near_zero = []
        alg_fail = 0
        for seed in range(10):
            torch.manual_seed(seed)
            A = torch.randn(*sa)
            B = torch.randn(*sb)
            ok_f, e_f = check_run_era(kernel_f32, A, B)
            ok_t, e_t = check_run_era(lambda a, b: a.float() @ b.float(), A, B)
            # "fp64 ideal" the way the check actually feeds a kernel:
            # c*A is rounded to fp32 BEFORE the kernel sees it, so even
            # exact accumulation inherits that input-rounding random walk
            ok_i, e_i = check_run_era(lambda a, b: a.double() @ b.double(), A, B)
            # pure-algebra control: scale in fp64 too -- isolates whether
            # the IDENTITY itself holds (it must, to ~1e-11)
            l_alg = (C * A.double()) @ B.double()
            r_alg = C * (A.double() @ B.double())
            alg_fail += not torch.allclose(l_alg, r_alg, atol=1e-4)
            ok_cur, _ = check_current(kernel_f32, A, B)
            f_fail += not ok_f
            t_fail += not ok_t
            i_fail += not ok_i
            cur_fail += not ok_cur
            errs_f.append(e_f)
            errs_t.append(e_t)
            errs_i.append(e_i)
            # where does the worst element live relative to output scale?
            lhs = kernel_f32(C * A, B)
            rhs = C * kernel_f32(A, B)
            j = (lhs - rhs).abs().argmax()
            near_zero.append(abs(float(rhs.flatten()[j])) /
                             float(rhs.abs().max()))
        fm = lambda v: f"[{min(v):.2e}, {max(v):.2e}]"
        print(f"  {pid} {sa}x{sb}:")
        print(f"    faithful-kernel fp32: fails {f_fail}/10, max_err {fm(errs_f)}"
              f"  ({min(errs_f)/1e-4:.0f}-{max(errs_f)/1e-4:.0f}x atol)")
        print(f"    torch.mm fp32       : fails {t_fail}/10, max_err {fm(errs_t)}")
        print(f"    fp64 accum (fp32-rounded c*A input, as the check feeds "
              f"kernels): fails {i_fail}/10, max_err {fm(errs_i)}")
        print(f"    fp64 pure algebra (exact scaling): fails {alg_fail}/10")
        print(f"    SHIPPED loosened chk: fails {cur_fail}/10")
        print(f"    worst element |y|/max|y|: median "
              f"{sorted(near_zero)[5]:.4f} (cancellation lives on near-zero "
              f"elements)")
        verdict = ("CHECK-DOMAIN FALSE ALARM"
                   if f_fail == 10 and t_fail == 10 and alg_fail == 0
                   else "NOT CLEAN -- see numbers")
        print(f"    -> {verdict}  (criterion: faithful AND independent "
              f"correct fp32 both fail unanimously; the algebraic identity "
              f"itself holds in fp64)")

    # ---- the (65,65) boundary-tile hypothesis --------------------------
    print("\n== (65,65) hypothesis: control shapes, run-era check on the "
          "faithful kernel + kernel-vs-torch.mm divergence")
    shapes = [((64, 64), (64, 64)), ((65, 65), (65, 65)),
              ((64, 256), (256, 32)), ((128, 128), (128, 128)),
              ((96, 96), (96, 96))]
    for sa, sb in shapes:
        fails = 0
        errs, kt = [], []
        for seed in range(10):
            torch.manual_seed(100 + seed)
            A, B = torch.randn(*sa), torch.randn(*sb)
            ok, e = check_run_era(kernel_f32, A, B)
            fails += not ok
            errs.append(e)
            kt.append(float((kernel_f32(A, B) - A @ B).abs().max()))
        pow2 = all((x & (x - 1)) == 0 for x in (*sa, *sb))
        print(f"  {str(sa) + 'x' + str(sb):22s} pow2={pow2!s:5s} "
              f"run-era FP {fails}/10, max_err [{min(errs):.2e}, "
              f"{max(errs):.2e}], kernel-vs-torch.mm max diff "
              f"{max(kt):.2e}")

    # ---- validity-domain constant --------------------------------------
    print("\n== validity domain: fp32 rescale-noise E vs K "
          "(median max_err over 10 seeds, unscaled units err/c)")
    for K in (32, 64, 128, 256, 512, 1024, 2048):
        errs = []
        cur_fp = 0
        for seed in range(10):
            torch.manual_seed(seed)
            A, B = torch.randn(64, K), torch.randn(K, 64)
            _, e = check_run_era(kernel_f32, A, B)
            errs.append(e / C)
            ok_cur, _ = check_current(kernel_f32, A, B)
            cur_fp += not ok_cur
        med = sorted(errs)[5]
        print(f"  K={K:5d}: E/c = {med:.2e}  (E/c / (u*sqrt(K)) = "
              f"{med / (2**-24 * math.sqrt(K)):.1f})   SHIPPED check FP "
              f"{cur_fp}/10")


if __name__ == "__main__":
    main()
