"""
GPU-side probe for the layernorm padded-lane variance fix
(TritonBench/reference/layernorm.py, diff = tl.where(mask, row-mean, 0.0)).

Covers regression criteria (b) and (c) of
verification_runs/layernorm_mask_bug_2026-08-27/FINDINGS.md §4:

  (c) pow2 bitwise identity  -- the buggy (pre-fix) kernel, reproduced
      inline below, must be torch.equal to the fixed reference at every
      power-of-two width in use (the 4 pow2 spec shapes + the corpus
      (64,128)); and the two must DIFFER at (1000,333) or the fix is not
      live in the staged tree.
  (b) cross_shape margin at (1000,333) -- the wrong_variance_estimate
      mutant vs the FIXED reference, 10 seeds, spec.make_inputs, must pass
      atol 1e-4 with the ~200x margin the emulation predicted (5-7e-7);
      vs the BUGGY kernel it must fail at ~0.025 (the banked catch).
      Plus fixed-reference vs float64 ideal math as an anchor.

Run on the T4 with PYTHONPATH=/content. Prints LN-GPU-PROBE-OK on success.
"""
import sys
sys.path.insert(0, "/content")

import torch
import triton
import triton.language as tl

from TritonBench.reference.layernorm import layernorm as layernorm_fixed
from verification.specs.layernorm import LayernormSpec

assert torch.cuda.is_available(), "probe requires the GPU"


# ---- the PRE-FIX kernel, verbatim (unmasked diff) ------------------------

@triton.jit
def _buggy_kernel(output_ptr, input_ptr, gamma_ptr, beta_ptr,
                  input_row_stride, output_row_stride,
                  n_rows, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    mean = tl.sum(row, axis=0) / n_cols
    diff = row - mean                       # THE BUG: pad lanes -> -mean
    variance = tl.sum(diff * diff, axis=0) / n_cols
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)
    out = (row - mean) / tl.sqrt(variance + eps) * gamma + beta
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, out, mask=mask)


def layernorm_buggy(x, gamma, beta, eps=1e-5):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    _buggy_kernel[(n_rows,)](y, x, gamma, beta, x.stride(0), y.stride(0),
                             n_rows, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y


def ideal_math(x, gamma, beta, eps=1e-5):
    x64 = x.double()
    mean = x64.mean(dim=-1, keepdim=True)
    var = ((x64 - mean) ** 2).mean(dim=-1, keepdim=True)
    return ((x64 - mean) / torch.sqrt(var + eps)
            * gamma.double() + beta.double())


spec = LayernormSpec(name="layernorm")
failures = []

# ---- (c) pow2 bitwise identity + fix liveness ----------------------------
POW2 = [(512, 512), (256, 1024), (1, 512), (2048, 128), (64, 128)]
print("== (c) pow2 bitwise identity")
for shape in POW2:
    torch.manual_seed(hash(shape) % (2**31))
    x, gamma, beta = spec.make_inputs(shape, "cuda", torch.float32)
    # non-trivial companions: the identity must not depend on gamma=1/beta=0
    gamma = torch.randn_like(gamma)
    beta = torch.randn_like(beta)
    a = layernorm_fixed(x, gamma, beta)
    b = layernorm_buggy(x, gamma, beta)
    eq = torch.equal(a, b)
    print(f"  {shape}: torch.equal = {eq}")
    if not eq:
        failures.append(f"pow2 bitwise identity broken at {shape}")

torch.manual_seed(333)
x, gamma, beta = spec.make_inputs((1000, 333), "cuda", torch.float32)
gamma = torch.randn_like(gamma); beta = torch.randn_like(beta)
d333 = (layernorm_fixed(x, gamma, beta)
        - layernorm_buggy(x, gamma, beta)).abs().max().item()
print(f"  (1000,333): fixed-vs-buggy max abs diff = {d333:.3e} "
      f"(must be > 0: fix is live)")
if d333 == 0.0:
    failures.append("fixed == buggy at (1000,333): fix not live in tree")

torch.manual_seed(127)
x, gamma, beta = spec.make_inputs((64, 127), "cuda", torch.float32)
d127 = (layernorm_fixed(x, gamma, beta)
        - layernorm_buggy(x, gamma, beta)).abs().max().item()
print(f"  (64,127) [the post-oob non_pow2 variant width]: "
      f"fixed-vs-buggy max abs diff = {d127:.3e}")

# ---- (b) cross_shape margin at (1000,333), 10 seeds ----------------------
print("\n== (b) wrong_variance_estimate mutant at (1000,333), 10 seeds")
from TritonBench.cheating.layer_norm.wrong_variance_estimate import (
    layernorm as layernorm_mutant)

ATOL = 1e-4
errs_fixed, errs_buggy, errs_ideal = [], [], []
for seed in range(10):
    torch.manual_seed(seed)
    inputs = spec.make_inputs((1000, 333), "cuda", torch.float32)
    m = spec.run_candidate(layernorm_mutant, inputs)
    rf = spec.run_reference(layernorm_fixed, inputs)
    rb = spec.run_reference(layernorm_buggy, inputs)
    ideal = ideal_math(*inputs)
    errs_fixed.append((m - rf).abs().max().item())
    errs_buggy.append((m - rb).abs().max().item())
    errs_ideal.append((rf.double() - ideal).abs().max().item())

fmt = lambda v: f"[{min(v):.3e}, {max(v):.3e}]"
print(f"  mutant vs FIXED ref : max_err range {fmt(errs_fixed)}  "
      f"(atol {ATOL}; margin {ATOL/max(errs_fixed):.0f}x)")
print(f"  mutant vs BUGGY ref : max_err range {fmt(errs_buggy)}  "
      f"(the banked 0.0249 catch lives here)")
print(f"  FIXED ref vs float64 ideal: {fmt(errs_ideal)}")
if max(errs_fixed) >= ATOL:
    failures.append("mutant fails cross_shape vs FIXED reference at "
                    "(1000,333) -- emulation prediction wrong")
if min(errs_buggy) < ATOL:
    failures.append("mutant PASSES vs buggy kernel at (1000,333) -- "
                    "banked catch not reproduced")
if max(errs_ideal) >= ATOL:
    failures.append("fixed reference disagrees with ideal math at "
                    "(1000,333)")

print()
if failures:
    print("LN-GPU-PROBE-FAIL")
    for f in failures:
        print("  *", f)
    sys.exit(1)
print("LN-GPU-PROBE-OK")
