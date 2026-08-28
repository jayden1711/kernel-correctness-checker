"""
Generates the near-miss mutant family: TritonBench/near_miss/<op>/m<NNN>.py

Each mutant is the reference kernel verbatim with its output multiplied by
(1 + DELTA) inside the kernel, DELTA chosen by design_deltas.py so the
perturbation-check margin lands at the file's target (margin = DELTA*M/tol,
tol/M measured per op). The scaling is a REAL kernel bug shape (a
mis-scaled epilogue), not a wrapper shim -- the kernel itself computes the
wrong value, so every Layer-1 structural check sees an ordinary kernel.

Run:  .venv/bin/python generate_mutants.py     (idempotent)
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DESIGN = os.path.join(os.path.dirname(__file__), "..", "data",
                      "design_deltas.json")
TARGETS = {"m050": 0.5, "m080": 0.8, "m100": 1.0, "m125": 1.25, "m200": 2.0}

HEADER = '''"""
NEAR-MISS mutant ({op}, target perturbation-check margin {margin}x).

The reference kernel with its output scaled by (1 + DELTA), DELTA = {delta}
chosen so max_err = DELTA * max|f| lands at {margin}x the adaptive
tolerance 3*P95(||f(x+d)-f(x)||) on corpus-distribution inputs
(rho = tol/max|f| measured in verification_runs/near_miss_2026-08-28/).
NOT part of the published corpus -- this family exists so tolerance
experiments have a non-flat response surface
(margin CV across input draws ~{cv:.0f}%; margins near 1.0x genuinely
straddle the boundary seed to seed, by design).
"""
import torch
import triton
import triton.language as tl

DELTA = {delta}

'''

BODIES = {
    "layernorm": '''
@triton.jit
def layernorm_kernel(output_ptr, input_ptr, gamma_ptr, beta_ptr,
                     input_row_stride, output_row_stride,
                     n_rows, n_cols, eps, delta, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    mean = tl.sum(row, axis=0) / n_cols
    diff = tl.where(mask, row - mean, 0.0)
    variance = tl.sum(diff * diff, axis=0) / n_cols
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)
    out = (row - mean) / tl.sqrt(variance + eps) * gamma + beta
    out = out * (1.0 + delta)                    # THE BUG: mis-scaled epilogue
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, out, mask=mask)


def layernorm(x, gamma, beta, eps=1e-5):
    n_rows, n_cols = x.shape
    if gamma.numel() != n_cols or beta.numel() != n_cols:
        raise ValueError("layernorm: companion length mismatch")
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    layernorm_kernel[(n_rows,)](y, x, gamma, beta, x.stride(0), y.stride(0),
                                n_rows, n_cols, eps, DELTA,
                                BLOCK_SIZE=BLOCK_SIZE)
    return y
''',
    "softmax": '''
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, delta, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    out = numerator / denominator * (1.0 + delta)   # THE BUG
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, out, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0),
                              n_rows, n_cols, DELTA, BLOCK_SIZE=BLOCK_SIZE)
    return y
''',
    "gelu": '''
@triton.jit
def gelu_kernel(output_ptr, input_ptr, n_elements, delta,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    y = y * (1.0 + delta)                            # THE BUG
    tl.store(output_ptr + offsets, y, mask=mask)


def gelu(x):
    x_flat = x.contiguous().view(-1)
    n_elements = x_flat.numel()
    y_flat = torch.empty_like(x_flat)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_kernel[grid](y_flat, x_flat, n_elements, DELTA,
                      BLOCK_SIZE=BLOCK_SIZE)
    return y_flat.view_as(x)
''',
    "l2norm": '''
@triton.jit
def l2norm_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                  n_rows, n_cols, eps, delta, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    sq_sum = tl.sum(row * row, axis=0)
    norm = tl.sqrt(sq_sum + eps)
    output = row / norm * (1.0 + delta)              # THE BUG
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, output, mask=mask)


def l2norm(x, eps=1e-12):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    l2norm_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_rows, n_cols,
                             eps, DELTA, BLOCK_SIZE=BLOCK_SIZE)
    return y
''',
    "sum_reduction": '''
@triton.jit
def sum_reduce_kernel(output_ptr, input_ptr, input_row_stride,
                      n_rows, n_cols, delta, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    result = tl.sum(row, axis=0) * (1.0 + delta)     # THE BUG
    tl.store(output_ptr + row_idx, result)


def sum_reduction(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty((n_rows,), device=x.device, dtype=x.dtype)
    sum_reduce_kernel[(n_rows,)](y, x, x.stride(0), n_rows, n_cols, DELTA,
                                 BLOCK_SIZE=BLOCK_SIZE)
    return y
''',
}


def main():
    design = json.load(open(DESIGN))
    base = os.path.join(ROOT, "TritonBench", "near_miss")
    os.makedirs(base, exist_ok=True)
    open(os.path.join(base, "__init__.py"), "w").write("")
    for op, body in BODIES.items():
        opdir = os.path.join(base, op)
        os.makedirs(opdir, exist_ok=True)
        open(os.path.join(opdir, "__init__.py"), "w").write("")
        for name, margin in TARGETS.items():
            delta = design[op]["deltas"][str(margin)]
            src = HEADER.format(op=op, margin=margin, delta=repr(delta),
                                cv=design[op]["cv_pct"]) + body
            with open(os.path.join(opdir, f"{name}.py"), "w") as f:
                f.write(src)
            print("wrote", os.path.relpath(os.path.join(opdir, f"{name}.py"),
                                           ROOT))


if __name__ == "__main__":
    main()
