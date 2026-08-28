"""
Two more SOTA baselines, added after the initial allclose/autokernel_gate
comparison, per the "next SOTA after AutoKernel" literature-review
follow-up:

  - gpuemu ("Test-Input Generation for Tensor Programs", Sarkar, ASU,
    arXiv 2606.27396) -- a switchable-knob fuzzer (shape / dtype /
    value-distribution). The paper reports it as TWO distinct strategies
    with very different tradeoffs -- boundary-shape (78% recall, 0%
    control false-positive) vs adversarial-value (99% recall, 94% false
    positive, because injected NaN/Inf cascades through correct kernels
    and trips the validator's NaN check) -- so this is exposed as two
    separate systems here, matching how the paper itself reports them as
    separate rows rather than one merged score.

  - Propilot ("Tensor Algebraic Property Skeletons", arXiv 2606.06747)
    -- LLM-driven algebraic/metamorphic property-based testing over
    reusable, operator-agnostic "skeletons" (commutativity, associativity,
    identity, permutation invariance, decomposition, idempotence), each
    with explicit per-operator applicability rules and an AllClose /
    exact-equality oracle.

Both are, same spirit as baselines.py's autokernel_gate, "deliberately
simple re-implementations of the published methodology," not the
original authors' code:

  - gpuemu's public artifacts (github.com/Skelf-Research/gpuemu) were not
    pulled in; the two-strategy structure and its reported tradeoff is
    reproduced from the paper's description, not from their fuzzer.
  - Propilot targets 212 TVM operators via an LLM-authored skeleton
    library (4,579 generated tests). Hand-picking skeletons here only
    covers matmul + the four pure-reduction ops (sum/mean/max/min) --
    the operators where a *generic* skeleton (identity, permutation
    invariance, reduction-decomposition, scalar linearity) unambiguously
    applies without operator-specific reasoning. Every other operator
    gets NO skeleton and trivially passes every mutant -- this is not a
    bug, it faithfully reflects the coverage gap between a small,
    hand-picked skeleton set and Propilot's LLM-authored, 212-operator
    library. Expect propilot_gate's aggregate catch rate to look poor on
    this 29-op corpus for exactly that reason.

Both operate on the torch-native entry fields (family, to_torch,
torch_ref_fn, torch_mutant_fn) that tritonbench_registry.py attaches to
every corpus entry, rather than the numpy-facing ref_fn/mutant_fn/
input_fn -- both need real shape/family awareness the abstract 5-key
corpus contract doesn't expose.
"""
import time

import numpy as np
import torch


def _run_torch(fn, to_torch, np_args):
    torch_inputs = to_torch(np_args)
    if isinstance(torch_inputs, tuple):
        return fn(*torch_inputs)
    return fn(torch_inputs)


# ---------------------------------------------------------------------------
# gpuemu -- boundary-shape strategy
# ---------------------------------------------------------------------------

def _gpuemu_boundary_inputs(family, rng):
    """Tiny / non-power-of-two / large shape variants, one per family.
    Deliberately independent of tritonbench_registry.py's own corpus
    generators -- this is gpuemu's OWN input-generation strategy being
    modeled, not a reuse of this project's baseline shapes."""
    def f32(*shape):
        return rng.normal(size=shape).astype(np.float32)

    if family == "single":
        return [(f32(1, 8),), (f32(17, 333),), (f32(256, 512),)]
    if family == "layernorm":
        return [
            (f32(1, 8), f32(8), f32(8)),
            (f32(17, 333), f32(333), f32(333)),
            (f32(256, 512), f32(512), f32(512)),
        ]
    if family == "instancenorm":
        return [
            (f32(1, 4, 2, 2), f32(4), f32(4)),
            (f32(2, 4, 5, 5), f32(4), f32(4)),
        ]
    if family == "rmsnorm":
        return [
            (f32(1, 8), f32(8)),
            (f32(17, 333), f32(333)),
            (f32(256, 512), f32(512)),
        ]
    if family == "matmul":
        return [
            (f32(1, 8), f32(8, 1)),
            (f32(17, 33), f32(33, 29)),
            (f32(128, 256), f32(256, 128)),
        ]
    if family == "attention":
        return [
            (f32(1, 16), f32(1, 16), f32(1, 16)),
            (f32(65, 32), f32(65, 32), f32(65, 32)),
        ]
    if family == "groupnorm":
        return [(f32(1, 8, 2, 2), 2, f32(8), f32(8))]
    if family == "batchnorm":
        return [(f32(1, 8, 2, 2), f32(8), np.abs(f32(8)) + 0.5, f32(8), f32(8))]
    if family == "cross_entropy":
        n_rows = 17
        return [(f32(n_rows, 33), rng.integers(0, 33, size=(n_rows,)).astype(np.int64))]
    if family == "pool1d":
        return [(f32(1, 2, 17), 3, 3, 0)]
    if family == "pool2d":
        return [(f32(1, 2, 17, 17), 3, 3, 0)]
    if family == "pool3d":
        return [(f32(1, 2, 9, 9, 9), 2, 2, 0)]
    return []


def gpuemu_boundary_shape_system(entry, is_mutant, rng):
    family = entry["family"]
    to_torch = entry["to_torch"]
    cand_fn = entry["torch_mutant_fn"] if is_mutant else entry["torch_ref_fn"]
    ref_fn = entry["torch_ref_fn"]

    t0 = time.perf_counter()
    variants = _gpuemu_boundary_inputs(family, rng)
    if not variants:
        return True, time.perf_counter() - t0, "no boundary-shape variant defined for this family"

    for np_args in variants:
        try:
            ref_out = _run_torch(ref_fn, to_torch, np_args)
            cand_out = _run_torch(cand_fn, to_torch, np_args)
        except Exception as e:
            return False, time.perf_counter() - t0, f"boundary_shape exception: {e}"

        if cand_out.shape != ref_out.shape:
            return False, time.perf_counter() - t0, "boundary_shape: shape mismatch"
        if not torch.allclose(cand_out.float(), ref_out.float(), atol=1e-2, rtol=1e-2):
            return False, time.perf_counter() - t0, "boundary_shape: numeric mismatch"

    return True, time.perf_counter() - t0, None


# ---------------------------------------------------------------------------
# gpuemu -- adversarial-value strategy
# ---------------------------------------------------------------------------

def _gpuemu_adversarial_value_inputs(base_args, rng, mode):
    """Value-distribution knob: extreme magnitude or NaN/Inf injection,
    SAME shape as base_args -- only floating-point ndarray entries are
    perturbed (int/scalar args like kernel_size, targets pass through)."""
    out = []
    for a in base_args:
        if isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.floating):
            if mode == "extreme":
                out.append((a * 1e6).astype(a.dtype))
            else:  # nan_inf
                b = a.copy()
                mask = rng.random(b.shape) < 0.01
                b[mask] = np.nan
                out.append(b)
        else:
            out.append(a)
    return tuple(out)


def gpuemu_adversarial_value_system(entry, is_mutant, rng):
    """Deliberately naive comparator: plain torch.allclose with NO
    equal_nan handling. This is exactly "the validator's NaN check" the
    paper describes tripping on correct kernels too -- a single injected
    NaN anywhere in a reduction row (softmax, norm, reduction, ...)
    legitimately poisons that row's whole output for BOTH reference and
    candidate, but comparing NaN against NaN with equal_nan=False still
    reports a mismatch. That's the paper's own reported 94% FP mechanism,
    not a bug in this re-implementation."""
    to_torch = entry["to_torch"]
    cand_fn = entry["torch_mutant_fn"] if is_mutant else entry["torch_ref_fn"]
    ref_fn = entry["torch_ref_fn"]

    t0 = time.perf_counter()
    base_args = entry["input_fn"](rng)

    for mode in ("extreme", "nan_inf"):
        adv_args = _gpuemu_adversarial_value_inputs(base_args, rng, mode)
        try:
            ref_out = _run_torch(ref_fn, to_torch, adv_args)
            cand_out = _run_torch(cand_fn, to_torch, adv_args)
        except Exception as e:
            return False, time.perf_counter() - t0, f"adversarial_value({mode}) exception: {e}"

        if cand_out.shape != ref_out.shape:
            return False, time.perf_counter() - t0, f"adversarial_value({mode}): shape mismatch"
        if not torch.allclose(cand_out.float(), ref_out.float(), atol=1e-2, rtol=1e-2):
            return False, time.perf_counter() - t0, f"adversarial_value({mode}): mismatch"

    return True, time.perf_counter() - t0, None


# ---------------------------------------------------------------------------
# Propilot -- generic algebraic property skeletons, applicability-gated
# ---------------------------------------------------------------------------

def _sk_identity_matmul(cand_fn, to_torch, base_args, rng):
    """A @ I == A. Built as a fresh (K,K) square identity independent of
    the corpus's own B shape, since the property only holds for a square
    identity -- this corpus's matmul entries use a non-square B."""
    A_np, B_np = base_args
    K = B_np.shape[0]
    I_np = np.eye(K, dtype=B_np.dtype)
    A_t, I_t = to_torch((A_np, I_np))
    out = cand_fn(A_t, I_t)
    return out.shape == A_t.shape and torch.allclose(out.float(), A_t.float(), atol=1e-2, rtol=1e-2)


def _sk_scalar_linearity_matmul(cand_fn, to_torch, base_args, rng):
    """(c*A) @ B == c*(A@B)."""
    A_np, B_np = base_args
    c = 10.0
    A_t, B_t = to_torch((A_np, B_np))
    lhs = cand_fn(A_t * c, B_t)
    rhs = cand_fn(A_t, B_t) * c
    return torch.allclose(lhs.float(), rhs.float(), atol=1e-2, rtol=1e-2)


def _sk_permutation_invariance_reduction(cand_fn, to_torch, base_args, rng):
    """reduce(permute(x)) == reduce(x) -- holds for a full-axis reduction
    to one value per row, regardless of element order."""
    x_t = to_torch(base_args)
    perm = torch.randperm(x_t.shape[-1])
    x_perm_t = x_t[..., perm].contiguous()
    out1 = cand_fn(x_t)
    out2 = cand_fn(x_perm_t)
    return torch.allclose(out1.float(), out2.float(), atol=1e-3, rtol=1e-3)


def _sk_decomposition_sum(cand_fn, to_torch, base_args, rng):
    """sum(x) == sum(x[:half]) + sum(x[half:]) -- Propilot's own named
    'reduction decomposition' skeleton."""
    x_t = to_torch(base_args)
    half = x_t.shape[-1] // 2
    whole = cand_fn(x_t)
    part = cand_fn(x_t[..., :half].contiguous()) + cand_fn(x_t[..., half:].contiguous())
    return torch.allclose(whole.float(), part.float(), atol=1e-2, rtol=1e-2)


def _sk_scalar_linearity_sum(cand_fn, to_torch, base_args, rng):
    """sum(c*x) == c*sum(x)."""
    x_t = to_torch(base_args)
    c = 10.0
    lhs = cand_fn(x_t * c)
    rhs = cand_fn(x_t) * c
    return torch.allclose(lhs.float(), rhs.float(), atol=1e-2, rtol=1e-2)


def _sk_decomposition_max(cand_fn, to_torch, base_args, rng):
    """max(x) == max(max(x[:half]), max(x[half:]))."""
    x_t = to_torch(base_args)
    half = x_t.shape[-1] // 2
    whole = cand_fn(x_t)
    part = torch.maximum(cand_fn(x_t[..., :half].contiguous()), cand_fn(x_t[..., half:].contiguous()))
    return torch.allclose(whole.float(), part.float(), atol=1e-2, rtol=1e-2)


def _sk_decomposition_min(cand_fn, to_torch, base_args, rng):
    """min(x) == min(min(x[:half]), min(x[half:]))."""
    x_t = to_torch(base_args)
    half = x_t.shape[-1] // 2
    whole = cand_fn(x_t)
    part = torch.minimum(cand_fn(x_t[..., :half].contiguous()), cand_fn(x_t[..., half:].contiguous()))
    return torch.allclose(whole.float(), part.float(), atol=1e-2, rtol=1e-2)


def _propilot_applicable_skeletons(op):
    """Applicability rules -- Propilot's own methodology gates skeletons
    by operator, it doesn't force every skeleton onto every operator."""
    if op == "matmul":
        return [("identity", _sk_identity_matmul),
                ("scalar_linearity", _sk_scalar_linearity_matmul)]
    if op == "sum_reduction":
        return [("permutation_invariance", _sk_permutation_invariance_reduction),
                ("decomposition", _sk_decomposition_sum),
                ("scalar_linearity", _sk_scalar_linearity_sum)]
    if op == "mean_reduction":
        return [("permutation_invariance", _sk_permutation_invariance_reduction),
                ("scalar_linearity", _sk_scalar_linearity_sum)]
    if op == "max_reduction":
        return [("permutation_invariance", _sk_permutation_invariance_reduction),
                ("decomposition", _sk_decomposition_max)]
    if op == "min_reduction":
        return [("permutation_invariance", _sk_permutation_invariance_reduction),
                ("decomposition", _sk_decomposition_min)]
    return []


def propilot_system(entry, is_mutant, rng):
    op = entry["op"]
    to_torch = entry["to_torch"]
    cand_fn = entry["torch_mutant_fn"] if is_mutant else entry["torch_ref_fn"]

    t0 = time.perf_counter()
    checks = _propilot_applicable_skeletons(op)
    if not checks:
        return True, time.perf_counter() - t0, "no applicable skeleton for this operator"

    base_args = entry["input_fn"](rng)
    for skeleton_name, skeleton_fn in checks:
        try:
            ok = skeleton_fn(cand_fn, to_torch, base_args, rng)
        except Exception as e:
            return False, time.perf_counter() - t0, f"{skeleton_name} exception: {e}"
        if not ok:
            return False, time.perf_counter() - t0, f"{skeleton_name} violated"

    return True, time.perf_counter() - t0, None
