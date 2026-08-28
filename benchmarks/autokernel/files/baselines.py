"""
SOTA baselines to compare your checker against.

Both baselines are deliberately simple re-implementations of the *published
methodology*, not the original papers' code (neither AutoKernel nor Sarkar's
fuzzer have been wired in verbatim -- this is an MVP). Swap in the real
implementations later if you want a tighter comparison; the harness/metrics
code doesn't change either way.
"""
import time
import numpy as np


def allclose_gate(ref_out: np.ndarray, cand_out: np.ndarray,
                   atol: float = 1e-2, rtol: float = 1e-2) -> bool:
    """Naive allclose baseline. Returns True if the candidate PASSES
    (i.e., is judged correct)."""
    if cand_out.shape != ref_out.shape:
        return False
    if not np.all(np.isfinite(cand_out)):
        return False
    return bool(np.allclose(cand_out, ref_out, atol=atol, rtol=rtol))


def _adversarial_stability_inputs(op: str, rng, base_shape):
    """Stand-in for AutoKernel's 'numerical-stability adversarial inputs'
    stage: large-identical-value rows, extreme dynamic range, near-zero
    variance -- the input classes it names explicitly."""
    variants = []
    if op == "softmax":
        variants.append((np.full(base_shape, 50.0),))          # large identical values
        variants.append((rng.normal(size=base_shape) * 1e4,))  # extreme dynamic range
    elif op == "layernorm":
        variants.append((np.full(base_shape, 3.0) + rng.normal(size=base_shape) * 1e-8,))  # near-zero variance
        variants.append((rng.normal(size=base_shape) * 1e4,))
    elif op == "matmul":
        m, k = base_shape[0], base_shape[1] if len(base_shape) > 1 else 32
        variants.append((rng.normal(size=(m, k)) * 1e3, rng.normal(size=(k, m)) * 1e3))
    return variants


def autokernel_gate(op: str, ref_fn, cand_fn, input_fn, rng,
                     atol: float = 1e-2, rtol: float = 1e-2,
                     n_shapes: int = 3):
    """Re-implementation of AutoKernel's published five-stage numerical gate
    (minus the compile stage, which doesn't apply to numpy stand-ins):

      1. smoke test          -- one default-shape input
      2. shape sweep          -- several shapes / sizes
      3. adversarial-stability inputs -- large-identical-value rows,
         extreme dynamic range, near-zero-variance normalization inputs
      4. determinism          -- run twice, compare

    Returns (passed: bool, failed_stage: str | None).
    """
    # Stage 1: smoke test
    args = input_fn(rng)
    try:
        ref_out = ref_fn(*args)
        cand_out = cand_fn(*args)
    except Exception:
        return False, "smoke_test"
    if not allclose_gate(ref_out, cand_out, atol, rtol):
        return False, "smoke_test"

    # Stage 2: shape sweep
    for _ in range(n_shapes):
        args_i = input_fn(rng)
        try:
            ref_i = ref_fn(*args_i)
            cand_i = cand_fn(*args_i)
        except Exception:
            return False, "shape_sweep"
        if not allclose_gate(ref_i, cand_i, atol, rtol):
            return False, "shape_sweep"

    # Stage 3: adversarial-stability inputs
    base_args = input_fn(rng)
    base_shape = base_args[0].shape
    for adv_args in _adversarial_stability_inputs(op, rng, base_shape):
        try:
            ref_adv = ref_fn(*adv_args)
            cand_adv = cand_fn(*adv_args)
        except Exception:
            return False, "adversarial_stability"
        if not allclose_gate(ref_adv, cand_adv, atol, rtol):
            return False, "adversarial_stability"

    # Stage 4: determinism
    det_args = input_fn(rng)
    try:
        out1 = cand_fn(*det_args)
        out2 = cand_fn(*det_args)
    except Exception:
        return False, "determinism"
    if not np.array_equal(out1, out2):
        return False, "determinism"

    return True, None


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return result, dt
