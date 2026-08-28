"""
AutoKernel's five-stage correctness gate, re-implemented against the
PUBLISHED specification (arXiv 2603.21331, Jaber & Jaber, "AutoKernel:
Autonomous GPU Kernel Optimization via Iterative Agent-Driven Search").

WHY THIS FILE EXISTS
--------------------
`baselines.py:autokernel_gate` is an earlier approximation written before
the paper's harness section was read closely. Auditing it against the
paper turned up five deviations, two of which are outright bugs that
manufacture false positives on a CORRECT reference kernel. Since
autokernel_gate is the single largest reported margin (18% FPR vs. this
project's 0%), that margin was resting on those two bugs. This file is
the corrected baseline; `baselines.py:autokernel_gate` is kept unchanged
so the delta between the two is measurable in one benchmark run rather
than being asserted.

THE PUBLISHED SPEC, QUOTED
--------------------------
  "The benchmark harness (bench.py, 1,416 lines) enforces correctness
   through five stages. All must pass before performance is measured.

   Stage 1: Smoke test. A single forward pass on a small input (e.g.
   128x128) catches compilation errors, shape mismatches, and gross
   numerical bugs in under 1 second.

   Stage 2: Shape sweep. The kernel runs across 8 to 10 input
   configurations and three data types. [...] This catches size-dependent
   bugs: boundary handling, tile remainder logic, and dtype-specific
   issues.

   Stage 3: Numerical stability. Adversarial inputs probe floating-point
   edge cases. For softmax: rows of large identical values. For matmul:
   extreme dynamic range. For normalization: near-zero variance.

   Stage 4: Determinism. Same input, three runs, bitwise identical
   outputs. Catches race conditions in parallel reductions and
   non-deterministic atomics.

   Stage 5: Edge cases. Non-power-of-two dimensions (1023, 4097, 1537)
   expose masking bugs and tile remainder errors.

   Tolerances are dtype-specific: FP16 uses atol=1e-2, BF16 uses 2e-2,
   FP32 uses 1e-4."

  Each kernel "has a PyTorch reference in reference.py serving as the
  correctness oracle."

AUDIT: baselines.py:autokernel_gate vs. the above
-------------------------------------------------
  1. TOLERANCE -- used atol=1e-2, rtol=1e-2 for an all-FP32 corpus. The
     paper's FP32 tolerance is atol=1e-4; 1e-2 is its FP16 tolerance.
     The old gate was therefore 100x LOOSER than published on every
     comparison, which UNDERSTATES AutoKernel's catch rate.
  2. SHAPE SWEEP -- ran n_shapes=3 by calling input_fn(rng) repeatedly,
     but every _mk_* generator in tritonbench_registry.py returns a
     FIXED shape. So stage 2 re-drew random VALUES at one shape and one
     dtype. It was not a shape sweep at all; published is 8-10 configs
     x 3 dtypes. Also understates catch rate.
  3. STAGE 3 ARITY BUG -- _adversarial_stability_inputs returned
     1-tuples for layernorm, but the layernorm reference is
     layernorm(x, gamma, beta). The 1-tuple hits _to_torch_triple's
     `x, w1, w2 = args` and raises ValueError, which the bare
     `except Exception` scores as a gate FAILURE. On a reference-vs-
     reference trial that is a false positive, 100% of the time.
  4. STAGE 3 DTYPE BUG -- it is the only input generator in the repo
     that never calls .astype(np.float32), so rng.normal() stays
     float64. torch.from_numpy gives an fp64 CUDA tensor, and Triton's
     tl.dot has no fp64 path, so the matmul reference raises. Same
     bare-except, same manufactured false positive, 100% of the time.
     (softmax is unaffected: correct arity, and its kernel is
     elementwise+reduction with no tl.dot, so fp64 executes fine --
     which is exactly why softmax shows 0% FP and layernorm/matmul show
     100% in results.md.)
  5. MISSING STAGE 5 -- omitted, on the stated grounds that stage 5 was
     "the compile stage, which doesn't apply to numpy stand-ins". Stage
     5 is not a compile stage; it is non-power-of-two edge-case
     coverage, which applies directly and would raise the catch rate.
     Stage 4 also ran twice, not the published three times.

  Net direction: (1), (2) and (5) all suppress AutoKernel's catch rate;
  (3) and (4) inflate its false-positive rate. Both errors flatter this
  project. That is the finding, and it is why this file exists.

DELIBERATE DEVIATIONS FROM THE PAPER (documented, not accidental)
-----------------------------------------------------------------
  - ABSOLUTE SHAPE SIZES. The paper's matmul sweep runs up to
    4096x11008x4096, sized for performance benchmarking of a handful of
    kernels. This corpus runs 40 mutants x 6 trials x 8 configs x 3
    dtypes per system; at the paper's sizes the sweep alone would
    dominate the benchmark's wall time by orders of magnitude. Shapes
    here preserve the STRUCTURE the paper's sweep tests for -- 8
    configurations spanning batch-1, tiny, non-power-of-two, and
    largest-in-family -- at sizes proportionate to this corpus. The
    fidelity claim is on the sweep's structure and count, not on its
    absolute dimensions.
  - rtol. The paper specifies an absolute tolerance only and never
    mentions a relative one. np.allclose's rtol defaults to 1e-5, and
    that default is what a PyTorch/NumPy implementation of "atol=1e-4"
    would inherit, so it is kept. RTOL is exposed as a parameter so the
    rtol=0 (strict-literal) reading can be measured too -- see
    tolerance_sensitivity() below.
  - REFERENCE-INFEASIBLE CONFIGS ARE SKIPPED, NOT FAILED. If the
    REFERENCE kernel itself raises on a (shape, dtype) config -- e.g. a
    TritonBench kernel with no bf16 path -- that is a limitation of
    this corpus's references, not evidence about the candidate. Such
    configs are skipped and counted in `skipped`. Failing the candidate
    for them would re-introduce exactly the class of artifact this file
    exists to remove. Configs where the reference succeeds and the
    CANDIDATE raises are still hard failures.
"""
import time

import numpy as np
import torch


# Tolerances READ FROM AutoKernel's OWN bench.py (github.com/RightNow-AI/
# autokernel), not inferred from the paper's prose. CORRECTED 2026-08-25.
#
# The paper's harness section gives an absolute tolerance and never mentions a
# relative one, which is why this file previously carried RTOL = 1e-5 (the
# NumPy/PyTorch default an "atol=1e-4" implementation would inherit). The
# SOURCE contradicts that: bench.py pairs every atol with an EQUAL rtol.
#
#     float16:  atol=1e-2,  rtol=1e-2
#     bfloat16: atol=2e-2,  rtol=2e-2
#     float32:  atol=1e-4,  rtol=1e-4
#
# For FP32 the real rtol is therefore 1e-4 -- 10x LOOSER than the 1e-5 this
# file used. That direction matters: the previously reported 80% catch rate
# was measured with a stricter comparator than AutoKernel actually uses, so it
# is an UPPER bound on the real gate's catch rate, not an estimate of it.
ATOL_BY_DTYPE = {
    torch.float32: 1e-4,
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
}

RTOL_BY_DTYPE = {
    torch.float32: 1e-4,
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
}

# Retained ONLY so `rtol=` overrides still resolve to something; the default
# path now reads RTOL_BY_DTYPE. See tolerance_sensitivity().
RTOL = None

SWEEP_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _allclose(ref_out, cand_out, atol, rtol):
    """Gate comparison. Mirrors baselines.py:allclose_gate's semantics
    (shape check, finiteness check, then allclose) so the only thing
    that differs between the old gate and this one is what the audit
    above identified -- not the comparator."""
    if cand_out.shape != ref_out.shape:
        return False
    if not torch.isfinite(cand_out).all():
        return False
    return bool(torch.allclose(cand_out.float(), ref_out.float(),
                               atol=atol, rtol=rtol))


# ---------------------------------------------------------------------------
# Stage 2: shape sweep -- 8 configurations per family.
#
# Deliberately independent of tritonbench_registry.py's own _mk_* corpus
# generators, which are fixed-shape by construction. Same precedent as
# sota_baselines.py:_gpuemu_boundary_inputs modelling gpuemu's own input
# strategy rather than reusing the corpus's.
#
# Each entry returns a full positional-argument tuple for that family,
# already at the requested dtype/device -- non-tensor hyperparameters
# (num_groups, kernel_size, stride, padding) pass through as ints.
# ---------------------------------------------------------------------------

def _sweep_shapes(family):
    """8 configs per family: batch-1, tiny, several mid sizes, two
    non-power-of-two, and the largest the family supports at this
    corpus's scale."""
    if family in ("single", "layernorm", "rmsnorm", "cross_entropy"):
        return [(1, 128), (8, 64), (32, 128), (64, 128),
                (17, 333), (64, 1023), (128, 512), (256, 512)]
    if family == "matmul":
        return [(1, 16, 1), (8, 8, 8), (32, 16, 32), (64, 64, 64),
                (17, 33, 29), (127, 65, 63), (128, 256, 128), (256, 128, 256)]
    if family == "attention":
        return [(1, 16), (8, 16), (32, 32), (64, 32),
                (65, 32), (127, 64), (128, 64), (256, 64)]
    if family == "instancenorm":
        return [(1, 4, 2, 2), (2, 4, 4, 4), (2, 8, 4, 4), (4, 8, 8, 8),
                (1, 3, 5, 5), (2, 6, 7, 7), (4, 16, 8, 8), (8, 16, 16, 16)]
    if family == "groupnorm":
        return [(1, 4, 2, 2, 2), (2, 8, 4, 4, 2), (2, 8, 4, 4, 4),
                (4, 16, 8, 8, 4), (1, 6, 5, 5, 3), (2, 12, 7, 7, 3),
                (4, 16, 8, 8, 8), (8, 32, 8, 8, 4)]
    if family == "batchnorm":
        return [(1, 4, 2, 2), (2, 8, 4, 4), (2, 8, 8, 8), (4, 16, 8, 8),
                (1, 3, 5, 5), (2, 6, 7, 7), (4, 16, 16, 16), (8, 32, 8, 8)]
    if family == "pool1d":
        return [(1, 1, 16, 4, 4, 0), (2, 3, 32, 4, 4, 0), (2, 3, 64, 2, 2, 0),
                (4, 8, 128, 4, 4, 0), (1, 2, 17, 3, 3, 0), (2, 3, 33, 3, 3, 0),
                (4, 8, 256, 8, 8, 0), (8, 16, 512, 4, 4, 0)]
    if family == "pool2d":
        return [(1, 1, 8, 8, 4, 4, 0), (2, 3, 16, 16, 4, 4, 0),
                (2, 3, 32, 32, 2, 2, 0), (4, 8, 64, 64, 4, 4, 0),
                (1, 2, 17, 17, 3, 3, 0), (2, 3, 33, 33, 3, 3, 0),
                (4, 8, 64, 64, 8, 8, 0), (8, 16, 32, 32, 4, 4, 0)]
    if family == "pool3d":
        return [(1, 1, 4, 4, 4, 2, 2, 0), (2, 3, 8, 8, 8, 2, 2, 0),
                (2, 3, 8, 8, 8, 4, 4, 0), (4, 8, 16, 16, 16, 2, 2, 0),
                (1, 2, 9, 9, 9, 3, 3, 0), (2, 3, 9, 9, 9, 3, 3, 0),
                (4, 8, 16, 16, 16, 4, 4, 0), (8, 8, 8, 8, 8, 2, 2, 0)]
    return []


def _build_args(family, shape, dtype, device, gen):
    """Materialize one family's full positional-arg tuple at `shape`."""
    def t(*dims):
        return torch.randn(*dims, device=device, dtype=dtype, generator=gen)

    if family == "single":
        return (t(*shape),)
    if family == "layernorm":
        n_cols = shape[-1]
        return (t(*shape), t(n_cols), t(n_cols))
    if family == "rmsnorm":
        return (t(*shape), t(shape[-1]))
    if family == "matmul":
        M, K, N = shape
        return (t(M, K), t(K, N))
    if family == "attention":
        N, D = shape
        return (t(N, D), t(N, D), t(N, D))
    if family == "instancenorm":
        C = shape[1]
        return (t(*shape), t(C), t(C))
    if family == "groupnorm":
        *tensor_shape, num_groups = shape
        C = tensor_shape[1]
        return (t(*tensor_shape), num_groups, t(C), t(C))
    if family == "batchnorm":
        C = shape[1]
        running_var = torch.rand(C, device=device, dtype=dtype, generator=gen) + 0.5
        return (t(*shape), t(C), running_var, t(C), t(C))
    if family == "cross_entropy":
        n_rows, n_cols = shape
        targets = torch.randint(0, n_cols, (n_rows,), device=device, generator=gen)
        return (t(n_rows, n_cols), targets)
    if family in ("pool1d", "pool2d", "pool3d"):
        *tensor_shape, kernel_size, stride, padding = shape
        return (t(*tensor_shape), kernel_size, stride, padding)
    return None


# ---------------------------------------------------------------------------
# Stage 3: numerical stability.
#
# CORRECTED 2026-08-25 against AutoKernel's OWN bench.py
# (github.com/RightNow-AI/autokernel). The previous model here was built from
# the paper's prose, which names three probe classes by example --
#   "For softmax: rows of large identical values. For matmul: extreme dynamic
#    range. For normalization: near-zero variance."
# -- and so applied a probe to only 11 of this corpus's 29 operators, leaving
# 18 reported as "uncovered".
#
# THE SOURCE DOES SOMETHING DIFFERENT AND BROADER. bench.py applies FIVE
# input-scaling transforms to EVERY kernel, keyed to nothing about the
# operator:
#
#   near_max     x * 60000 for fp16, x * 1e30 otherwise
#   near_zero    x * 1e-6
#   mixed_scale  each element scaled by 1e3 or 1e-3 at random
#   all_zeros    x := 0
#   all_same     x := 0.5
#
# and it RELAXES the tolerance 10x for these inputs. So the real stage 3 is
# wider in coverage (29 of 29 operators, not 11) and looser in strictness
# (10x tolerance) than the version this file used to implement. Both
# directions had to change; they do not cancel, and which dominates is an
# empirical question -- see the re-run note in AUTOKERNEL_BASELINE_AUDIT.md §7.
#
# Applied to the PRIMARY tensor only, at its exact shape/dtype/device. That is
# unchanged and remains load-bearing: rebuilding the whole argument tuple is
# what produced the arity and dtype bugs documented in the audit.
# ---------------------------------------------------------------------------

# bench.py relaxes tolerance 10x for adversarial inputs.
STABILITY_TOLERANCE_RELAXATION = 10.0


def _stability_variants(base_primary, gen):
    """The five transforms bench.py applies, in its order. Returns
    [(name, tensor), ...] -- every operator gets all five."""
    x = base_primary
    near_max_scale = 60000.0 if x.dtype == torch.float16 else 1e30

    mixed = torch.where(
        torch.rand(x.shape, device=x.device, generator=gen) < 0.5,
        torch.full_like(x, 1e3),
        torch.full_like(x, 1e-3),
    )

    return [
        ("near_max", x * near_max_scale),
        ("near_zero", x * 1e-6),
        ("mixed_scale", x * mixed),
        ("all_zeros", torch.zeros_like(x)),
        ("all_same", torch.full_like(x, 0.5)),
    ]


# ---------------------------------------------------------------------------
# Stage 5: edge cases -- non-power-of-two dimensions.
#
# Published values are 1023, 4097, 1537. 4097 is retained for families
# whose cost is linear in that dim; for matmul/attention (quadratic and
# cubic respectively) it is dropped to keep the 240-trial benchmark
# tractable, and that drop is recorded in the returned detail rather
# than hidden.
# ---------------------------------------------------------------------------

_NPOT = (1023, 1537, 4097)


def _edge_case_shapes(family):
    if family in ("single", "layernorm", "rmsnorm", "cross_entropy"):
        return [(8, n) for n in _NPOT], []
    if family == "matmul":
        return [(17, 1023, 33), (127, 1537, 65)], [4097]
    if family == "attention":
        return [(1023, 32)], [1537, 4097]
    if family == "instancenorm":
        return [(1, 3, 33, 31)], list(_NPOT)
    if family == "groupnorm":
        return [(1, 6, 33, 31, 3)], list(_NPOT)
    if family == "batchnorm":
        return [(1, 3, 33, 31)], list(_NPOT)
    if family == "pool1d":
        return [(1, 2, 1023, 3, 3, 0), (1, 2, 1537, 3, 3, 0)], [4097]
    if family == "pool2d":
        return [(1, 2, 33, 31, 3, 3, 0)], list(_NPOT)
    if family == "pool3d":
        return [(1, 2, 9, 11, 13, 2, 2, 0)], list(_NPOT)
    return [], list(_NPOT)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def autokernel_gate_faithful(entry, is_mutant, rng, rtol=None):
    """
    Five-stage gate per arXiv 2603.21331. Harness system contract:
    (entry, is_mutant, rng) -> (passed, dt, detail).

    Operates on the torch-native entry fields (family / torch_ref_fn /
    torch_mutant_fn) rather than the numpy-facing ref_fn/mutant_fn,
    because a faithful shape sweep needs real family awareness -- the
    abstract 5-key corpus contract cannot express "build this family's
    argument tuple at this shape", and guessing it is what produced the
    arity bug in the original.
    """
    # rtol is a PARAMETER, not the module global, so the strict-literal
    # atol-only reading (rtol=0) can run as another SYSTEM in the same
    # benchmark pass -- via functools.partial in run_benchmark.py -- instead
    # of requiring a second full Colab session with the global flipped.
    rtol = RTOL if rtol is None else rtol

    family = entry["family"]
    op = entry["op"]
    ref_fn = entry["torch_ref_fn"]
    cand_fn = entry["torch_mutant_fn"] if is_mutant else entry["torch_ref_fn"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed the torch generator FROM the harness's shared numpy rng rather
    # than from a constant. harness.py re-runs the reference N_TRIALS_FPR=5
    # times specifically because "some checks are stochastic"; a constant
    # seed would hand all 5 trials byte-identical inputs, so they would
    # return the same verdict by construction and a 0% FP rate over 5
    # trials would carry the evidential weight of 1. Drawing from the
    # shared rng also matches how every other system in the harness
    # consumes randomness.
    # TIMER SCOPE: t0 before the seed draw, so this system times everything it
    # does to reach a verdict -- matching every other system in the harness.
    # See the convention note in harness.run().
    t0 = time.perf_counter()

    gen = torch.Generator(device=device)
    gen.manual_seed(int(rng.integers(0, 2**31 - 1)))

    skipped = []

    def _elapsed():
        return time.perf_counter() - t0

    def _run_config(shape, dtype, stage):
        """Returns (verdict, detail) where verdict is True (pass),
        False (candidate failed), or None (skip -- reference infeasible)."""
        args = _build_args(family, shape, dtype, device, gen)
        if args is None:
            return None, f"{stage}: no builder for family {family}"
        try:
            ref_out = ref_fn(*args)
        except Exception as e:
            skipped.append(f"{stage} {shape}/{dtype}: reference infeasible ({type(e).__name__})")
            return None, None
        try:
            cand_out = cand_fn(*args)
        except Exception as e:
            return False, f"{stage}: candidate raised on {shape}/{dtype}: {type(e).__name__}: {e}"
        if not _allclose(ref_out, cand_out, ATOL_BY_DTYPE[dtype],
                         RTOL_BY_DTYPE[dtype] if rtol is None else rtol):
            return False, f"{stage}: mismatch at {shape}/{dtype}"
        return True, None

    # -- Stage 1: smoke test. Single small forward pass, FP32. ------------
    sweep = _sweep_shapes(family)
    if not sweep:
        return True, _elapsed(), f"no sweep defined for family {family} -- gate vacuous"
    verdict, detail = _run_config(sweep[1], torch.float32, "smoke_test")
    if verdict is False:
        return False, _elapsed(), detail

    # -- Stage 2: shape sweep. 8 configs x 3 dtypes. ----------------------
    for shape in sweep:
        for dtype in SWEEP_DTYPES:
            verdict, detail = _run_config(shape, dtype, "shape_sweep")
            if verdict is False:
                return False, _elapsed(), detail

    # -- Stage 3: numerical stability. ------------------------------------
    # Five transforms, every operator, tolerance relaxed 10x. See the block
    # comment above for why this replaced the three-probe-class model.
    base_shape = sweep[3]
    base_args = _build_args(family, base_shape, torch.float32, device, gen)
    if base_args is None:
        skipped.append(f"numerical_stability: no builder for family {family}")
    else:
        stab_atol = ATOL_BY_DTYPE[torch.float32] * STABILITY_TOLERANCE_RELAXATION
        stab_rtol = (RTOL_BY_DTYPE[torch.float32] if rtol is None else rtol) \
            * STABILITY_TOLERANCE_RELAXATION
        for variant_name, adv_primary in _stability_variants(base_args[0], gen):
            adv_args = (adv_primary,) + tuple(base_args[1:])
            try:
                ref_out = ref_fn(*adv_args)
            except Exception as e:
                # Reference-infeasible => skip, same rule as the sweep. Failing
                # the candidate here would re-introduce the artifact class the
                # audit removed.
                skipped.append(f"numerical_stability/{variant_name}: "
                               f"reference infeasible ({type(e).__name__})")
                continue
            # SAME RULE, SILENT CASE -- added 2026-08-25 after it fired.
            # `near_max` scales the primary by 1e30; for the attention family
            # QK^T then reaches ~1e60 and OVERFLOWS fp32 (max ~3.4e38). The
            # reference does not raise -- it returns inf. `_allclose` rejects
            # non-finite candidate output, so the reference's own overflowed
            # result was scored as a CANDIDATE failure: on a
            # reference-vs-reference trial, a deterministic false positive.
            # Measured: 25 of 26 FPs in the first corrected run, 100% of
            # flash_attention and scaled_dot_product_attention trials.
            #
            # A config whose REFERENCE cannot produce a finite answer carries
            # no information about the candidate, exactly as when it raises.
            # This is the audit's own reference-infeasible rule applied to the
            # silent case; it is NOT a new policy. The old stage 3 never hit
            # it because it probed only 11 of 29 operators, none of which
            # overflowed.
            if not torch.isfinite(ref_out).all():
                skipped.append(f"numerical_stability/{variant_name}: "
                               f"reference non-finite (overflow) -- skipped")
                continue
            try:
                cand_out = cand_fn(*adv_args)
            except Exception as e:
                return False, _elapsed(), (f"numerical_stability/{variant_name}: "
                                           f"candidate raised: {type(e).__name__}: {e}")
            if not _allclose(ref_out, cand_out, stab_atol, stab_rtol):
                return False, _elapsed(), f"numerical_stability/{variant_name}: mismatch"

    # -- Stage 4: determinism. Three runs, bitwise identical. -------------
    det_args = _build_args(family, base_shape, torch.float32, device, gen)
    try:
        outs = [cand_fn(*det_args) for _ in range(3)]
    except Exception as e:
        return False, _elapsed(), f"determinism: candidate raised: {type(e).__name__}: {e}"
    if not (torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])):
        return False, _elapsed(), "determinism: three runs not bitwise identical"

    # -- Stage 5: edge cases. Non-power-of-two dimensions. ----------------
    edge_shapes, dropped = _edge_case_shapes(family)
    if dropped:
        skipped.append(f"edge_cases: published dims {dropped} dropped for family "
                       f"{family} (cost superlinear in that dim at 240-trial scale)")
    for shape in edge_shapes:
        verdict, detail = _run_config(shape, torch.float32, "edge_cases")
        if verdict is False:
            return False, _elapsed(), detail

    return True, _elapsed(), ("; ".join(skipped) if skipped else None)


def tolerance_sensitivity():
    """Tolerances now come from AutoKernel's bench.py directly, so there is no
    longer a competing 'reading' to measure -- the rtol=0 strict-literal
    reading was an inference from the paper's prose and the source contradicts
    it. Retained as a reporting helper only."""
    return {
        "source": "github.com/RightNow-AI/autokernel bench.py (read 2026-08-25)",
        "atol_by_dtype": {str(k): v for k, v in ATOL_BY_DTYPE.items()},
        "rtol_by_dtype": {str(k): v for k, v in RTOL_BY_DTYPE.items()},
        "stability_relaxation": STABILITY_TOLERANCE_RELAXATION,
        "retired_reading": "rtol=0 -- contradicted by source, do not re-add",
    }
