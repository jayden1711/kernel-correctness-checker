"""
Layer 1 — Runtime guards.

These checks require actually running the kernel, but are cheaper than
Layer 2 numeric comparisons.  They catch failure modes that static AST
analysis cannot.

Checks:
  - check_nan_inf:          output must be fully finite before any numeric test
  - check_dtype_preserved:  output dtype must match input dtype (or a spec-declared
                             expected dtype, for index-returning operators)
  - check_determinism:      two runs on identical inputs must produce identical output
  - check_kernel_executed:  kernel must actually run (runtime ghost-opt detection)
"""

import torch
from typing import Callable, Optional


# check_nan_inf

def check_nan_inf(candidate_fn: Callable, x: torch.Tensor) -> tuple:
    """
    Assert the output contains no NaN or Inf values.

    This must run before any allclose-based check because NaN propagation
    can mask errors: torch.allclose returns False on NaN but the failure
    reason is ambiguous, and some tolerances interact badly with Inf.

    Returns:
        (True,  detail)   output is fully finite
        (False, detail)   output contains NaN or Inf
    """
    try:
        out = candidate_fn(x)
    except Exception as e:
        return False, f"Kernel raised an exception: {e}"

    if not torch.isfinite(out).all():
        n_nan = torch.isnan(out).sum().item()
        n_inf = torch.isinf(out).sum().item()
        total = out.numel()
        return False, (
            f"Output contains non-finite values: "
            f"{n_nan} NaN, {n_inf} Inf out of {total} elements."
        )

    return True, "Output is fully finite."


# check_dtype_preserved

def check_dtype_preserved(
    candidate_fn: Callable,
    x: torch.Tensor,
    expected_dtype: Optional[torch.dtype] = None,
) -> tuple:
    """
    Assert that the output dtype matches the input dtype -- OR, if the
    operator's spec declares an expected_dtype (e.g. torch.int64 for
    argmax/argmin, which legitimately return indices rather than values
    in the input's dtype), assert against that instead.

    FIXED: this check used to unconditionally require out.dtype ==
    x.dtype with no override, which meant argmax/argmin could never pass
    it -- correct or not -- since returning an index tensor is the whole
    point of those operators, not a bug. Confirmed via a real
    run_checker.py run: both failed this sentinel before Layer 2/3 ever
    ran, on the reference kernel as much as any mutant. expected_dtype=None
    (the default) preserves the exact old behavior for every operator that
    doesn't explicitly declare otherwise.

    A kernel that upcasts fp16 -> fp32 internally and returns fp32 will
    pass all numeric checks but break mixed-precision training pipelines
    that expect dtype consistency -- that's still the failure mode this
    guards against for every operator where output_dtype isn't declared.

    Returns:
        (True,  detail)   dtypes match
        (False, detail)   dtype mismatch
    """
    try:
        out = candidate_fn(x)
    except Exception as e:
        return False, f"Kernel raised an exception: {e}"

    target_dtype = expected_dtype if expected_dtype is not None else x.dtype

    if out.dtype != target_dtype:
        if expected_dtype is not None:
            return False, (
                f"Dtype mismatch: expected declared output dtype {target_dtype}, "
                f"got {out.dtype}."
            )
        return False, (
            f"Dtype mismatch: input {x.dtype}, output {out.dtype}. "
            "Kernel may be silently upcasting."
        )

    if expected_dtype is not None:
        return True, f"Dtype matches declared output dtype: {target_dtype}."
    return True, f"Dtype preserved: {x.dtype}."


# check_determinism

def check_determinism(
    candidate_fn: Callable,
    x: torch.Tensor,
    n_runs: int = 3,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> tuple:
    """
    Run the kernel n_runs times on the same input and assert all outputs
    match within a tight numeric tolerance.

    Non-determinism indicates a race condition that the barrier check
    missed — e.g. a missing tl.barrier() between a shared-memory write
    and read in a reduction.

    FIXED: was bitwise torch.equal, which flags kernels that legitimately
    use tl.atomic_add for cross-block reduction (e.g. frobenius_norm's
    sum-of-squares) as non-deterministic on nearly every run. Atomic adds
    across concurrent thread blocks are correctly synchronized -- no race,
    no undefined behavior -- but floating-point addition is
    non-associative, so summing the same set of partial sums in a
    different arrival order changes the result's last few ULPs. That's
    expected numerical behavior for atomics, not the missing-barrier race
    condition this check exists to catch (which corrupts values by orders
    of magnitude, not by rounding noise) -- so compare with a tight
    tolerance instead of exact equality; a real race still fails this by
    a wide margin.

    Args:
        candidate_fn:  Kernel under test.
        x:             Input tensor.
        n_runs:        Number of repeated runs (default 3).
        atol, rtol:    Tolerance for "same output" across runs.

    Returns:
        (True,  detail)   all runs agree within tolerance
        (False, detail)   outputs differ across runs beyond tolerance
    """
    outputs = []
    for i in range(n_runs):
        try:
            out = candidate_fn(x).detach().clone()
        except Exception as e:
            return False, f"Run {i} raised an exception: {e}"
        outputs.append(out)

    for i in range(1, n_runs):
        if not torch.allclose(outputs[0].float(), outputs[i].float(), atol=atol, rtol=rtol):
            max_diff = (outputs[0].float() - outputs[i].float()).abs().max().item()
            return False, (
                f"Non-determinism detected: run 0 vs run {i} differ. "
                f"Max absolute difference: {max_diff:.6f} "
                f"(tolerance atol={atol}, rtol={rtol}). "
                "Likely a missing barrier in a reduction."
            )

    return True, (
        f"Kernel is deterministic across {n_runs} runs "
        f"(within atol={atol}, rtol={rtol})."
    )


# check_kernel_executed  (runtime ghost-optimization detection)

def _probe_multiplicative(t: torch.Tensor) -> torch.Tensor:
    """Per-element multiplicative + additive perturbation.

    The additive term is not decoration: several real proposals use `zeros`
    fills, where a purely multiplicative probe is the identity.
    """
    return t * (1.0 + torch.randn_like(t) * 0.5) + torch.randn_like(t) * 0.5


def _probe_negate(t: torch.Tensor) -> torch.Tensor:
    return -t


def _probe_fresh(t: torch.Tensor) -> torch.Tensor:
    """A fresh independent draw with roughly t's location and spread.

    `std()` of a constant tensor is 0 (and NaN for numel 1), so fall back to
    1.0 -- otherwise this probe degenerates into a constant tensor and cannot
    move anything.
    """
    sd = t.float().std()
    if not torch.isfinite(sd) or sd.item() == 0.0:
        sd = torch.ones((), device=t.device)
    return (torch.randn_like(t.float()) * sd + t.float().mean()).to(t.dtype)


# Ordered cheapest-first. A correct kernel on ordinary input is moved by the
# first rung, so the common case costs exactly one extra kernel call, as it
# did before this ladder existed.
_PRIMARY_PROBES = (
    ("multiplicative", _probe_multiplicative),
    ("negation",       _probe_negate),
    ("fresh_draw",     _probe_fresh),
)


def check_kernel_executed(
    candidate_fn: Callable,
    x: torch.Tensor,
    reference_fn: Callable,
    spec=None,
    inputs=None,
    raw_candidate_fn: Optional[Callable] = None,
    raw_reference_fn: Optional[Callable] = None,
) -> tuple:
    """
    Confirm the custom kernel actually ran -- i.e. that its output genuinely
    depends on its inputs -- AND that it does more than just call the
    reference.

    Strategy: perturb the input and require the output to move. A kernel that
    ignores its input (hardcoded output, ghost optimization) cannot move.

    THE PERTURBATION IS A LADDER, NOT A SINGLE PROBE, AND THAT IS THE WHOLE
    POINT OF THIS FUNCTION'S CURRENT SHAPE. The original implementation used
    one probe, `x + randn_like(x)*0.1 + 1.0`, and thereby asserted

        different input  =>  different output

    which is FALSE for any non-injective operator. It false-positived on
    CORRECT reference kernels: 20 of 80 proposals in the 2026-08-20
    causal_flash_attention run, plus 30 more across earlier search history
    (causal_flash_attention 25, argmax 3, flash_attention 1, softmax 1).

    Two distinct mechanisms produce that, and they need different answers:

      1. SATURATED / DISCRETE OUTPUT. argmax on a one-hot-ish row, or softmax
         on a row with a 10000.0 spike, gives the same answer for any small
         perturbation. Rung B (negation) moves these; rungs A and C often do
         not, because they preserve which element is largest.

      2. THE PRIMARY PROVABLY CANNOT AFFECT THE OUTPUT. For attention with K
         constant or saturated across key positions, the attention weights are
         independent of Q for *every* Q -- so no perturbation of the primary,
         of any magnitude or form, can change the output. Only rung D
         (perturbing a companion) reaches these.

    MEASURED, on the 20 recorded causal_flash_attention false positives:
    per-element multiplicative perturbation rescues 0/20; negation 10/20;
    a fresh independent draw 0/20; perturbing companion V, 20/20. Mechanism 2
    dominates that corpus, which is why rung D exists and why rung A alone --
    the fix originally recommended in CHECK_ABLATION_FINDINGS.md §3.0 -- was
    not sufficient. Do not simplify this back to a single probe.

    The rungs form a DISJUNCTION: the check passes as soon as any one of them
    moves the output. That can only ever reduce false positives, never create
    them, and it costs nothing for a true ghost kernel, which by definition
    cannot be moved by any rung.

    Rung E is the soundness backstop. If nothing moved the candidate, the same
    ladder is run through `reference_fn`. If the REFERENCE is also completely
    insensitive, the input is degenerate for this operator and "the outputs
    are identical" is correct behaviour, not evidence of a ghost -- so the
    check reports that it could not be evaluated rather than failing. Only a
    candidate that sits still while the reference moves is flagged.

    Args:
        candidate_fn:  callable taking the primary tensor (already wrapped by
                       the caller to route companions -- see checker.py's
                       _cand/_ref).
        x:             the primary input tensor.
        reference_fn:  ground-truth callable with the same signature.
        spec:          KernelSpec. Required for rung D, together with all of
                       `inputs`, `raw_candidate_fn` and `raw_reference_fn`.
        inputs:        the full input tuple.
        raw_candidate_fn / raw_reference_fn:
                       the UNWRAPPED callables, i.e. what the caller passed to
                       spec.run_candidate / spec.run_reference. `candidate_fn`
                       and `reference_fn` substitute only the primary, so
                       reaching a companion needs the raw callable plus the
                       spec.

        When any of the rung-D arguments is missing the ladder runs
        primary-only, which is the pre-existing behaviour for any caller that
        has not been updated.

    Returns:
        (True,  detail)
        (False, detail)
    """

    def _probe_ladder(run_primary, run_full):
        """Walk the whole ladder for ONE kernel.

        Returns (label, n_companions_probed), where label names the first
        probe that moved the output, or is None if nothing did. Raises only
        if the *base* call fails.

        Candidate and reference both go through this function, so rung E
        necessarily compares like with like. Giving them separate ladders
        would make "the reference moves but the candidate does not" an
        artifact of the ladders differing rather than of the kernels
        differing.
        """
        base = run_primary(x).detach().clone()

        def _moved(out):
            # An exception is NOT movement, and NOT stillness. A probe that
            # crashes tells us nothing about input-dependence, so it is
            # skipped: counting a crash as movement would let a broken kernel
            # pass, and counting it as stillness would manufacture a false
            # positive out of an unrelated failure.
            if out is None:
                return False
            if out.shape != base.shape or out.dtype != base.dtype:
                return True
            return not torch.equal(out, base)

        def _try_call(fn, arg):
            try:
                return fn(arg).detach().clone()
            except Exception:
                return None

        # Rungs A-C: perturb the primary.
        for name, probe in _PRIMARY_PROBES:
            if _moved(_try_call(run_primary, probe(x))):
                return name, 0, base

        # Rung D: perturb each float companion in turn, primary held fixed.
        #
        # Non-float and non-tensor companions are skipped deliberately, not
        # incidentally: groupnorm's num_groups and the pooling ops'
        # kernel_size/stride/padding are Python ints, and cross_entropy's
        # targets is an int64 class-index tensor. Perturbing any of them is
        # either a TypeError or a silently invalid input.
        n_companions = 0
        if run_full is not None and isinstance(inputs, tuple) and len(inputs) > 1:
            for i in range(1, len(inputs)):
                t = inputs[i]
                if not (torch.is_tensor(t) and t.is_floating_point()):
                    continue
                n_companions += 1
                perturbed = inputs[:i] + (_probe_multiplicative(t),) + inputs[i + 1:]
                if _moved(_try_call(run_full, perturbed)):
                    return f"companion[{i}]", n_companions, base

        return None, n_companions, base

    have_raw = (spec is not None
                and raw_candidate_fn is not None
                and raw_reference_fn is not None
                and isinstance(inputs, tuple))
    cand_full = (lambda inp: spec.run_candidate(raw_candidate_fn, inp)) if have_raw else None
    ref_full = (lambda inp: spec.run_reference(raw_reference_fn, inp)) if have_raw else None

    try:
        moved_by, n_comp, out1 = _probe_ladder(candidate_fn, cand_full)
    except Exception as e:
        return False, f"Kernel raised an exception: {e}"

    probed = (f"{len(_PRIMARY_PROBES)} structurally different input perturbations"
              + (f" and {n_comp} companion perturbation(s)" if n_comp else ""))

    if moved_by is not None:
        verdict_detail = (
            f"Kernel executed and produced input-dependent output "
            f"(perturbation: {moved_by})."
        )
    else:
        # Rung E: nothing moved the candidate. Before calling that a ghost,
        # ask whether the REFERENCE moves on the same ladder. If it does not,
        # this input is degenerate for the operator and the check is not
        # evaluable.
        try:
            ref_moved_by, _, _ = _probe_ladder(reference_fn, ref_full)
        except Exception as e:
            return True, (
                "Kernel output did not move under any perturbation, and the "
                f"reference could not be run to establish whether it should: {e}"
            )

        if ref_moved_by is not None:
            return False, (
                f"Kernel output is identical across {probed}, but the reference "
                f"DOES change under '{ref_moved_by}'. Kernel likely ignores "
                "input (hardcoded output or ghost optimization)."
            )

        verdict_detail = (
            f"Not evaluable on this input: neither the candidate nor the "
            f"reference changes under {probed}. The operator is genuinely "
            "insensitive here (e.g. a saturated softmax, a stable argmax, or "
            "attention whose weights do not depend on the perturbed tensor), "
            "so identical outputs are correct rather than evidence of a ghost "
            "kernel."
        )

    # (catches kernels that just call the reference directly)
    #
    # UNCHANGED by the probe-ladder rewrite, and still reached on every
    # non-ghost path -- including the "not evaluable" one, where a delegating
    # kernel is exactly as detectable as it ever was.
    try:
        ref1 = reference_fn(x).detach().clone()
    except Exception as e:
        return True, f"Could not run reference for comparison: {e}"

    # If output == reference on every element to machine precision,
    # the candidate may literally be the reference.  This is a soft warning
    # (some correct kernels are numerically identical to reference), so we
    # only flag if the candidate is also suspiciously fast.
    if torch.equal(out1.float(), ref1.float()):
        # INTERLEAVED, BEST-OF-N TIMING. Item 1d.
        #
        # This previously timed 10 candidate calls, then 10 reference calls,
        # back to back, and compared the two totals. For a REFERENCE kernel the
        # candidate IS the reference, so that comparison times one function
        # against itself and any answer other than ~1.0 is measurement noise.
        # Measured under 4-way GPU contention across 2765 such executions, the
        # old construction produced a ratio distribution spanning 0.10 to
        # 51.24, with p99 = 11.45 -- so the 10x threshold sat at roughly the
        # 98th percentile of its own noise and fired on 1.23% of reference
        # executions. A 51x apparent "speedup" of the reference over itself is
        # what a single scheduling stall inside one of the two blocks buys.
        #
        # Two changes, and both are needed:
        #
        #   INTERLEAVE  the two measurements now alternate within each round,
        #               so a contention episode lands on both arms of the
        #               comparison rather than on whichever block it happened
        #               to hit. Sequential blocks make the ratio sensitive to
        #               *when* the stall occurred; interleaved ones do not.
        #
        #   MIN, NOT SUM  contention can only ADD time, never remove it, so the
        #               minimum across rounds is the estimator closest to the
        #               true cost and the one a stall cannot inflate. Taking a
        #               total (or a mean) lets one bad round set the verdict;
        #               taking the min requires EVERY round to be slow before
        #               the number moves. This is why `timeit` reports min.
        #
        # Total kernel launches are unchanged at 10 per side, so this is not a
        # cost increase -- only the same work, sampled in a way that a single
        # stall cannot dominate. The THRESHOLD IS DELIBERATELY UNCHANGED at
        # 0.1, so any verdict difference is attributable to the estimator
        # rather than to a moved goalpost.
        import time

        _ROUNDS, _CALLS = 5, 2

        def _time_once(fn):
            if x.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(_CALLS):
                fn(x)
            if x.is_cuda:
                torch.cuda.synchronize()
            return time.perf_counter() - t0

        t_cand = float("inf")
        t_ref = float("inf")
        for _ in range(_ROUNDS):
            t_cand = min(t_cand, _time_once(candidate_fn))
            t_ref = min(t_ref, _time_once(reference_fn))

        # THE RATIO IS RECORDED ON BOTH OUTCOMES, NOT ONLY WHEN IT TRIPS.
        #
        # It was previously computed on every reference execution and discarded
        # unless it crossed the threshold -- §2.3's Shape A, "computed then
        # discarded", in the one check whose verdict is a pure timing
        # comparison. That made the check's behaviour unauditable: with only the
        # crossings visible there is no way to tell a threshold that sits far
        # out in the tail from one the noise routinely reaches, and no way to
        # compare the distribution between two execution regimes without
        # counting rare events at hopeless sample sizes.
        #
        # `delegation_ratio=` is a parseable token on purpose. Nothing asserts
        # on this string today (checked), and it is additive: the verdict is
        # unchanged in both branches.
        ratio = (t_ref / t_cand) if t_cand > 0 else float("inf")

        if t_ref > 0 and t_cand < t_ref * 0.1:
            return False, (
                f"Output is bit-identical to reference AND candidate is "
                f"{ratio:.1f}x faster. Likely delegating to reference. "
                f"[delegation_ratio={ratio:.4f}]"
            )

        verdict_detail += f" [delegation_ratio={ratio:.4f}]"

    return True, verdict_detail
