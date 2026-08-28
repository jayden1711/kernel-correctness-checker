"""
KernelChecker — three-layer verification pipeline.

Usage example (softmax):
    from verification.specs.softmax import get_spec
    from verification.checker import KernelChecker
    from TritonBench.reference.softmax import softmax as reference_fn
    from TritonBench.cheating.softmax.first_tile import softmax as candidate_fn
    from TritonBench.cheating.softmax.first_tile import softmax_kernel_cheat_first_tile as raw_kernel
    import torch

    spec = get_spec()
    checker = KernelChecker(spec)
    x = torch.randn(512, 512, device="cuda")
    results = checker.run(candidate_fn, raw_kernel, reference_fn, x)
    print(checker.summary(results))

Usage example (layernorm):
    inputs = (x, gamma, beta)
    results = checker.run(candidate_fn, raw_kernel, reference_fn, inputs)

Usage example (matmul):
    inputs = (A, B)
    results = checker.run(candidate_fn, raw_kernel, reference_fn, inputs)

Usage example (flash attention):
    inputs = (Q, K, V)
    results = checker.run(candidate_fn, raw_kernel, reference_fn, inputs)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any, List, Dict
import os
import time
import zlib
import torch


# ---------------------------------------------------------------------------
# OPT-IN INSTRUMENTATION (added 2026-08-25). Both switches default OFF and the
# default path is byte-identical to the uninstrumented checker.
#
# KCC_CHECK_TIMING=1
#   Record per-check wall time on CheckResult.duration_ms.
#
#   THIS IS OPT-IN RATHER THAN ALWAYS-ON FOR A REASON, and it is not the cost
#   of perf_counter. Timing a CUDA check honestly requires a
#   torch.cuda.synchronize() on both sides of it, which SERIALISES the
#   pipeline. That changes the very thing being measured: with sync on, every
#   check pays its own launch latency instead of overlapping with the next.
#   So instrumented totals are NOT comparable to uninstrumented totals, and
#   the flag exists so that the benchmark's published latency numbers are
#   never accidentally produced under it. Per-check SHARES remain meaningful;
#   absolute per-check times are an upper bound.
#
# KCC_DISABLE_CHECKS=name1,name2
#   Skip the named checks entirely (they record outcome "skip" and cannot
#   fail). Exists ONLY to run ablation arms; it is not a tuning knob and
#   nothing in the shipped pipeline reads it.
# ---------------------------------------------------------------------------

_TIMING = os.environ.get("KCC_CHECK_TIMING") == "1"
_DISABLED = {n.strip() for n in os.environ.get("KCC_DISABLE_CHECKS", "").split(",") if n.strip()}

# KCC_ABLATION_SEED=1 -- reseed torch's global generator from the CHECK NAME
# before every check.
#
# WHY THIS IS REQUIRED FOR ANY DISABLE ARM, and it is not hygiene. Checks
# consume RNG: check_weight_magnitude calls spec.make_inputs and torch.randn,
# check_perturbation_tolerance draws 20 randn_like per invocation. Skipping a
# check therefore does NOT leave the rest of the pipeline untouched -- every
# later check sees a different RNG stream, and a marginal verdict can flip for
# a reason that has nothing to do with the check that was removed. That is the
# same class of defect as the unseeded-executor finding (SESSION_HANDOFF §7).
#
# Seeding per check name makes each check's draws independent of what ran
# before it, so "with" and "without" arms differ only by the removed check.
# It DOES change absolute verdicts relative to an unseeded run, so the
# baseline arm must also run with it -- compare arms to each other, never to
# an unseeded run.
_ABLATION_SEED = os.environ.get("KCC_ABLATION_SEED") == "1"


def _reseed(name):
    if _ABLATION_SEED:
        # zlib.crc32, NOT hash(). Python randomises str hashing per process
        # unless PYTHONHASHSEED is pinned, so hash() would give every ablation
        # arm a DIFFERENT per-check seed and the arms would not be comparable
        # -- silently, and in exactly the direction that manufactures spurious
        # verdict movement between arms.
        torch.manual_seed(zlib.crc32(name.encode()) % (2 ** 31))


def _sync():
    if _TIMING and torch.cuda.is_available():
        torch.cuda.synchronize()


def _input_stats(t):
    """Numeric-regime fingerprint of an adversarial input. Mirrors
    shape_generalization._tensor_stats so the per-spec adversarial battery and
    check_weight_magnitude's variants can be compared on the tensors they
    actually feed the kernel, not on their names."""
    if not _TIMING:
        return None
    try:
        f = t.detach().float()
        return {"shape": list(t.shape), "dtype": str(t.dtype),
                "min": f.min().item(), "max": f.max().item(),
                "absmax": f.abs().max().item(),
                "mean": f.mean().item(), "std": f.std().item() if f.numel() > 1 else 0.0}
    except Exception:
        return None

from verification.layer1_structural.ast_analysis import (
    check_ghost_optimization,
    check_missing_barriers,
    check_timing_manipulation,
    check_partial_computation,
)
from verification.layer1_structural.tile_coverage import (
    check_all_tiles_visited,
    check_all_tiles_visited_generic,
)
from verification.layer1_structural.runtime_guards import (
    check_nan_inf,
    check_dtype_preserved,
    check_determinism,
    check_kernel_executed,
)
from verification.layer2_numeric_oracle.perturbation import check_perturbation_tolerance
from verification.layer2_numeric_oracle.shape_generalization import (
    check_output_shape,
    check_weight_magnitude,
    check_backward_pass,
)


@dataclass
class CheckResult:
    passed: bool
    layer: int
    check_name: str
    details: Optional[str] = None
    # ADDITIVE, both default None so every existing consumer is unaffected.
    # duration_ms is populated only under KCC_CHECK_TIMING=1.
    # subchecks carries a check's optional third return element (currently
    # only check_weight_magnitude emits one); it was previously discarded by
    # _run_check, which is why per-variant attribution did not exist.
    duration_ms: Optional[float] = None
    subchecks: Optional[List[Dict]] = None
    # scope_flags is populated only under KCC_SCOPE_DETECT=1. It records that
    # this check ran outside the regime its guarantee was verified in.
    #
    # IT IS METADATA ABOUT THE CHECK, NOT ABOUT THE KERNEL, and nothing in
    # KernelChecker reads it. `passed` is computed before it is attached and is
    # never revisited -- so turning the detector on cannot move a verdict, by
    # construction rather than by convention. See scope_detect.py for why that
    # is the design and what would have to be measured before it changed.
    scope_flags: Optional[List[Dict]] = None


class KernelChecker:
    def __init__(self, spec):
        self.spec = spec

    def run(
        self,
        candidate_fn: Callable,
        raw_kernel,
        reference_fn: Callable,
        inputs: Any,
    ):
        """
        Run all three verification layers in order.

        Args:
            candidate_fn:  Python wrapper around the custom Triton kernel.
            raw_kernel:    The @triton.jit function for static analysis.
            reference_fn:  Ground-truth implementation.
            inputs:        Tensor or tuple of tensors matching the spec.
        """
        results = []
        spec = self.spec
        primary = spec.primary_input(inputs)

        # Thin wrappers that correctly route perturbed primary input
        # through the spec so multi-input kernels (layernorm, matmul, attn)
        # still receive all their required tensors.
        def _cand(x):
            if isinstance(inputs, tuple):
                new_inputs = (x,) + inputs[1:]
            else:
                new_inputs = x
            return spec.run_candidate(candidate_fn, new_inputs)

        def _ref(x):
            if isinstance(inputs, tuple):
                new_inputs = (x,) + inputs[1:]
            else:
                new_inputs = x
            return spec.run_reference(reference_fn, new_inputs)

        # Sentinel guards — run first, abort immediately on failure
        results.append(self._run_check(1, "nan_inf",
            lambda: check_nan_inf(_cand, primary)))
        results.append(self._run_check(1, "dtype_preserved",
            lambda: check_dtype_preserved(_cand, primary, expected_dtype=spec.output_dtype)))

        if any(not r.passed for r in results):
            return results

        # Layer 1: Structural
        results.append(self._run_check(1, "ghost_optimization",
            lambda: check_ghost_optimization(candidate_fn)))
        results.append(self._run_check(1, "timing_manipulation",
            lambda: check_timing_manipulation(candidate_fn)))
        results.append(self._run_check(1, "partial_computation",
            lambda: check_partial_computation(candidate_fn)))
        results.append(self._run_check(1, "determinism",
            lambda: check_determinism(_cand, primary)))
        # spec/inputs/raw_* enable the companion-perturbation rung: _cand and
        # _ref substitute only the primary, so reaching a companion tensor
        # needs the unwrapped callables plus the spec. Without them the check
        # cannot clear the false positives on operators whose output does not
        # depend on the primary at all (attention with constant K/V).
        results.append(self._run_check(1, "kernel_executed",
            lambda: check_kernel_executed(
                _cand, primary, _ref,
                spec=spec, inputs=inputs,
                raw_candidate_fn=candidate_fn, raw_reference_fn=reference_fn)))


        results.append(self._run_check(1, "tile_coverage_structural",
            lambda: check_all_tiles_visited_generic(spec, candidate_fn, inputs)))
        # GATED to spec.name == "softmax": this check was built and
        # validated for softmax's output semantics specifically (its own
        # name says "positivity"). The old trigger condition
        # (primary.dim() == 2 and not tuple) fires on ANY 2D single-tensor
        # operator -- which now also includes log_softmax, sum/mean/max/
        # min_reduction, l1norm, l2norm, argmax, argmin -- none of which
        # share softmax's invariants (reductions collapse a dimension
        # entirely; l1norm/l2norm output is signed, not positive-summing).
        # Confirmed via a real run_checker.py run: this produced a garbage
        # "-1" sentinel FAIL for the four reduction operators and a
        # plausible-looking but almost certainly wrong "columns written"
        # FAIL for l1norm/l2norm -- neither related to any injected bug,
        # both would fire identically on a CORRECT kernel for those
        # operators. Same root cause as the Layer-3 shape-guessing
        # collision fixed via kernelbench_operator_registry -- shape alone
        # doesn't identify the operator, so use identity instead of shape.
        if primary.dim() == 2 and not isinstance(inputs, tuple) and spec.name == "softmax":
            results.append(self._run_check(1, "tile_coverage_softmax_positivity",
                lambda rk=raw_kernel: check_all_tiles_visited(candidate_fn, rk, primary)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if any(not r.passed for r in results):
            return results

        # Layer 2: Algebraic Properties -- runs BEFORE numeric (see below)
        for prop_name, prop_fn in spec.algebraic_properties:
            results.append(self._run_check(2, prop_name,
                lambda fn=prop_fn: fn(candidate_fn, inputs)))

        if any(not r.passed for r in results):
            return results

        # Layer 3: Numeric Oracle -- RUNS LAST (reordered 2026-08-20)
        #
        # Numeric is by far the most expensive layer: warm p50 15.71ms
        # against algebraic 1.17ms and structural 3.97ms. Because the
        # layers short-circuit, running it last means it is only paid for
        # when the two cheap layers have BOTH failed to catch the bug.
        #
        # The catch sets are nested -- structural (4 of 40) subset of
        # algebraic (18) subset of numeric (40) -- so reordering can never
        # change a VERDICT, only which layer reports the catch first.
        # That containment is what makes this safe, and it is asserted in
        # tests/instrumentation/check_layer_order.py.
        #
        # Benefit is bounded, not universal: only the 14 mutants caught by
        # algebraic-but-not-structural skip numeric entirely. Correct
        # kernels pass every layer, so they see NO speedup at all, and the
        # 22 numeric-only mutants pay algebraic first (+1.17ms).
        results.append(self._run_check(3, "output_shape",
            lambda: check_output_shape(_cand, _ref, primary)))
        # op_name/companions are inert unless KCC_STRUCTURAL_L=1; they carry
        # the operator identity and the non-primary tensors (gamma, B, targets,
        # K/V) that the closed-form Jacobian needs. `_companions` mirrors
        # exactly what _ref substitutes -- the primary is replaced, the rest
        # ride along -- so the formula sees the same operands the kernel does.
        _companions = tuple(inputs[1:]) if isinstance(inputs, tuple) else ()
        results.append(self._run_check(3, "perturbation_tolerance",
            lambda: check_perturbation_tolerance(
                _cand, _ref, primary, batch_samples=spec.batch_samples,
                op_name=spec.name, companions=_companions)))
        results.append(self._run_check(3, "cross_shape",
            lambda: _check_cross_shape(candidate_fn, reference_fn, spec)))
        results.append(self._run_check(3, "weight_magnitude",
            lambda: check_weight_magnitude(candidate_fn, reference_fn, spec)))


        if spec.requires_backward:
            results.append(self._run_check(3, "backward_pass",
                lambda: check_backward_pass(_cand, _ref, primary)))

        # Adversarial inputs
        try:
            adversarial_pairs = spec.get_adversarial_inputs(inputs)
        except Exception as e:
            adversarial_pairs = []
            results.append(CheckResult(
                passed=False, layer=3,
                check_name="adversarial_setup",
                details=f"Could not generate adversarial inputs: {e}",
            ))

        for name, adv_inputs in adversarial_pairs:
            adv_primary = spec.primary_input(adv_inputs)
            # Fingerprint the tensor this variant actually feeds the kernel.
            # Attached to the CheckResult below under KCC_CHECK_TIMING=1 only.
            _adv_stats = _input_stats(adv_primary)

            def _adv_cand(x, ai=adv_inputs):
                new_inputs = (x,) + ai[1:] if isinstance(ai, tuple) else x
                return spec.run_candidate(candidate_fn, new_inputs)

            def _adv_ref(x, ai=adv_inputs):
                new_inputs = (x,) + ai[1:] if isinstance(ai, tuple) else x
                return spec.run_reference(reference_fn, new_inputs)

            # Discrete/index outputs (argmax, argmin -- spec.output_dtype
            # declared) route through exact equality, not adaptive
            # perturbation tolerance. CONFIRMED via a real run: adaptive
            # tolerance is self-defeating here -- it scales with how much
            # the REFERENCE itself wobbles under tiny perturbation, which
            # is exactly what's large near a tie, so a stronger tie-break
            # trigger simultaneously makes the tolerance looser in lockstep
            # and never actually catches a wrong tie-break convention. An
            # index is either right or wrong; there's no "close enough."
            if spec.output_dtype is not None:
                _r = self._run_check(3, f"adversarial_{name}",
                    lambda c=_adv_cand, r=_adv_ref, ap=adv_primary:
                        _check_exact_match(c, r, ap))
                if _adv_stats is not None:
                    # APPEND -- check_perturbation_tolerance may already have
                    # populated subchecks with its sensitivity vector under
                    # KCC_RECORD_SENSITIVITIES=1. Replacing would silently drop it.
                    _r.subchecks = (_r.subchecks or []) + [
                        {"name": name, "input_stats": _adv_stats,
                         "comparator": "exact_match"}]
                results.append(_r)
            else:
                _adv_companions = (tuple(adv_inputs[1:])
                                   if isinstance(adv_inputs, tuple) else ())
                _r = self._run_check(3, f"adversarial_{name}",
                    lambda c=_adv_cand, r=_adv_ref, ap=adv_primary,
                           ac=_adv_companions:
                        check_perturbation_tolerance(
                            c, r, ap, batch_samples=spec.batch_samples,
                            op_name=spec.name, companions=ac))
                if _adv_stats is not None:
                    _r.subchecks = (_r.subchecks or []) + [
                        {"name": name, "input_stats": _adv_stats,
                         "comparator": "perturbation_tolerance"}]
                results.append(_r)

        return results

    # Helpers

    def _run_check(self, layer: int, name: str, fn: Callable) -> CheckResult:
        # Ablation hook -- see the KCC_DISABLE_CHECKS note at module top.
        # A disabled check reports passed=True so it cannot manufacture a
        # catch; the arm is "this check is absent", not "this check failed".
        if name in _DISABLED:
            return CheckResult(passed=True, layer=layer, check_name=name,
                               details="skipped -- KCC_DISABLE_CHECKS", duration_ms=None)
        _reseed(name)
        _sync()
        t0 = time.perf_counter() if _TIMING else None
        try:
            result = fn()
            subs = None
            scopes = None
            if isinstance(result, (list, tuple)):
                passed = bool(result[0])
                details = str(result[1]) if len(result) > 1 else None
                # Third element, when present, is per-variant attribution.
                # Previously dropped on the floor here.
                if len(result) > 2 and isinstance(result[2], list):
                    subs = result[2]
                    # Promote scope records out of the shared third element
                    # into their own typed field. Done here rather than by
                    # giving check_perturbation_tolerance a fourth return
                    # element, so the wire format every existing caller
                    # unpacks stays a 2- or 3-tuple.
                    scopes = [d for d in subs
                              if isinstance(d, dict)
                              and d.get("kind") == "scope_divergence"] or None
            else:
                passed = bool(result)
                details = None
            _sync()
            return CheckResult(passed=passed, layer=layer,
                               check_name=name, details=details,
                               duration_ms=(1000 * (time.perf_counter() - t0)) if _TIMING else None,
                               subchecks=subs, scope_flags=scopes)
        except Exception as e:
            _sync()
            return CheckResult(
                passed=False, layer=layer, check_name=name,
                details=f"{type(e).__name__}: {e}",
                duration_ms=(1000 * (time.perf_counter() - t0)) if _TIMING else None,
            )

    def verdict(self, results) -> str:
        if all(r.passed for r in results):
            return "PASS"
        failed = [r for r in results if not r.passed]
        items = ", ".join(f"[L{r.layer}] {r.check_name}" for r in failed)
        return f"FAIL — {len(failed)} check(s) failed: {items}"

    def summary(self, results) -> str:
        lines = []
        current_layer = None
        for r in results:
            if r.layer != current_layer:
                current_layer = r.layer
                lines.append(f"\n  Layer {r.layer}:")
            status = "PASS" if r.passed else "FAIL"
            line = f"    {status} {r.check_name}"
            if r.details:
                line += f"  — {r.details}"
            lines.append(line)
        lines.append(f"\nVerdict: {self.verdict(results)}")
        return "\n".join(lines)


# Cross-shape check — uses spec.make_inputs so it handles all kernel types

def _check_cross_shape(
    candidate_fn: Callable,
    reference_fn: Callable,
    spec,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> tuple:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    failures = []
    # Per-shape outcomes, returned as an optional THIRD element. This check
    # collapses up to 5 shapes into one pass/fail, which hides which shape
    # actually caught a mutant -- exactly the granularity the per-check
    # ablation needs. Callers that only unpack [0] and [1]
    # (KernelChecker._run_check, checker_adapter._try) are unaffected.
    subs = []

    for shape in spec.valid_shapes:
        try:
            inputs = spec.make_inputs(shape, device, dtype)
            ref_out = spec.run_reference(reference_fn, inputs)
            cand_out = spec.run_candidate(candidate_fn, inputs)
        except Exception as e:
            failures.append(f"shape={shape}: exception — {e}")
            # "error", not "fail": the check crashed rather than detecting a
            # disagreement, so crediting it with a catch would misattribute.
            subs.append({"name": f"shape={shape}", "outcome": "error",
                         "detail": f"{type(e).__name__}: {e}"})
            continue

        if cand_out.shape != ref_out.shape:
            failures.append(f"shape={shape}: shape mismatch "
                            f"{tuple(cand_out.shape)} vs {tuple(ref_out.shape)}")
            subs.append({"name": f"shape={shape}", "outcome": "fail",
                         "detail": f"shape mismatch {tuple(cand_out.shape)} "
                                   f"vs {tuple(ref_out.shape)}"})
            continue

        if not torch.allclose(cand_out.float(), ref_out.float(), atol=atol, rtol=rtol):
            max_err = (cand_out.float() - ref_out.float()).abs().max().item()
            failures.append(f"shape={shape}: max_err={max_err:.6f}")
            subs.append({"name": f"shape={shape}", "outcome": "fail",
                         "detail": f"max_err={max_err:.6f}"})
        else:
            subs.append({"name": f"shape={shape}", "outcome": "pass",
                         "detail": None})

    if failures:
        return False, "Cross-shape failures: " + "; ".join(failures), subs
    return True, f"Cross-shape passed on {len(spec.valid_shapes)} shapes.", subs


def _check_exact_match(
    candidate_fn: Callable,
    reference_fn: Callable,
    x: torch.Tensor,
) -> tuple:
    """
    For discrete/index-valued outputs (spec.output_dtype declared) --
    exact equality, no tolerance. An index is either right or wrong;
    "close" has no meaning the way it does for a continuous value, and
    (confirmed via a real run) adaptive perturbation tolerance actively
    fails here since it scales with the reference's own instability near
    ties, which defeats any adversarial trigger that leans into a tie.
    """
    try:
        cand_out = candidate_fn(x)
        ref_out = reference_fn(x)
    except Exception as e:
        return False, f"Exception during exact-match check: {e}"

    if cand_out.shape != ref_out.shape:
        return False, f"Shape mismatch: {tuple(cand_out.shape)} vs {tuple(ref_out.shape)}"

    if torch.equal(cand_out, ref_out):
        return True, "Exact match with reference."

    n_diff = (cand_out != ref_out).sum().item()
    return False, f"{n_diff}/{cand_out.numel()} element(s) differ from reference exactly."
