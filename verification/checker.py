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

from dataclasses import dataclass
from typing import Callable, Optional, Any
import torch

from verification.layer1_structural.ast_analysis import (
    check_ghost_optimization,
    check_missing_barriers,
    check_timing_manipulation,
    check_partial_computation,
)
from verification.layer1_structural.tile_coverage import check_all_tiles_visited
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
                # Replace primary input (first element) with perturbed x
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
            lambda: check_dtype_preserved(_cand, primary)))

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
        results.append(self._run_check(1, "kernel_executed",
            lambda: check_kernel_executed(_cand, primary, _ref)))

        if primary.dim() == 2 and not isinstance(inputs, tuple):
            results.append(self._run_check(1, "tile_coverage",
                lambda rk=raw_kernel: check_all_tiles_visited(candidate_fn, rk, primary)))
             # Reset CUDA context after triton-viz tracing to prevent state corruption
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        if any(not r.passed for r in results):
            return results

        # Layer 2: Numeric Oracle
        results.append(self._run_check(2, "output_shape",
            lambda: check_output_shape(_cand, _ref, primary)))
        results.append(self._run_check(2, "perturbation_tolerance",
            lambda: check_perturbation_tolerance(_cand, _ref, primary)))
        results.append(self._run_check(2, "cross_shape",
            lambda: _check_cross_shape(candidate_fn, reference_fn, spec)))

        if spec.requires_backward:
            results.append(self._run_check(2, "backward_pass",
                lambda: check_backward_pass(_cand, _ref, primary)))

        # Adversarial inputs
        try:
            adversarial_pairs = spec.get_adversarial_inputs(inputs)
        except Exception as e:
            adversarial_pairs = []
            results.append(CheckResult(
                passed=False, layer=2,
                check_name="adversarial_setup",
                details=f"Could not generate adversarial inputs: {e}",
            ))

        for name, adv_inputs in adversarial_pairs:
            adv_primary = spec.primary_input(adv_inputs)
            results.append(self._run_check(2, f"adversarial_{name}",
                lambda ai=adv_inputs, ap=adv_primary: check_perturbation_tolerance(
                    lambda x: spec.run_candidate(candidate_fn, ai),
                    lambda x: spec.run_reference(reference_fn, ai),
                    ap,
                )))

        if any(not r.passed for r in results):
            return results

        # Layer 3: Algebraic Properties
        for prop_name, prop_fn in spec.algebraic_properties:
            results.append(self._run_check(3, prop_name,
                lambda fn=prop_fn: fn(candidate_fn, inputs)))

        return results

    # Helpers

    def _run_check(self, layer: int, name: str, fn: Callable) -> CheckResult:
        try:
            result = fn()
            if isinstance(result, (list, tuple)):
                passed = bool(result[0])
                details = str(result[1]) if len(result) > 1 else None
            else:
                passed = bool(result)
                details = None
            return CheckResult(passed=passed, layer=layer,
                               check_name=name, details=details)
        except Exception as e:
            return CheckResult(
                passed=False, layer=layer, check_name=name,
                details=f"{type(e).__name__}: {e}",
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

    for shape in spec.valid_shapes:
        try:
            inputs = spec.make_inputs(shape, device, dtype)
            ref_out = spec.run_reference(reference_fn, inputs)
            cand_out = spec.run_candidate(candidate_fn, inputs)
        except Exception as e:
            failures.append(f"shape={shape}: exception — {e}")
            continue

        if cand_out.shape != ref_out.shape:
            failures.append(f"shape={shape}: shape mismatch "
                            f"{tuple(cand_out.shape)} vs {tuple(ref_out.shape)}")
            continue

        if not torch.allclose(cand_out.float(), ref_out.float(), atol=atol, rtol=rtol):
            max_err = (cand_out.float() - ref_out.float()).abs().max().item()
            failures.append(f"shape={shape}: max_err={max_err:.6f}")

    if failures:
        return False, "Cross-shape failures: " + "; ".join(failures)
    return True, f"Cross-shape passed on {len(spec.valid_shapes)} shapes."