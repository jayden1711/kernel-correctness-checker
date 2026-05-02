from dataclasses import dataclass
from typing import Callable, Optional
from verification.layer1_structural.ast_analysis import (
    check_ghost_optimization,
    check_missing_barriers,
    check_timing_manipulation,
    check_partial_computation,
)
from verification.layer1_structural.tile_coverage import check_all_tiles_visited
from verification.layer2_numeric_oracle.perturbation import check_perturbation_tolerance
from verification.layer2_numeric_oracle.shape_generalization import (
    check_cross_shape,
    check_output_shape,
    check_weight_magnitude,
    check_backward_pass,
)
from verification.layer3_algebraic.softmax_properties import (
    check_rows_sum_to_one,
    check_shift_invariance,
    check_monotonicity,
    check_precision_coercion,
)
# TODO: import other kernel properties, after implementing their spec class

@dataclass
class CheckResult:
    passed: bool
    layer: int
    check_name: str
    details: Optional[str] = None

class KernelChecker:
    def __init__(self, spec):
        self.spec = spec

    def run(self, candidate_fn, raw_kernel, reference_fn, x):
        results = []

        # Layer 1: Structural (cheapest, runs first)
        results.append(self._run_check(1, "ghost_optimization",
            lambda: check_ghost_optimization(raw_kernel)))
        results.append(self._run_check(1, "missing_barriers",
            lambda: check_missing_barriers(raw_kernel)))
        results.append(self._run_check(1, "timing_manipulation",
            lambda: check_timing_manipulation(raw_kernel)))
        results.append(self._run_check(1, "partial_computation",
            lambda: check_partial_computation(raw_kernel)))
        results.append(self._run_check(1, "tile_coverage",
            lambda: check_all_tiles_visited(candidate_fn, raw_kernel, x)))

        # Fail fast: don't run expensive layers if structural checks fail
        if any(not r.passed for r in results):
            return results

        # Layer 2: Numeric Oracle
        results.append(self._run_check(2, "output_shape",
            lambda: check_output_shape(candidate_fn, reference_fn, x)))
        results.append(self._run_check(2, "perturbation_tolerance",
            lambda: check_perturbation_tolerance(candidate_fn, reference_fn, x)))
        results.append(self._run_check(2, "cross_shape",
            lambda: check_cross_shape(candidate_fn, reference_fn, self.spec)))
        results.append(self._run_check(2, "weight_magnitude",
            lambda: check_weight_magnitude(candidate_fn, reference_fn, self.spec)))
        results.append(self._run_check(2, "backward_pass",
            lambda: check_backward_pass(candidate_fn, reference_fn, x)))

        # Layer 2: Adversarial
        for name, adversarial_input in self.spec.adversarial_inputs(x):
            results.append(self._run_check(2, f"adversarial_{name}",
                lambda ai=adversarial_input: check_perturbation_tolerance(
                    candidate_fn, reference_fn, ai)))

        if any(not r.passed for r in results):
            return results

        # Layer 3: Algebraic Properties
        for name, prop_fn in self.spec.algebraic_properties:
            results.append(self._run_check(3, name,
                lambda fn=prop_fn: fn(candidate_fn, x)))

        return results

    def _run_check(self, layer, name, fn):
        try:
            passed, *details = fn()
            return CheckResult(
                passed=passed,
                layer=layer,
                check_name=name,
                details=str(details) if details else None
            )
        except Exception as e:
            return CheckResult(
                passed=False,
                layer=layer,
                check_name=name,
                details=f"Exception: {str(e)}"
            )

    def verdict(self, results):
        if all(r.passed for r in results):
            return "PASS"
        failed = [r for r in results if not r.passed]
        return f"FAIL — {len(failed)} check(s) failed: " \
               f"{', '.join(f'[L{r.layer}] {r.check_name}' for r in failed)}"