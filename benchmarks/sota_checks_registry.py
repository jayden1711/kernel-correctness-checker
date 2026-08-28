"""
Centralized registry of every correctness check across every SOTA method
this project has been benchmarked against, plus this project's own
checker -- the "common ground" collection point requested because the
checks were scattered across baselines.py, sota_baselines.py, and
verification/layer{1,2,3}_* with no single place to see the whole
landscape side by side.

Deliberately an MVP: one flat list of CheckEntry rows, no framework, no
abstraction beyond what's needed to print a comparison table. This is
meant to be read (via the generated markdown) AND queried (import
CHECKS from Python) -- e.g. to find every check tagged layer="algebraic"
across all methods when designing an integration method, without
re-deriving the landscape from scratch each time.

Source of truth for every row below is the actual code already in this
repo (baselines.py, sota_baselines.py, verification/) -- nothing here is
invented; each `source` field points at exactly where to verify it.

Usage:
    python benchmarks/sota_checks_registry.py
    # writes benchmarks/SOTA_CHECKS_REGISTRY.md
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckEntry:
    method: str            # which system this check belongs to
    check_name: str
    layer: str              # structural | numeric | algebraic
    what_it_tests: str
    methodology: str        # tolerance style / technique
    operator_scope: str     # which operators this applies to
    source: str              # file:function to verify against


CHECKS = [
    # ---- allclose (naive baseline) -----------------------------------
    CheckEntry("allclose", "allclose", "numeric",
               "output matches reference on one random input",
               "fixed tolerance (atol=1e-2, rtol=1e-2)",
               "all",
               "benchmarks/autokernel/files/baselines.py:allclose_gate"),

    # ---- autokernel_gate (AutoKernel, arXiv 2603.21331) --------------
    CheckEntry("autokernel_gate", "smoke_test", "numeric",
               "single default-shape allclose",
               "fixed tolerance",
               "all",
               "benchmarks/autokernel/files/baselines.py:autokernel_gate"),
    CheckEntry("autokernel_gate", "shape_sweep", "numeric",
               "allclose across N random shapes",
               "fixed tolerance",
               "all",
               "benchmarks/autokernel/files/baselines.py:autokernel_gate"),
    CheckEntry("autokernel_gate", "adversarial_stability", "numeric",
               "allclose on large-identical-value / extreme-dynamic-range / "
               "near-zero-variance inputs",
               "fixed tolerance",
               "softmax/layernorm/matmul input patterns modeled",
               "benchmarks/autokernel/files/baselines.py:_adversarial_stability_inputs"),
    CheckEntry("autokernel_gate", "determinism", "structural",
               "two runs on the same input match",
               "bitwise equality (np.array_equal)",
               "all",
               "benchmarks/autokernel/files/baselines.py:autokernel_gate"),

    # ---- gpuemu (Sarkar, arXiv 2606.27396) ---------------------------
    CheckEntry("gpuemu", "boundary_shape", "numeric",
               "allclose across tiny / non-power-of-two / large shape variants",
               "fixed tolerance",
               "all (per-family shape variants)",
               "benchmarks/autokernel/files/sota_baselines.py:gpuemu_boundary_shape_system"),
    CheckEntry("gpuemu", "adversarial_value(extreme)", "numeric",
               "allclose at 1e6x input magnitude",
               "fixed tolerance, no NaN handling",
               "all",
               "benchmarks/autokernel/files/sota_baselines.py:gpuemu_adversarial_value_system"),
    CheckEntry("gpuemu", "adversarial_value(nan_inf)", "numeric",
               "allclose with 1% of input elements set to NaN",
               "fixed tolerance, deliberately naive (no equal_nan) -- this is "
               "the mechanism behind the paper's own reported high FP rate",
               "all",
               "benchmarks/autokernel/files/sota_baselines.py:gpuemu_adversarial_value_system"),

    # ---- propilot (arXiv 2606.06747) ---------------------------------
    CheckEntry("propilot", "identity", "algebraic",
               "A @ I == A",
               "allclose",
               "matmul only",
               "benchmarks/autokernel/files/sota_baselines.py:_sk_identity_matmul"),
    CheckEntry("propilot", "scalar_linearity", "algebraic",
               "c*f(x) == f(c*x)",
               "allclose",
               "matmul, sum_reduction, mean_reduction",
               "benchmarks/autokernel/files/sota_baselines.py:_sk_scalar_linearity_*"),
    CheckEntry("propilot", "permutation_invariance", "algebraic",
               "reduce(permute(x)) == reduce(x)",
               "allclose",
               "sum/mean/max/min_reduction",
               "benchmarks/autokernel/files/sota_baselines.py:_sk_permutation_invariance_reduction"),
    CheckEntry("propilot", "decomposition", "algebraic",
               "reduce(x) == reduce(x[:half]) (+/max/min) reduce(x[half:])",
               "allclose",
               "sum/max/min_reduction",
               "benchmarks/autokernel/files/sota_baselines.py:_sk_decomposition_*"),

    # ---- your_checker: Layer 1 (structural) --------------------------
    CheckEntry("your_checker", "nan_inf", "structural",
               "output is fully finite",
               "boolean",
               "all",
               "verification/layer1_structural/runtime_guards.py:check_nan_inf"),
    CheckEntry("your_checker", "dtype_preserved", "structural",
               "output dtype matches input (or spec-declared expected dtype)",
               "exact",
               "all",
               "verification/layer1_structural/runtime_guards.py:check_dtype_preserved"),
    CheckEntry("your_checker", "ghost_optimization", "structural",
               "kernel actually launches, isn't bypassed by a dead/conditional branch",
               "AST static analysis",
               "all",
               "verification/layer1_structural/ast_analysis.py:check_ghost_optimization"),
    CheckEntry("your_checker", "timing_manipulation", "structural",
               "no missing-sync or off-stream timing tricks",
               "AST static analysis",
               "all",
               "verification/layer1_structural/ast_analysis.py:check_timing_manipulation"),
    CheckEntry("your_checker", "partial_computation", "structural",
               "kernel does most of its own compute, not delegated back to PyTorch",
               "AST static analysis (call-ratio heuristic)",
               "all",
               "verification/layer1_structural/ast_analysis.py:check_partial_computation"),
    CheckEntry("your_checker", "determinism", "structural",
               "repeated runs agree",
               "ADAPTIVE tolerance (atol=1e-4, rtol=1e-4) -- fixed this project "
               "from bitwise equality, which false-positived on legitimate "
               "atomic-add reduction non-associativity",
               "all",
               "verification/layer1_structural/runtime_guards.py:check_determinism"),
    CheckEntry("your_checker", "kernel_executed", "structural",
               "output is input-dependent (not hardcoded / ghost), and isn't "
               "silently just re-calling the reference",
               "runtime comparison across two different inputs + timing check",
               "all",
               "verification/layer1_structural/runtime_guards.py:check_kernel_executed"),
    CheckEntry("your_checker", "tile_coverage", "structural",
               "every output element was actually written by the kernel "
               "(vs. a partial-tile bug leaving some elements untouched)",
               "NaN-sentinel buffer patching",
               "all",
               "verification/layer1_structural/tile_coverage.py"),

    # ---- your_checker: Layer 2 (numeric) -----------------------------
    CheckEntry("your_checker", "output_shape", "numeric",
               "output shape matches reference",
               "exact",
               "all",
               "verification/layer2_numeric_oracle/shape_generalization.py:check_output_shape"),
    CheckEntry("your_checker", "perturbation_tolerance", "numeric",
               "output matches reference within a tolerance DERIVED FROM the "
               "reference's own measured sensitivity to a tiny input "
               "perturbation, not a fixed constant",
               "ADAPTIVE (scale x P95 of reference's own perturbation "
               "response) -- the key differentiator vs. every fixed-tolerance "
               "SOTA baseline above",
               "all",
               "verification/layer2_numeric_oracle/perturbation.py:check_perturbation_tolerance"),
    CheckEntry("your_checker", "cross_shape", "numeric",
               "correctness holds across 5 pre-defined shapes per operator "
               "(incl. non-power-of-two, batch=1, large batch)",
               "fixed tolerance",
               "all (per-operator shape lists)",
               "verification/checker.py:_check_cross_shape"),
    CheckEntry("your_checker", "weight_magnitude", "numeric",
               "correctness under adversarial-magnitude primary input "
               "(large-uniform, large-random, monotone-rows, alternating-sign)",
               "fixed tolerance",
               "all",
               "verification/layer2_numeric_oracle/shape_generalization.py:check_weight_magnitude"),
    CheckEntry("your_checker", "adversarial_inputs", "numeric",
               "per-operator hand-designed adversarial input battery "
               "(e.g. softmax: max-in-last-tile, equal-logits, extreme-range, "
               "non-power-of-two, near-zero-variance -- 5 per operator on "
               "average, unique per spec)",
               "adaptive perturbation tolerance or exact match per variant",
               "per-operator (29 specs, each with its own battery)",
               "verification/specs/*.py:get_adversarial_inputs"),
    CheckEntry("your_checker", "backward_pass", "numeric",
               "gradient correctness vs. reference autograd",
               "fixed tolerance",
               "only where spec.requires_backward=True (none currently)",
               "verification/layer2_numeric_oracle/shape_generalization.py:check_backward_pass"),

    # ---- your_checker: Layer 3 (algebraic) ---------------------------
    # Catalogued by property TYPE (matching Propilot's own "skeleton"
    # framing) rather than one row per operator instantiation -- there
    # are 29 spec files each with 3-5 of these, ~100+ individual checks.
    CheckEntry("your_checker", "zero_mean / unit_variance", "algebraic",
               "normalized output has zero mean / unit variance (gamma=1, beta=0)",
               "fixed tolerance",
               "layernorm, groupnorm, instancenorm, batchnorm",
               "verification/layer3_properties/layernorm_properties.py"),
    CheckEntry("your_checker", "scale_invariance", "algebraic",
               "f(c*x) == f(x) for c>0 (norm ops strip scale before affine)",
               "fixed tolerance",
               "layernorm, rmsnorm, l2norm, groupnorm, instancenorm",
               "verification/layer3_properties/*_properties.py"),
    CheckEntry("your_checker", "shift_invariance", "algebraic",
               "f(x+c) == f(x) for softmax/log_softmax (per-row constant shift)",
               "fixed tolerance",
               "softmax, log_softmax",
               "verification/layer3_properties/softmax_properties.py"),
    CheckEntry("your_checker", "monotonicity", "algebraic",
               "relative order of elements is preserved through the operator",
               "ADAPTIVE (sorted-step / argmax-value comparison, not exact "
               "index equality) -- fixed this project from exact-index "
               "comparison, which false-positived on near-tied elements",
               "softmax, log_softmax",
               "verification/layer3_properties/softmax_properties.py, "
               "log_softmax_properties.py"),
    CheckEntry("your_checker", "affine_correctness / gamma_correctness", "algebraic",
               "f(x, gamma=2, beta=3) == 2*f_normalized(x)+3 -- catches "
               "kernels that ignore or misapply gamma/beta",
               "fixed tolerance",
               "layernorm, rmsnorm",
               "verification/layer3_properties/layernorm_properties.py, "
               "rmsnorm_properties.py"),
    CheckEntry("your_checker", "precision_coercion", "algebraic",
               "fp32 must be meaningfully more accurate than fp16 vs. a "
               "double-precision reference -- catches silent fp16 accumulation",
               "fixed tolerance, fp32-vs-fp16 comparison",
               "most operators",
               "verification/layer3_properties/*_properties.py"),
    CheckEntry("your_checker", "rows_sum_to_one / exp_sums_to_one", "algebraic",
               "softmax/log_softmax row sums to 1 (in probability space)",
               "fixed tolerance",
               "softmax, log_softmax",
               "verification/layer3_properties/softmax_properties.py, "
               "log_softmax_properties.py"),
    CheckEntry("your_checker", "distributivity / scalar_associativity", "algebraic",
               "A@(B+C) == A@B+A@C ; (c*A)@B == c*(A@B)",
               "fixed tolerance (atol=2e-3, rtol=1e-3 -- loosened this "
               "project after false-positiving on ordinary fp32 cancellation "
               "at c=100 scale)",
               "matmul",
               "verification/layer3_properties/matmul_properties.py"),
]


def _synthesis() -> str:
    return """
## Synthesis: what this suggests for an integration method

**Checks unique to your_checker, present in no SOTA baseline above:**
- All of Layer 1's AST-based structural analysis (ghost_optimization,
  timing_manipulation, partial_computation) -- no SOTA baseline reads
  kernel source at all, all of them are purely black-box numeric.
- perturbation_tolerance's ADAPTIVE tolerance (scaled to the reference's
  own measured sensitivity) -- every SOTA baseline above uses a FIXED
  tolerance, which is the direct mechanism behind autokernel_gate's 18%
  and gpuemu adversarial_value's 82% false-positive rates measured on
  the TritonBench corpus.
- tile_coverage's sentinel-buffer patching -- no SOTA baseline detects
  partial-write bugs this way.
- The depth of per-operator algebraic coverage (~100+ checks across 29
  specs) vastly exceeds propilot's 4 generic skeletons applied to 5
  operators here, though propilot's real strength (212 TVM operators via
  LLM-authored skeletons) isn't represented by this project's own
  necessarily-smaller re-implementation.

**Checks SOTA baselines have that your_checker does NOT have a generic
equivalent for (candidates for the "integration" direction):**
- gpuemu's boundary_shape/adversarial_value are GENERIC, automatic input
  fuzzers that need zero per-operator authoring -- your_checker's
  adversarial_inputs are hand-designed per spec (more precise, but only
  exist for operators someone has written them for). A generic fuzzing
  PASS, gated behind the same equal_nan/adaptive-tolerance fixes already
  proven this project, would give automatic baseline coverage on any new
  operator before a hand-written battery exists for it.
- propilot's skeleton-matching (identity / permutation_invariance /
  decomposition / scalar_linearity, applied via operator-family
  applicability rules) is a GENERIC pattern your_checker's algebraic
  layer doesn't have -- it's all hand-written per operator. A generic
  skeleton pass would give automatic Layer-3-equivalent coverage on
  operators without a hand-written spec/properties file yet.

**The integration argument this sets up**: your_checker's fixed
per-operator authoring gives higher precision where it exists (100+
hand-designed checks vs. propilot's 4 generic ones); SOTA methods above
give breadth/automation where hand-authoring hasn't happened yet. A
system that runs your_checker's hand-authored checks where they exist
and falls back to generic fuzzing (gpuemu-style) + generic skeleton
matching (propilot-style) elsewhere would plausibly beat every method
above individually -- worth prototyping as the "integration method"
your advisor's notes call out, once this registry is the shared
reference point for what "generic fallback" actually needs to cover.
"""


def build_markdown() -> str:
    lines = ["# SOTA Checks Registry\n",
             "One row per check, across every method this project has been "
             "benchmarked against. Source of truth is the code cited in "
             "`source`, not this file -- regenerate with "
             "`python benchmarks/sota_checks_registry.py` after any change.\n"]

    lines.append(f"**{len(CHECKS)} checks catalogued across "
                 f"{len(sorted(set(c.method for c in CHECKS)))} methods.**\n")

    lines.append("| Method | Check | Layer | What it tests | Methodology | Scope |")
    lines.append("|---|---|---|---|---|---|")
    for c in CHECKS:
        lines.append(f"| {c.method} | {c.check_name} | {c.layer} | "
                     f"{c.what_it_tests} | {c.methodology} | {c.operator_scope} |")

    lines.append("")
    lines.append(_synthesis())

    return "\n".join(lines)


if __name__ == "__main__":
    md = build_markdown()
    out_path = Path(__file__).parent / "SOTA_CHECKS_REGISTRY.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path} ({len(CHECKS)} checks)")
