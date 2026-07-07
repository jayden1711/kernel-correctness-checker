"""
run_checker_manual.py  Run your three-layer checker on a single
user-provided Triton kernel file.

Usage:
    python run_checker_manual.py path/to/kernel.py --operator softmax
    python run_checker_manual.py path/to/kernel.py --operator layernorm
    python run_checker_manual.py path/to/kernel.py --operator matmul
    python run_checker_manual.py path/to/kernel.py --operator flash_attention

The kernel file should be a raw Python file that defines a callable
function implementing the operator. The function can be named anything
 this script will try common names automatically, or you can specify
the exact name with --func.

Examples of valid kernel files:

    def softmax(x): ...
    def softmax_forward(x): ...
    def forward(x): ...

    # matmul:
    def matmul(A, B): ...

    # flash_attention:
    def flash_attention(Q, K, V): ...

The script runs naive allclose first, then the full three-layer checker,
and prints a detailed report.
"""

import os
import sys
import argparse
import importlib.util
import torch

# Locate kernel-correctness-checker

CHECKER_ROOT = os.path.dirname(os.path.abspath(__file__))

if CHECKER_ROOT not in sys.path:
    sys.path.insert(0, CHECKER_ROOT)

from verification.checker import KernelChecker
from verification.specs.softmax import SoftmaxSpec
from verification.specs.layernorm import LayernormSpec
from verification.specs.matmul import MatmulSpec
from verification.specs.flash_attention import FlashAttentionSpec

# Config

SPEC_MAP = {
    "softmax":         SoftmaxSpec,
    "layernorm":       LayernormSpec,
    "matmul":          MatmulSpec,
    "flash_attention": FlashAttentionSpec,
}

# Candidate function names to try for each operator
FUNC_CANDIDATES = {
    "softmax":         ["softmax", "softmax_forward", "fused_softmax",
                        "triton_softmax", "forward", "run"],
    "layernorm":       ["layernorm", "layer_norm", "layernorm_forward",
                        "triton_layernorm", "forward", "run"],
    "matmul":          ["matmul", "matrix_multiply", "triton_matmul",
                        "gemm", "forward", "run"],
    "flash_attention": ["flash_attention", "flash_attn", "attention",
                        "scaled_dot_product_attention", "forward", "run"],
}

# Default test inputs per operator
def make_test_inputs(operator: str) -> tuple:
    if operator == "softmax":
        return (torch.rand(512, 512, device="cuda"),)
    elif operator == "layernorm":
        return (torch.rand(512, 512, device="cuda"),)
    elif operator == "matmul":
        return (torch.rand(256, 256, device="cuda"),
                torch.rand(256, 256, device="cuda"))
    elif operator == "flash_attention":
        return (torch.rand(128, 64, device="cuda"),
                torch.rand(128, 64, device="cuda"),
                torch.rand(128, 64, device="cuda"))
    raise ValueError(f"Unknown operator: {operator}")


def reference_output(operator: str, inputs: tuple) -> torch.Tensor:
    if operator == "softmax":
        return torch.softmax(inputs[0], dim=1)
    elif operator == "layernorm":
        x = inputs[0]
        return torch.nn.functional.layer_norm(x, [x.shape[-1]])
    elif operator == "matmul":
        return torch.matmul(inputs[0], inputs[1])
    elif operator == "flash_attention":
        Q, K, V = inputs
        d = Q.shape[-1]
        scores = Q @ K.T / (d ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        return weights @ V
    raise ValueError(f"Unknown operator: {operator}")

# Load kernel

def load_kernel(path: str, operator: str, func_name: str | None):
    """
    Dynamically import the kernel file and return the kernel function.
    If func_name is given, use that. Otherwise try FUNC_CANDIDATES.
    Returns (fn, error_str).
    """
    if not os.path.isfile(path):
        return None, f"File not found: {path}"

    spec = importlib.util.spec_from_file_location("_user_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        return None, f"Failed to import {path}: {e}"

    if func_name:
        fn = getattr(mod, func_name, None)
        if fn is None:
            return None, f"Function '{func_name}' not found in {path}."
        if not callable(fn):
            return None, f"'{func_name}' exists but is not callable."
        return fn, None

    # Try candidates
    candidates = FUNC_CANDIDATES[operator]
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn, None

    # Last resort: find any callable that isn't a class or import
    all_callables = [
        name for name in dir(mod)
        if callable(getattr(mod, name))
        and not name.startswith("_")
        and not isinstance(getattr(mod, name), type)
    ]
    if all_callables:
        fn = getattr(mod, all_callables[0])
        print(f"  Note: no standard name found. Using '{all_callables[0]}' "
              f"(first callable). Use --func to specify explicitly.")
        return fn, None

    return None, (
        f"No callable function found in {path}.\n"
        f"Tried: {candidates}\n"
        f"Use --func <name> to specify the function name explicitly."
    )

# Checks

def run_naive_allclose(kernel_fn, operator: str) -> tuple[bool, str]:
    print("  Running naive allclose check...")
    try:
        inputs = make_test_inputs(operator)
        ref = reference_output(operator, inputs)
        out = kernel_fn(*inputs)
        if out is None:
            return False, "kernel returned None"
        if out.shape != ref.shape:
            return False, f"shape mismatch: got {out.shape}, expected {ref.shape}"
        passed = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
        max_err = (out.float() - ref.float()).abs().max().item()
        detail = f"max_err={max_err:.6f}, atol=1e-3"
        return passed, detail
    except Exception as e:
        return False, str(e)


def run_full_checker(kernel_fn, operator: str):
    print("  Running full three-layer checker...")
    spec = SPEC_MAP[operator](name=operator)
    checker = KernelChecker(spec)

    if operator == "softmax":
        x = torch.rand(512, 512, device="cuda")
        ref = lambda inp: torch.softmax(inp, dim=1)
        return checker.run(kernel_fn, None, ref, x)
    elif operator == "layernorm":
        x = torch.rand(512, 512, device="cuda")
        ref = lambda inp: torch.nn.functional.layer_norm(inp, [inp.shape[-1]])
        return checker.run(kernel_fn, None, ref, x)
    elif operator == "matmul":
        A = torch.rand(256, 256, device="cuda")
        B = torch.rand(256, 256, device="cuda")
        ref = lambda a, b: torch.matmul(a, b)
        return checker.run(kernel_fn, None, ref, (A, B))
    elif operator == "flash_attention":
        Q = torch.rand(128, 64, device="cuda")
        K = torch.rand(128, 64, device="cuda")
        V = torch.rand(128, 64, device="cuda")
        d = Q.shape[-1]
        ref = lambda q, k, v: torch.softmax(q @ k.T / (d**0.5), dim=-1) @ v
        return checker.run(kernel_fn, None, ref, (Q, K, V))
    else:
        raise ValueError(f"Unknown operator: {operator}")

# Report

def print_report(path: str, operator: str, func_name_used: str,
                 allclose_pass: bool, allclose_detail: str,
                 checker_result):

    print("\n" + "=" * 65)
    print(f"  Kernel:   {os.path.basename(path)}")
    print(f"  Operator: {operator}")
    print(f"  Function: {func_name_used}")
    print("=" * 65)

    # Naive allclose
    allclose_str = "PASS" if allclose_pass else "FAIL"
    print(f"\n  Naive allclose : {allclose_str}  ({allclose_detail})")

    # Layer-by-layer checker output
    print("\n  Full checker:")
    layer_labels = {
        "nan_inf": "L1", "dtype_preserved": "L1", "ghost_optimization": "L1",
        "timing_manipulation": "L1", "partial_computation": "L1",
        "determinism": "L1", "kernel_executed": "L1", "tile_coverage": "L1",
        "output_shape": "L2", "perturbation_tolerance": "L2",
        "cross_shape": "L2", "backward_pass": "L2",
    }

    failed = []
    for r in checker_result:
        name, passed, detail = r.check_name, r.passed, r.details
        label = layer_labels.get(name, "L3")
        status = "PASS" if passed else "FAIL"
        print(f"    {status}  [{label}] {name:<30}  {detail}")
        if not passed:
            failed.append(f"[{label}] {name}")

    # Verdict
    print("\n" + "=" * 65)
    if not failed:
        print("  Verdict: PASS  kernel appears correct.")
    else:
        print(f"  Verdict: FAIL  {len(failed)} check(s) failed:")
        for f in failed:
            print(f"     {f}")

    # Extra note for the interesting case
    if allclose_pass and failed:
        print("\n  ⚠  This kernel PASSED naive allclose but FAILED deeper")
        print("     verification  exactly the class of subtle bug this")
        print("     checker is designed to catch.")

    print("=" * 65 + "\n")

# Main

def main():
    parser = argparse.ArgumentParser(
        description="Run the three-layer checker on a single Triton kernel file."
    )
    parser.add_argument("kernel_path",
                        help="Path to the kernel Python file.")
    parser.add_argument("--operator", required=True,
                        choices=list(SPEC_MAP.keys()),
                        help="Operator type: softmax | layernorm | matmul | flash_attention")
    parser.add_argument("--func", type=str, default=None,
                        help="Exact function name to use (optional). "
                             "If not given, common names are tried automatically.")
    args = parser.parse_args()

    path = os.path.abspath(args.kernel_path)

    print(f"\nLoading kernel from: {path}")
    kernel_fn, err = load_kernel(path, args.operator, args.func)
    if kernel_fn is None:
        print(f"ERROR: {err}")
        sys.exit(1)

    # Determine what name was used
    func_name_used = args.func or kernel_fn.__name__
    print(f"Found function: {func_name_used}")

    # Naive allclose
    allclose_pass, allclose_detail = run_naive_allclose(kernel_fn, args.operator)

    # Full checker
    try:
        checker_result = run_full_checker(kernel_fn, args.operator)
    except Exception as e:
        print(f"\nERROR running checker: {e}")
        sys.exit(1)

    # Report
    print_report(
        path, args.operator, func_name_used,
        allclose_pass, allclose_detail,
        checker_result,
    )


if __name__ == "__main__":
    main()