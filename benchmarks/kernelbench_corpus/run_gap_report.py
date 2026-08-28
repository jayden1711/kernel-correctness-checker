"""
Gap report for the hand-authored KernelBench mutant corpus (4 operators:
softmax, swish, gelu, matmul -- see problems/ and candidates/).

Unlike run_kernelbench_eval.py (which scans a directory of many LLM-
generated candidates via iter_pairs()), this corpus is small and
explicit, so the pairing is just a literal list below rather than a
directory-scanning convention.

REQUIRES A REAL GPU: every candidate compiles and launches an actual
CUDA kernel via torch.utils.cpp_extension.load_inline -- there is no
CPU path. None of this has been compiled or executed; it was written
and hand-verified against standard CUDA reduction/matmul patterns on a
machine with no GPU and no CUDA toolchain, not runtime-tested.

    python run_gap_report.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verification.kernel_adapter import run_with_timeout

_HERE = Path(__file__).parent
_PROBLEMS = _HERE / "problems"
_CANDIDATES = _HERE / "candidates"

# (op, mutant_name, problem_file, correct_candidate_file, mutant_candidate_file)
CORPUS = [
    ("softmax", "first_tile", "23_Softmax.py",
     "softmax_correct.py", "softmax_mutant_first_tile.py"),
    ("swish", "linear_sigmoid_approx", "25_Swish.py",
     "swish_correct.py", "swish_mutant_linear_sigmoid.py"),
    ("gelu", "sigmoid_approx", "26_GELU_.py",
     "gelu_correct.py", "gelu_mutant_sigmoid_approx.py"),
    ("matmul", "partial_k_reduct", "1_Square_matrix_multiplication_.py",
     "matmul_correct.py", "matmul_mutant_partial_k.py"),
    ("relu", "leaky", "19_ReLU.py",
     "relu_correct.py", "relu_mutant_leaky.py"),
    # unstable_exp was originally designed as a gap case (matches allclose
    # at base scale, only diverges once fp32 exp() overflows ~x>88). CONFIRMED
    # via a real run this DOESN'T show as a gap in practice: unlike
    # verification/checker.py's KernelChecker (used for TritonBench), this
    # kernel_adapter.py pathway has no weight_magnitude-style large-magnitude
    # probe, so nothing here reaches the overflow scale -- an honest, traced
    # limitation of this pathway, not a bug. Kept in the corpus as a real
    # (if currently missed) mutant; see the session's own notes on this.
    ("sigmoid", "unstable_exp", "21_Sigmoid.py",
     "sigmoid_correct.py", "sigmoid_mutant_unstable_exp.py"),
    ("tanh", "hard_approx", "22_Tanh.py",
     "tanh_correct.py", "tanh_mutant_hard_approx.py"),
    ("layernorm", "skip_mean_subtract", "40_LayerNorm.py",
     "layernorm_correct.py", "layernorm_mutant_skip_mean_subtract.py"),
    ("leaky_relu", "wrong_slope", "20_LeakyReLU.py",
     "leaky_relu_correct.py", "leaky_relu_mutant_wrong_slope.py"),
    ("elu", "missing_minus_one", "31_ELU.py",
     "elu_correct.py", "elu_mutant_missing_minus_one.py"),
    ("selu", "missing_scale", "27_SELU_.py",
     "selu_correct.py", "selu_mutant_missing_scale.py"),
    ("hardsigmoid", "wrong_divisor", "28_HardSigmoid.py",
     "hardsigmoid_correct.py", "hardsigmoid_mutant_wrong_divisor.py"),
    ("softplus", "linear_approx", "29_Softplus.py",
     "softplus_correct.py", "softplus_mutant_linear_approx.py"),
    ("softsign", "wrong_denom", "30_Softsign.py",
     "softsign_correct.py", "softsign_mutant_wrong_denom.py"),
    ("hardtanh", "wrong_bounds", "32_HardTanh.py",
     "hardtanh_correct.py", "hardtanh_mutant_wrong_bounds.py"),
]


def _failing_checks(result):
    """Which check(s) failed, and why -- AdapterResult.trial_checks has
    this (each trial's {check_name: (passed, detail)} dict), but nothing
    upstream of this function was pulling it out; only the aggregate
    checker_pass bool was being reported, which tells you THAT something
    failed but not what."""
    failures = {}
    for trial in result.trial_checks:
        for name, val in trial.get("checks", {}).items():
            if isinstance(val, (list, tuple)) and val[0] is False:
                failures.setdefault(name, val[1])
    return failures


def _run_one(op, mutant_name, problem_file, candidate_file, is_mutant, n_trials, timeout):
    problem_path = str(_PROBLEMS / problem_file)
    candidate_src = (_CANDIDATES / candidate_file).read_text(encoding="utf-8")

    t0 = time.perf_counter()
    result = run_with_timeout(problem_path, candidate_src, n_trials=n_trials, timeout_seconds=timeout)
    dt = time.perf_counter() - t0

    return {
        "op": op,
        "mutant_name": mutant_name,
        "is_mutant": is_mutant,
        "candidate_file": candidate_file,
        "load_error": result.load_error,
        "resolved_operator": result.resolved_operator,
        "candidate_format": result.candidate_format,
        "allclose_pass": result.allclose_pass,
        "checker_pass": result.checker_pass,
        "is_gap": result.is_gap,
        "failing_checks": _failing_checks(result),
        "wall_time_s": round(dt, 2),
    }


def main():
    n_trials = 5
    timeout = 120

    rows = []
    for op, mutant_name, problem_file, correct_file, mutant_file in CORPUS:
        print(f"\n=== {op}/{mutant_name} ===")

        print(f"  reference candidate ({correct_file}) ...")
        ref_row = _run_one(op, mutant_name, problem_file, correct_file,
                            is_mutant=False, n_trials=n_trials, timeout=timeout)
        rows.append(ref_row)
        print(f"    resolved_operator={ref_row['resolved_operator']} "
              f"checker_pass={ref_row['checker_pass']} "
              f"load_error={ref_row['load_error']}")
        if ref_row["failing_checks"]:
            for name, detail in ref_row["failing_checks"].items():
                print(f"      FAILED [{name}]: {detail}")

        print(f"  mutant candidate ({mutant_file}) ...")
        mut_row = _run_one(op, mutant_name, problem_file, mutant_file,
                            is_mutant=True, n_trials=n_trials, timeout=timeout)
        rows.append(mut_row)
        print(f"    resolved_operator={mut_row['resolved_operator']} "
              f"allclose_pass={mut_row['allclose_pass']} "
              f"checker_pass={mut_row['checker_pass']} "
              f"is_gap={mut_row['is_gap']} "
              f"load_error={mut_row['load_error']}")
        if mut_row["failing_checks"]:
            for name, detail in mut_row["failing_checks"].items():
                print(f"      FAILED [{name}]: {detail}")

    # Summary: false positives on the reference, catches on the mutant,
    # and the headline "gap" number -- allclose passed but checker caught it.
    n_ref = sum(1 for r in rows if not r["is_mutant"])
    n_ref_fp = sum(1 for r in rows if not r["is_mutant"] and r["checker_pass"] is False)
    n_mut = sum(1 for r in rows if r["is_mutant"])
    n_mut_caught = sum(1 for r in rows if r["is_mutant"] and r["checker_pass"] is False)
    n_gap = sum(1 for r in rows if r["is_mutant"] and r["is_gap"])

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"  reference false-positive rate: {n_ref_fp}/{n_ref}")
    print(f"  mutant catch rate:              {n_mut_caught}/{n_mut}")
    print(f"  GAP (allclose passed, checker caught it): {n_gap}/{n_mut}")
    for r in rows:
        if r["load_error"]:
            print(f"  [LOAD ERROR] {r['op']}/{r['mutant_name']} "
                  f"({'mutant' if r['is_mutant'] else 'ref'}): {r['load_error']}")

    out_path = _HERE / "kernelbench_corpus_gap_report.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
