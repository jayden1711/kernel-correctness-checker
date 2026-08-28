"""
run_kernelbench_eval.py

Bridge script: run the three-layer checker against KernelBench problems +
LLM-generated candidate kernels, and report where allclose passes but the
checker fails (the gap the paper is claiming exists).

ASSUMED DIRECTORY LAYOUT -- confirm and edit iter_pairs() before trusting
output:

    KERNELBENCH_ROOT/
        KernelBench/level1/1_Square_matmul.py      <- reference problems
        KernelBench/level2/...
    CANDIDATES_ROOT/
        level1/1_Square_matmul/sample_0.py          <- LLM-generated ModelNew

This mirrors KernelBench's own generation-script layout, but has NOT been
run against your actual runs/ directory. If your generation script wrote
a different structure (JSON manifest, different sample naming, flat
directory, etc.), fix iter_pairs() -- everything past that point is
layout-independent.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')

from verification.kernel_adapter import run_with_timeout


def iter_pairs(kernelbench_root: str, candidates_root: str):
    """
    Yields (problem_path, candidate_path) pairs. EDIT THIS to match your
    actual runs/ layout -- see module docstring.
    """
    kb_root = Path(kernelbench_root) / "KernelBench"
    cand_root = Path(candidates_root)

    for level_dir in sorted(kb_root.glob("level*")):
        level_name = level_dir.name
        for problem_file in sorted(level_dir.glob("*.py")):
            problem_stem = problem_file.stem
            candidate_dir = cand_root / level_name / problem_stem
            if not candidate_dir.exists():
                continue
            for candidate_file in sorted(candidate_dir.glob("*.py")):
                yield str(problem_file), str(candidate_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernelbench-root", required=True)
    parser.add_argument("--candidates-root", required=True)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out", default="kernelbench_gap_report.json")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap number of (problem, candidate) pairs -- use for a smoke "
             "test before committing to a full run",
    )
    args = parser.parse_args()

    pairs = list(iter_pairs(args.kernelbench_root, args.candidates_root))
    if args.limit:
        pairs = pairs[:args.limit]

    if not pairs:
        print(
            "No (problem, candidate) pairs found -- check iter_pairs() "
            "against your actual layout before assuming this means "
            "'no bugs'."
        )
        return

    print(f"Running checker against {len(pairs)} candidate kernels...")

    report = {
        "total": len(pairs),
        "load_errors": 0,
        "allclose_pass": 0,
        "checker_pass": 0,
        "gaps": [],  # allclose passed, checker failed -- the paper's headline number
        "suspected_leaks": [],
        "details": [],
    }

    for i, (problem_path, candidate_path) in enumerate(pairs):
        with open(candidate_path, "r", encoding="utf-8") as f:
            candidate_src = f.read()

        result = run_with_timeout(
            problem_path, candidate_src,
            n_trials=args.n_trials, timeout_seconds=args.timeout,
        )
        result.candidate_file = candidate_path

        status = (
            "LOAD_ERROR" if result.load_error else
            "GAP" if result.is_gap else
            "CAUGHT" if result.checker_pass is False else
            "PASS"
        )
        print(f"  [{i+1}/{len(pairs)}] {problem_path} :: {candidate_path} -> {status}")

        if result.load_error:
            report["load_errors"] += 1
        else:
            if result.allclose_pass:
                report["allclose_pass"] += 1
            if result.checker_pass:
                report["checker_pass"] += 1
            if result.is_gap:
                report["gaps"].append({
                    "problem": problem_path,
                    "candidate": candidate_path,
                    "failing_checks": [
                        {k: v for k, v in t["checks"].items()
                         if isinstance(v, tuple) and v[0] is False}
                        for t in result.trial_checks
                    ],
                })

        if result.suspected_leak:
            report["suspected_leaks"].append({
                "candidate": candidate_path, "reason": result.suspected_leak,
            })

        report["details"].append({
            "problem": problem_path,
            "candidate": candidate_path,
            "load_error": result.load_error,
            "candidate_format": result.candidate_format,
            "resolved_operator": result.resolved_operator,
            "allclose_pass": result.allclose_pass,
            "checker_pass": result.checker_pass,
            "is_gap": result.is_gap,
        })

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"  total candidates:     {report['total']}")
    print(f"  load/compile errors:  {report['load_errors']}")
    print(f"  allclose pass:        {report['allclose_pass']}")
    print(f"  checker pass:         {report['checker_pass']}")
    print(f"  GAP (allclose ok, checker fail): {len(report['gaps'])}")
    print(f"  suspected leaks:      {len(report['suspected_leaks'])}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
