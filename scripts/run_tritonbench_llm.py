"""
run_tritonbench_llm.py

Standalone CLI for checking LLM_generated/*.jsonl candidates from
TritonBench-G against their matched ground-truth references in
TritonBench_G_v1/, using verification/tritonbench_adapter.py.

Replaces run_tritonbench.py entirely (that script tested TritonBench-G's
own reference kernels against themselves via guessed calling conventions;
this script tests REAL LLM-generated candidates against matched ground
truth, which is what the project actually needs).

Usage:
    python run_tritonbench_llm.py --jsonl path/to/deepseek_tune_255.jsonl --n 30
    python run_tritonbench_llm.py --jsonl path/to/qwen_tune_rag_250.jsonl --n 30 --out results.json

Results are bucketed into PASS / FAIL_real / LOAD_ERROR_call_acc /
LOAD_ERROR_leakage / MATCH_ERROR, see verification/tritonbench_adapter.py's
docstring and this session's findings for why these are kept separate
rather than collapsed into one pass rate.
"""

import argparse
import json
import os
import sys
import time

# Locate project root the same way run_tritonbench.py did (NOT hardcoded
# to a personal path -- see the CHECKER_ROOT bug this project already hit
# once in run_checker_manual.py).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from verification import tritonbench_adapter as tba


def find_reference_file(label: str, stats_data: list) -> str:
    """Match a JSONL record's 'label' (gold code) against
    TritonBench_G_v1.json's 'output' field to recover the original
    reference filename. Mirrors EVAL/eval_G/0_call_acc.py's own matching
    logic. Raises rather than guessing if no match is found."""
    clean_label = label.replace("<|im_end|>", "").replace("<|EOT|>", "")
    for item in stats_data:
        if clean_label in item["output"]:
            return item["file"]
    raise ValueError("No match found in stats data for this record's label")


def categorize_load_error(load_error: str) -> str:
    if "redefines" in load_error and "module scope" in load_error:
        return "LOAD_ERROR_leakage"
    return "LOAD_ERROR_call_acc"


def run_batch(
    jsonl_path: str,
    repo_dir: str,
    stats_path: str,
    n: int,
    timeout_seconds: int = 20,
) -> dict:
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_data = json.loads(f.read())

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    n = min(n, len(records)) if n > 0 else len(records)

    buckets = {
        "PASS": [], "FAIL_real": [], "LOAD_ERROR_call_acc": [],
        "LOAD_ERROR_leakage": [], "MATCH_ERROR": [],
    }

    t0 = time.time()
    for i in range(n):
        rec = records[i]
        try:
            matched_file = find_reference_file(rec["label"], stats_data)
        except ValueError as e:
            buckets["MATCH_ERROR"].append({"index": i, "error": str(e)})
            print(f"[{i+1}/{n}] {time.time()-t0:6.1f}s  MATCH_ERROR", flush=True)
            continue

        reference_path = os.path.join(repo_dir, matched_file)
        result = tba.run_with_timeout(
            reference_path, candidate_src=rec["predict"], timeout_seconds=timeout_seconds
        )

        if result.load_error:
            bucket = categorize_load_error(result.load_error)
            buckets[bucket].append({
                "index": i, "file": matched_file, "error": result.load_error[:300]
            })
        else:
            all_pass = all(r["passed"] for r in result.per_call_checks)
            bucket = "PASS" if all_pass else "FAIL_real"
            buckets[bucket].append({"index": i, "file": matched_file})

        print(f"[{i+1}/{n}] {time.time()-t0:6.1f}s  {bucket:22s} {matched_file}", flush=True)

    return buckets


def main():
    parser = argparse.ArgumentParser(
        description="Check LLM_generated candidates against TritonBench_G_v1 ground truth."
    )
    parser.add_argument("--jsonl", required=True, help="Path to a LLM_generated/*.jsonl file.")
    parser.add_argument("--repo", required=True, help="Path to TritonBench_G_v1 folder.")
    parser.add_argument("--stats", required=True, help="Path to TritonBench_G_v1.json.")
    parser.add_argument("--n", type=int, default=30, help="Number of records to test (0 = all).")
    parser.add_argument("--timeout", type=int, default=20, help="Per-candidate timeout in seconds.")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write JSON results.")
    args = parser.parse_args()

    print(f"\nTesting: {args.jsonl}")
    print(f"Against: {args.repo}\n")

    buckets = run_batch(args.jsonl, args.repo, args.stats, args.n, args.timeout)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = sum(len(v) for v in buckets.values())
    for name, items in buckets.items():
        pct = (len(items) / total * 100) if total else 0
        print(f"  {name:22s} {len(items):4d} / {total}  ({pct:.1f}%)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(buckets, f, indent=2)
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()