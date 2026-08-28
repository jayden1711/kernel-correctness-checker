"""
scripts/run_adversarial_search.py

CLI entry point for the deep-agent adversarial input search.

Usage:
    python scripts/run_adversarial_search.py --operator softmax
    python scripts/run_adversarial_search.py --operator all --strategy diverse
    python scripts/run_adversarial_search.py --operator layernorm --model gpt-4o --workers 6
    python scripts/run_adversarial_search.py --operator matmul --resume <run_id>
    python scripts/run_adversarial_search.py --operator all --strategy beam --beam-width 4

Environment (.env):
    ANTHROPIC_API_KEY   claude-* models
    OPENAI_API_KEY      gpt-* models
    DEEPSEEK_API_KEY    deepseek/* models
    GEMINI_API_KEY      gemini/* models
    MISTRAL_API_KEY     mistral/* models
    CHECKER_ROOT        project root (defaults to cwd)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CHECKER_ROOT = os.environ.get("CHECKER_ROOT", str(Path(__file__).parent.parent))
if CHECKER_ROOT not in sys.path:
    sys.path.insert(0, CHECKER_ROOT)

from verification.adversarial_search.coordinator import SearchCoordinator
from verification.adversarial_search.prompts.base import validate_bug_pattern_hints
from verification.adversarial_search.strategy import STRATEGIES
from verification.adversarial_search.history.store import SearchHistoryStore


# ── Kernel path registry ──────────────────────────────────────────────────────

_REFERENCE_MAP = {
    "softmax":         "TritonBench/reference/softmax.py",
    "layernorm":       "TritonBench/reference/layernorm.py",
    "matmul":          "TritonBench/reference/mat_mult.py",
    "flash_attention": "TritonBench/reference/flash_attention.py",
    "rmsnorm":         "TritonBench/reference/rmsnorm.py",

    # Front-door wiring for the 16 operators executor.py/materializer.py
    # already support at the execution layer -- paths verified against
    # TritonBench/reference/ and TritonBench/cheating/ directly (same
    # mapping used in benchmarks/autokernel/files/tritonbench_registry.py).
    "log_softmax":                  "TritonBench/reference/log_softmax.py",
    "swish":                        "TritonBench/reference/swish.py",
    "gelu":                         "TritonBench/reference/gelu.py",
    "sum_reduction":                "TritonBench/reference/sum_reduction.py",
    "mean_reduction":               "TritonBench/reference/mean_reduction.py",
    "max_reduction":                "TritonBench/reference/max_reduction.py",
    "min_reduction":                "TritonBench/reference/min_reduction.py",
    "l1norm":                       "TritonBench/reference/l1norm.py",
    "l2norm":                       "TritonBench/reference/l2norm.py",
    "frobenius_norm":               "TritonBench/reference/frobenius_norm.py",
    "argmax":                       "TritonBench/reference/argmax.py",
    "argmin":                       "TritonBench/reference/argmin.py",
    "instancenorm":                 "TritonBench/reference/instancenorm.py",
    "batchnorm":                    "TritonBench/reference/batchnorm.py",
    "scaled_dot_product_attention": "TritonBench/reference/scaled_dot_product_attention.py",
    "causal_flash_attention":       "TritonBench/reference/causal_flash_attention.py",
}

_MUTANT_MAP = {
    "softmax": {
        "first_tile":      "TritonBench/cheating/softmax/first_tile.py",
        "wrong_reduction": "TritonBench/cheating/softmax/wrong_reduction.py",
    },
    "layernorm": {
        "ignore_gamma_beta":  "TritonBench/cheating/layer_norm/ignore_gamma_beta.py",
        "skip_mean_subtract": "TritonBench/cheating/layer_norm/skip_mean_subtract.py",
        "wrong_variance":     "TritonBench/cheating/layer_norm/wrong_variance_estimate.py",
    },
    "matmul": {
        "partial_k_reduct": "TritonBench/cheating/matmult/partial_k_reduct.py",
        "skip_boundary":    "TritonBench/cheating/matmult/skip_boundary_tiles.py",
        "swapped_strides":  "TritonBench/cheating/matmult/swapped_strides.py",
        "wrong_dtype":      "TritonBench/cheating/matmult/wrong_dtype.py",
    },
    "flash_attention": {
        "approx_denom":   "TritonBench/cheating/flash_attention/approx_denom.py",
        "drop_last_tile": "TritonBench/cheating/flash_attention/drop_last_tile.py",
        "skip_rescaling": "TritonBench/cheating/flash_attention/skip_rescaling.py",
        "wrong_mask":     "TritonBench/cheating/flash_attention/wrong_mask.py",
    },
    "rmsnorm": {
        "ignore_gamma":      "TritonBench/cheating/rmsnorm/ignore_gamma.py",
        "wrong_norm":        "TritonBench/cheating/rmsnorm/wrong_norm.py",
        "partial_reduction": "TritonBench/cheating/rmsnorm/partial_reduction.py",
    },

    # Each of these 16 has exactly one mutant on disk (see
    # TritonBench/cheating/<dir>/) -- unlike the original 5, which were
    # hand-built with several named bugs apiece.
    "log_softmax": {
        "skip_max_subtraction": "TritonBench/cheating/log_softmax/skip_max_subtraction.py",
    },
    "swish": {
        "linear_sigmoid_approx": "TritonBench/cheating/swish/linear_sigmoid_approx.py",
    },
    "gelu": {
        "sigmoid_approx": "TritonBench/cheating/gelu/sigmoid_approx.py",
    },
    "sum_reduction": {
        "partial_reduction": "TritonBench/cheating/sum_reduction/partial_reduction.py",
    },
    "mean_reduction": {
        "partial_reduction": "TritonBench/cheating/mean_reduction/partial_reduction.py",
    },
    "max_reduction": {
        "wrong_padding": "TritonBench/cheating/max_reduction/wrong_padding.py",
    },
    "min_reduction": {
        "wrong_padding": "TritonBench/cheating/min_reduction/wrong_padding.py",
    },
    "l1norm": {
        "partial_reduction": "TritonBench/cheating/l1norm/partial_reduction.py",
    },
    "l2norm": {
        "wrong_norm": "TritonBench/cheating/l2norm/wrong_norm.py",
    },
    "frobenius_norm": {
        "wrong_norm": "TritonBench/cheating/frobenius_norm/wrong_norm.py",
    },
    "argmax": {
        "tiebreak": "TritonBench/cheating/argmax/tiebreak.py",
    },
    "argmin": {
        "tiebreak": "TritonBench/cheating/argmin/tiebreak.py",
    },
    "instancenorm": {
        "skip_eps": "TritonBench/cheating/instancenorm/skip_eps.py",
    },
    "batchnorm": {
        "wrong_running_stats_broadcast":
            "TritonBench/cheating/batchnorm/wrong_running_stats_broadcast.py",
    },
    "scaled_dot_product_attention": {
        "wrong_mask": "TritonBench/cheating/scaled_dot_product_attention/wrong_mask.py",
    },
    "causal_flash_attention": {
        "wrong_causal_mask": "TritonBench/cheating/causal_flash_attention/wrong_causal_mask.py",
    },
}


def _resolve_paths(operator: str, filter_mutants=None):
    ref_rel = _REFERENCE_MAP.get(operator)
    if ref_rel is None:
        raise ValueError(f"Unknown operator: {operator!r}")
    ref_path = os.path.join(CHECKER_ROOT, ref_rel)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference kernel not found: {ref_path}")

    mutants_raw = _MUTANT_MAP.get(operator, {})
    if filter_mutants:
        mutants_raw = {k: v for k, v in mutants_raw.items() if k in filter_mutants}

    mutant_paths = {}
    for name, rel in mutants_raw.items():
        abs_path = os.path.join(CHECKER_ROOT, rel)
        if not os.path.exists(abs_path):
            print(f"[warn] Mutant not found, skipping: {abs_path}")
            continue
        mutant_paths[name] = abs_path

    if not mutant_paths:
        raise FileNotFoundError(
            f"No mutant kernels found for '{operator}'. "
            f"Check TritonBench/cheating/{operator}/"
        )

    # Fail fast if any mutant has no bug-pattern hint. The hint lookup in
    # prompts/base.py is a silent `.get(key, "")`, so a missing entry costs a
    # whole search run's worth of unguided proposals and reports nothing --
    # exactly how "wrong_causal_mask" went unnoticed across 120 proposals
    # (adversarial_results/CFA_NONHIT_ROOTCAUSE.md). Startup is the only place
    # this is cheap to catch.
    missing_hints = validate_bug_pattern_hints(mutant_paths.keys())
    if missing_hints:
        raise ValueError(
            f"No BUG_PATTERN_HINTS entry for mutant(s) {missing_hints} of "
            f"operator '{operator}'. The prompt's hint lookup fails silently to "
            f"an empty string, so the worker would get no bug-specific guidance "
            f"for the entire run. Add an entry to BUG_PATTERN_HINTS in "
            f"verification/adversarial_search/prompts/base.py."
        )

    return ref_path, mutant_paths


def _set_api_key(model: str, api_key: Optional[str] = None):
    key_map = {
        "claude":   "ANTHROPIC_API_KEY",
        "gpt":      "OPENAI_API_KEY",
        "openai":   "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini":   "GEMINI_API_KEY",
        "mistral":  "MISTRAL_API_KEY",
    }
    for prefix, env_var in key_map.items():
        if prefix in model.lower():
            if api_key:
                os.environ[env_var] = api_key
            elif not os.environ.get(env_var):
                print(f"[warn] {env_var} not set — {model} calls will fail.")
            return


# ── CLI ───────────────────────────────────────────────────────────────────────

ALL_OPERATORS = [
    "softmax", "layernorm", "matmul", "flash_attention", "rmsnorm",
    "log_softmax", "swish", "gelu", "sum_reduction", "mean_reduction",
    "max_reduction", "min_reduction", "l1norm", "l2norm", "frobenius_norm",
    "argmax", "argmin", "instancenorm", "batchnorm",
    "scaled_dot_product_attention", "causal_flash_attention",
]
DEFAULT_MODEL  = "claude-sonnet-4-6"

from typing import Optional


def main():
    parser = argparse.ArgumentParser(
        description="Deep-agent adversarial input search for Triton kernel verification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies:
  greedy   Single best proposal per iteration. Lowest cost, fastest.
  beam     Top-B proposals in parallel per iteration. Better coverage.
  diverse  Beam search with diversity penalty across bug patterns.
           Recommended for full operator sweeps.

Examples:
  python scripts/run_adversarial_search.py --operator softmax
  python scripts/run_adversarial_search.py --operator all --strategy diverse --workers 6
  python scripts/run_adversarial_search.py --operator layernorm --model gpt-4o
  python scripts/run_adversarial_search.py --operator matmul --resume abc123
  python scripts/run_adversarial_search.py --history          # print history summary
        """,
    )

    parser.add_argument("--operator",  choices=ALL_OPERATORS + ["all"], default=None)
    parser.add_argument("--model",     default=DEFAULT_MODEL,
                        help=f"LiteLLM model string (default: {DEFAULT_MODEL})")
    parser.add_argument("--strategy",  choices=list(STRATEGIES.keys()), default="beam")
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--max-iter",  type=int, default=20, dest="max_iter")
    parser.add_argument("--beam-width",type=int, default=None, dest="beam_width",
                        help="Beam width (defaults to --workers)")
    parser.add_argument("--timeout",   type=int, default=30)
    parser.add_argument("--mutants",   nargs="+", default=None, metavar="NAME")
    parser.add_argument("--output-dir",default="adversarial_results", dest="output_dir")
    parser.add_argument("--api-key",   default=None, dest="api_key")
    parser.add_argument("--resume",    default=None, metavar="RUN_ID",
                        help="Resume an interrupted run by its run_id.")
    parser.add_argument("--diversity-weight", type=float, default=3.0,
                        dest="diversity_weight",
                        help="λ for diverse beam strategy (default 3.0).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-batch", action="store_true", dest="no_batch",
                        help="Run each kernel in its own subprocess (the "
                             "pre-batching path). Default is one subprocess "
                             "per proposal. Use this for the unbatched arm of "
                             "a latency A/B, or as an escape hatch: it also "
                             "leaves the RNG unseeded, so reference and "
                             "mutants draw different tensors, which is the "
                             "old semantics and NOT verdict-equivalent to the "
                             "batched path.")
    parser.add_argument("--forkserver",
                        action=argparse.BooleanOptionalAction, dest="forkserver",
                        default=True,
                        help="Create batched subprocesses by forking a "
                             "torch-preloaded server instead of booting a fresh "
                             "interpreter, removing the 85%% of startup that is "
                             "`import torch`. Independent of --no-batch and "
                             "ignored when it is set: the single-kernel path is "
                             "spawn-only by design, because it does not seed and "
                             "a fork would hand every child the SAME generator "
                             "state. DEFAULT ON since 2026-08-28 (GPU A/B in "
                             "verification_runs/forkserver_ab/: 36-41%% "
                             "end-to-end, gates green; re-verified at the new "
                             "default). --no-forkserver restores spawn.")
    parser.add_argument("--history",   action="store_true",
                        help="Print search history summary and exit.")
    parser.add_argument("--coverage",  action="store_true",
                        help="Print coverage report (which bugs confirmed) and exit.")

    args = parser.parse_args()

    # History/coverage subcommands
    if args.history or args.coverage:
        db_path = Path(args.output_dir) / "search_history.db"
        if not db_path.exists():
            print(f"No history DB at {db_path}")
            return
        with SearchHistoryStore(str(db_path)) as store:
            if args.history:
                runs = store.list_runs()
                print(f"\n{'='*70}")
                print(f"  {'run_id':<10} {'operator':<16} {'strategy':<10} "
                      f"{'status':<10} {'model':<25}")
                print(f"  {'-'*70}")
                for r in runs:
                    print(f"  {r['run_id']:<10} {r['operator']:<16} "
                          f"{r['strategy']:<10} {r['status']:<10} {r['model']:<25}")
            if args.coverage:
                report = store.coverage_report()
                print(f"\nCoverage report (confirmed hits with gap):")
                for op, patterns in report.items():
                    print(f"  {op}:")
                    for pattern, mutants in patterns.items():
                        print(f"    [{pattern}] → caught {set(mutants)}")
        return

    if args.operator is None:
        parser.error("--operator is required unless using --history or --coverage")

    _set_api_key(args.model, args.api_key)

    operators = ALL_OPERATORS if args.operator == "all" else [args.operator]
    n_workers = args.workers

    overall_t0 = time.perf_counter()
    summary = []

    for operator in operators:
        print(f"\n{'='*60}\n  {operator}\n{'='*60}")
        try:
            ref_path, mutant_paths = _resolve_paths(operator, args.mutants)
        except (FileNotFoundError, ValueError) as e:
            print(f"[error] {e}")
            summary.append((operator, "SKIPPED", 0, 0.0))
            continue

        coord = SearchCoordinator(
            operator=operator,
            reference_src_path=ref_path,
            mutant_src_paths=mutant_paths,
            model=args.model,
            strategy=args.strategy,
            n_workers=n_workers,
            batch_executions=not args.no_batch,
            use_forkserver=args.forkserver,
            max_iterations=args.max_iter,
            timeout_per_exec=args.timeout,
            output_dir=args.output_dir,
            resume_run_id=args.resume if len(operators) == 1 else None,
            diversity_weight=args.diversity_weight,
            temperature=args.temperature,
        )

        result = coord.run()
        status = "HIT" if result.winning_proposal else "NO_HIT"
        summary.append((operator, status, result.total_proposals, result.wall_time_s))

    overall_s = time.perf_counter() - overall_t0

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"  {'Operator':<18} {'Status':<10} {'Proposals':>10} {'Time':>8}")
    print(f"  {'-'*50}")
    for op, status, props, t in summary:
        print(f"  {op:<18} {status:<10} {props:>10} {t:>7.1f}s")
    print(f"\n  Total: {overall_s:.1f}s   Output: {args.output_dir}/")


if __name__ == "__main__":
    main()