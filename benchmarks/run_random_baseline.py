"""
scripts/run_random_baseline.py

Random-search ablation: same operators, same mutant corpus, same hit
invariant, same proposal budget as the LLM-guided adversarial search  -
the only thing that changes is how InputProposals are generated.

DESIGN NOTES (fixes vs. the first draft):

  - Shape sampling is not drawn from a curated list containing known
    adversarial shapes (e.g. a hand-picked non-power-of-two or odd-prime
    entry). That would leak the exact structural insight the LLM search
    is supposed to contribute, biasing the comparison in either
    direction depending on how often the curated entry gets picked.
    Instead, dimensions are drawn uniformly from a broad range -
    non-power-of-two shapes show up at the same rate any other value
    does, not because they were shortlisted.

  - Runs N_SEEDS independent seeded trials per operator, not one. A
    single random run's proposals-to-hit is one noisy sample; report
    mean/median hit-proposal-count and hit-rate-within-budget across
    seeds, not a single number.

  - Budget matches the LLM system's budget by default (44) for the
    headline comparison. Pass --budget to run a second pass (e.g. 440)
    if you also want to report whether random catches up given far more
    tries - keep that as an explicitly separate reported number, not
    blended into the same-budget comparison.
"""

import argparse
import json
import os
import random
import statistics
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from verification.adversarial_search.schemas import InputProposal, TensorDescriptor
from verification.adversarial_search.executor import execute_proposal


# INFERRED from run_checker.py's import statements -- confirm against
# your actual TritonBench/ directory layout before trusting results.
REFERENCE_PATHS = {
    "softmax":         "TritonBench/reference/softmax.py",
    "layernorm":       "TritonBench/reference/layernorm.py",
    "matmul":          "TritonBench/reference/mat_mult.py",
    "flash_attention": "TritonBench/reference/flash_attention.py",
    "rmsnorm":         "TritonBench/reference/rmsnorm.py",
}

MUTANT_PATHS: Dict[str, List[Tuple[str, str]]] = {
    "softmax": [
        ("softmax/first_tile", "TritonBench/cheating/softmax/first_tile.py"),
        ("softmax/wrong_reduction", "TritonBench/cheating/softmax/wrong_reduction.py"),
    ],
    "layernorm": [
        ("layernorm/ignore_gamma_beta", "TritonBench/cheating/layer_norm/ignore_gamma_beta.py"),
        ("layernorm/skip_mean_subtract", "TritonBench/cheating/layer_norm/skip_mean_subtract.py"),
        ("layernorm/wrong_variance", "TritonBench/cheating/layer_norm/wrong_variance_estimate.py"),
    ],
    "matmul": [
        ("matmul/partial_k_reduct", "TritonBench/cheating/matmult/partial_k_reduct.py"),
        ("matmul/skip_boundary", "TritonBench/cheating/matmult/skip_boundary_tiles.py"),
        ("matmul/swapped_strides", "TritonBench/cheating/matmult/swapped_strides.py"),
        ("matmul/wrong_dtype", "TritonBench/cheating/matmult/wrong_dtype.py"),
    ],
    "flash_attention": [
        ("flash_attn/approx_denom", "TritonBench/cheating/flash_attention/approx_denom.py"),
        ("flash_attn/drop_last_tile", "TritonBench/cheating/flash_attention/drop_last_tile.py"),
        ("flash_attn/skip_rescaling", "TritonBench/cheating/flash_attention/skip_rescaling.py"),
        ("flash_attn/wrong_mask", "TritonBench/cheating/flash_attention/wrong_mask.py"),
    ],
    "rmsnorm": [
        ("rmsnorm/ignore_gamma", "TritonBench/cheating/rmsnorm/ignore_gamma.py"),
        ("rmsnorm/wrong_norm", "TritonBench/cheating/rmsnorm/wrong_norm.py"),
        ("rmsnorm/partial_reduction", "TritonBench/cheating/rmsnorm/partial_reduction.py"),
    ],
}


OPERATORS = ["softmax", "layernorm", "matmul", "flash_attention", "rmsnorm"]

TENSOR_KEYS = {
    "softmax":         ["x"],
    "layernorm":       ["x", "gamma", "beta"],
    "matmul":          ["A", "B"],
    "flash_attention": ["Q", "K", "V"],
    "rmsnorm":         ["x", "gamma"],
}

FILLS = ["randn", "ones", "zeros", "arange"]

# Broad, unbiased ranges -- NOT a curated shortlist of known-adversarial
# values. Odd/prime/non-power-of-two dimensions occur here at whatever
# rate uniform sampling produces them, same as any other value.
DIM_RANGE = (32, 1024)
SCALE_RANGE_LOG10 = (-2, 4)  # sampled in log space: 1e-2 .. 1e4


def _random_dim(rng: random.Random) -> int:
    return rng.randint(*DIM_RANGE)


def _random_scale(rng: random.Random) -> float:
    exponent = rng.uniform(*SCALE_RANGE_LOG10)
    return 10 ** exponent


def random_proposal(operator: str, worker_id: str, iteration: int, rng: random.Random) -> InputProposal:
    """Generate a random InputProposal with no LLM and no curated
    edge-case shortlist -- shape dims and scale are drawn from broad
    uniform/log-uniform ranges, not hand-picked gotcha values."""
    primary_shape = [_random_dim(rng), _random_dim(rng)] if operator != "flash_attention" else [_random_dim(rng), _random_dim(rng)]
    fill = rng.choice(FILLS)
    scale = _random_scale(rng)
    shift = rng.choice([0.0, 0.0, 0.0, rng.uniform(-10, 10)])  # mostly centered, occasionally shifted

    tensors = {}
    for key in TENSOR_KEYS[operator]:
        if key == "gamma":
            tensors[key] = TensorDescriptor(
                shape=[primary_shape[-1]], dtype="float32",
                fill=rng.choice(["ones", "randn"]),
                scale=_random_scale(rng),
                shift=0.0,
            )
        elif key == "beta":
            tensors[key] = TensorDescriptor(
                shape=[primary_shape[-1]], dtype="float32",
                fill="zeros", scale=1.0,
                shift=rng.uniform(-5, 5),
            )
        else:
            tensors[key] = TensorDescriptor(
                shape=list(primary_shape), dtype="float32",
                fill=fill, scale=scale, shift=shift,
            )

    return InputProposal(
        proposal_id=str(uuid.uuid4()),
        worker_id=worker_id,
        iteration=iteration,
        operator=operator,
        tensors=tensors,
        rationale="random baseline",
        predicted_failure_mode="random",
    )


@dataclass
class SeedResult:
    seed: int
    hit: bool
    proposals_to_hit: Optional[int]  # None if budget exhausted without a hit


def _evaluate_proposal(proposal: InputProposal, operator: str, timeout_seconds: int) -> bool:
    """
    Reproduces ProposalVerdict's hit invariant exactly:
      1. reference_passed: candidate==reference self-check passes the
         full three-layer checker (input is semantically valid, not
         just crash-free)
      2. at least one mutant has passed_checker=False AND passed_naive=True
         (checker caught a bug that naive allclose missed -- the gap)
    Does NOT reimplement a looser or stricter version of this rule --
    any difference here would break the apples-to-apples comparison
    with the LLM-guided coordinator's own verdicts.
    """
    reference_path = REFERENCE_PATHS[operator]

    ref_result = execute_proposal(
        proposal, kernel_id="reference",
        candidate_src_path=reference_path, reference_src_path=reference_path,
        operator=operator, timeout_seconds=timeout_seconds,
    )
    if not ref_result.passed_checker:
        return False  # invalid input -- can't be a hit regardless of mutants

    for mutant_id, mutant_path in MUTANT_PATHS[operator]:
        mr = execute_proposal(
            proposal, kernel_id=mutant_id,
            candidate_src_path=mutant_path, reference_src_path=reference_path,
            operator=operator, timeout_seconds=timeout_seconds,
        )
        if (not mr.passed_checker) and mr.passed_naive:
            return True  # gap confirmed -- checker caught what naive testing missed

    return False


def run_operator(operator: str, budget: int, seed: int, timeout_seconds: int) -> SeedResult:
    rng = random.Random(seed)
    worker_id = f"random-baseline-{operator}-{seed}"

    for i in range(1, budget + 1):
        proposal = random_proposal(operator, worker_id, i, rng)
        if _evaluate_proposal(proposal, operator, timeout_seconds):
            return SeedResult(seed=seed, hit=True, proposals_to_hit=i)

    return SeedResult(seed=seed, hit=False, proposals_to_hit=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=44,
                         help="proposals per seed -- match your LLM system's budget for the headline comparison")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=30,
                         help="per-execute_proposal subprocess timeout, seconds")
    parser.add_argument("--checker-root", default=".",
                         help="project root added to sys.path inside each executor subprocess "
                              "(executor.py reads this from the CHECKER_ROOT env var)")
    parser.add_argument("--out", default="random_baseline_report.json")
    args = parser.parse_args()

    os.environ["CHECKER_ROOT"] = args.checker_root

    report = {"budget": args.budget, "n_seeds": args.n_seeds, "operators": {}}

    for operator in OPERATORS:
        results = [
            run_operator(operator, args.budget, seed, args.timeout)
            for seed in range(args.n_seeds)
        ]
        hits = [r for r in results if r.hit]
        proposals_to_hit = [r.proposals_to_hit for r in hits]

        summary = {
            "hit_rate": len(hits) / len(results),
            "mean_proposals_to_hit": statistics.mean(proposals_to_hit) if proposals_to_hit else None,
            "median_proposals_to_hit": statistics.median(proposals_to_hit) if proposals_to_hit else None,
            "stdev_proposals_to_hit": statistics.stdev(proposals_to_hit) if len(proposals_to_hit) > 1 else None,
            "per_seed": [{"seed": r.seed, "hit": r.hit, "proposals_to_hit": r.proposals_to_hit} for r in results],
        }
        report["operators"][operator] = summary

        print(f"{operator:16s} hit_rate={summary['hit_rate']:.2f} "
              f"mean_proposals={summary['mean_proposals_to_hit']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()