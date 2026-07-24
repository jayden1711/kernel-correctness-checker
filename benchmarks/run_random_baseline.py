"""
scripts/run_random_baseline.py

Random-search ablation: same operators, same mutant corpus, same hit
invariant, same proposal budget as the LLM-guided adversarial search --
the only thing that changes is how InputProposals are generated.

DESIGN NOTES (fixes vs. the first draft):

  - Shape sampling is NOT drawn from a curated list containing known
    adversarial shapes (e.g. a hand-picked non-power-of-two or odd-prime
    entry). That would leak the exact structural insight the LLM search
    is supposed to contribute, biasing the comparison in either
    direction depending on how often the curated entry gets picked.
    Instead, dimensions are drawn uniformly from a broad range --
    non-power-of-two shapes show up at the same rate any other value
    does, not because they were shortlisted.

  - Runs N_SEEDS independent seeded trials per operator, not one. A
    single random run's proposals-to-hit is one noisy sample; report
    mean/median hit-proposal-count and hit-rate-within-budget across
    seeds, not a single number.

  - Attention operators (flash_attention, scaled_dot_product_attention,
    causal_flash_attention) draw head dim D from a fixed power-of-2 set,
    not the broad DIM_RANGE -- the underlying Triton kernels use
    tl.arange(0, D), which requires D to be a power of 2 at compile
    time. Confirmed this constraint is inherited from flash_attention's
    original kernel, not new to the 2 new attention operators.

  - Norm-family operators (layernorm, rmsnorm, instancenorm, batchnorm)
    draw their primary tensor's scale from NORM_SCALE_RANGE_LOG10
    (1e0..1e4), not the broad SCALE_RANGE_LOG10 (1e-2..1e4) used
    everywhere else. Reason: positive_scale_invariance checks compare
    normalize(c*x) to normalize(x); that equality only holds when
    variance(x) dominates eps in the denominator. Confirmed via
    instancenorm: a proposal with scale=0.0175 produced max_diff=1.86
    under rescale, a false-looking failure driven entirely by eps
    dominating a near-zero variance, not a kernel bug. Every other
    generator (softmax-like, elementwise, matmul, attention) is
    UNCHANGED and still uses the original broad _random_scale --
    narrowing those would bias the random baseline for operators that
    were never implicated in this finding.

  - OPEN QUESTION, not yet resolved: l1norm and l2norm (routed through
    _gen_softmax_like, still on the broad scale range) may have the
    same eps-vs-tiny-variance structure in their own
    positive_scale_invariance checks as instancenorm did. Check
    verification/layer3_properties/norm_properties.py's
    check_positive_scale_invariance and its eps handling before
    trusting l1norm/l2norm's random-baseline hit rate at small scales --
    if the same failure mode applies, _gen_softmax_like needs to be
    split (or l1norm/l2norm given their own generator) rather than
    patching _make_tensor globally again.

WIRED TO THE REAL PIPELINE:
  _evaluate_proposal now calls executor.execute_proposal against the
  reference kernel and every mutant for the operator, and reproduces
  ProposalVerdict's exact hit invariant (reference_passed AND at least
  one mutant with passed_checker=False AND passed_naive=True) rather
  than a re-derived approximation of it.

RESOLVED (previously blocking, now fixed):
  executor.py's FUNC_NAMES and SPEC_MAP dicts only covered the
  original 5 operators -- every one of the 16 new pure-tensor-signature
  operators hit a KeyError on the very first call, before any Triton
  compilation. Fixed in executor.py: both dicts extended, imports added
  for all 16 new specs. Confirmed via per-operator debug harness that
  all 16 now produce real check_results with no exception, EXCEPT the
  two attention operators (see D-must-be-power-of-2 note above -- fixed
  via the generator change, not an executor change) and instancenorm
  (see NORM_SCALE_RANGE_LOG10 note above).

  Separately, GroupNorm (num_groups: int) and the six pooling operators
  (kernel_size/stride/padding: int) still cannot be represented as a
  valid InputProposal AT ALL -- TensorDescriptor/InputProposal
  (schemas.py) only has a slot for named tensors, no field for a plain
  scalar hyperparameter. This needs a schema change, not a generator
  fix, and isn't attempted here. cross_entropy (targets: int64 tensor)
  is deferred separately -- dtype="int64" support in the materializer
  is unconfirmed.

ONE REMAINING UNCONFIRMED PIECE -- file paths (original 5 operators only).
  executor.execute_proposal takes filesystem paths (candidate_src_path,
  reference_src_path), loaded via importlib.spec_from_file_location, not
  Python dotted import paths. REFERENCE_PATHS / MUTANT_PATHS for the
  original 5 (softmax/layernorm/matmul/flash_attention/rmsnorm) are
  INFERRED from the dotted imports seen in run_checker.py, not confirmed
  against your actual directory layout -- the 16 new operators' paths
  are HIGH confidence and confirmed against the real directory listing.
"""

import argparse
import json
import os
import random
import statistics
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from verification.adversarial_search.schemas import InputProposal, TensorDescriptor
from verification.adversarial_search.executor import execute_proposal


# ---------------------------------------------------------------------------
# Reference / mutant file paths
# ---------------------------------------------------------------------------

REFERENCE_PATHS = {
    # Original 5 -- INFERRED, unconfirmed against your actual layout.
    "softmax":         "TritonBench/reference/softmax.py",
    "layernorm":       "TritonBench/reference/layernorm.py",
    "matmul":          "TritonBench/reference/mat_mult.py",
    "flash_attention": "TritonBench/reference/flash_attention.py",
    "rmsnorm":         "TritonBench/reference/rmsnorm.py",

    # 16 new, pure-tensor-signature operators -- confirmed against real
    # directory listing.
    "log_softmax":                    "TritonBench/reference/log_softmax.py",
    "swish":                          "TritonBench/reference/swish.py",
    "gelu":                           "TritonBench/reference/gelu.py",
    "sum_reduction":                  "TritonBench/reference/sum_reduction.py",
    "mean_reduction":                 "TritonBench/reference/mean_reduction.py",
    "max_reduction":                  "TritonBench/reference/max_reduction.py",
    "min_reduction":                  "TritonBench/reference/min_reduction.py",
    "l1norm":                         "TritonBench/reference/l1norm.py",
    "l2norm":                         "TritonBench/reference/l2norm.py",
    "frobenius_norm":                 "TritonBench/reference/frobenius_norm.py",
    "argmax":                         "TritonBench/reference/argmax.py",
    "argmin":                         "TritonBench/reference/argmin.py",
    "instancenorm":                   "TritonBench/reference/instancenorm.py",
    "batchnorm":                      "TritonBench/reference/batchnorm.py",
    "scaled_dot_product_attention":   "TritonBench/reference/scaled_dot_product_attention.py",
    "causal_flash_attention":         "TritonBench/reference/causal_flash_attention.py",

    # Deferred (schema/materializer blockers) -- paths listed for when
    # they're unblocked, NOT added to OPERATORS below.
    "cross_entropy":  "TritonBench/reference/cross_entropy.py",
    "groupnorm":       "TritonBench/reference/groupnorm.py",
    "max_pool1d":      "TritonBench/reference/max_pool1d.py",
    "max_pool2d":      "TritonBench/reference/max_pool2d.py",
    "max_pool3d":      "TritonBench/reference/max_pool3d.py",
    "avg_pool1d":      "TritonBench/reference/avg_pool1d.py",
    "avg_pool2d":      "TritonBench/reference/avg_pool2d.py",
    "avg_pool3d":      "TritonBench/reference/avg_pool3d.py",
}

MUTANT_PATHS: Dict[str, List[Tuple[str, str]]] = {
    # Original 5
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

    # 16 new, pure-tensor-signature operators
    "log_softmax": [
        ("log_softmax/skip_max_subtraction", "TritonBench/cheating/log_softmax/skip_max_subtraction.py"),
    ],
    "swish": [
        ("swish/linear_sigmoid_approx", "TritonBench/cheating/swish/linear_sigmoid_approx.py"),
    ],
    "gelu": [
        ("gelu/sigmoid_approx", "TritonBench/cheating/gelu/sigmoid_approx.py"),
    ],
    "sum_reduction": [
        ("sum_reduction/partial_reduction", "TritonBench/cheating/sum_reduction/partial_reduction.py"),
    ],
    "mean_reduction": [
        ("mean_reduction/partial_reduction", "TritonBench/cheating/mean_reduction/partial_reduction.py"),
    ],
    "max_reduction": [
        ("max_reduction/wrong_padding", "TritonBench/cheating/max_reduction/wrong_padding.py"),
    ],
    "min_reduction": [
        ("min_reduction/wrong_padding", "TritonBench/cheating/min_reduction/wrong_padding.py"),
    ],
    "l1norm": [
        ("l1norm/partial_reduction", "TritonBench/cheating/l1norm/partial_reduction.py"),
    ],
    "l2norm": [
        ("l2norm/wrong_norm", "TritonBench/cheating/l2norm/wrong_norm.py"),
    ],
    "frobenius_norm": [
        ("frobenius_norm/wrong_norm", "TritonBench/cheating/frobenius_norm/wrong_norm.py"),
    ],
    "argmax": [
        ("argmax/tiebreak", "TritonBench/cheating/argmax/tiebreak.py"),
    ],
    "argmin": [
        ("argmin/tiebreak", "TritonBench/cheating/argmin/tiebreak.py"),
    ],
    "instancenorm": [
        ("instancenorm/skip_eps", "TritonBench/cheating/instancenorm/skip_eps.py"),
    ],
    "batchnorm": [
        ("batchnorm/wrong_running_stats_broadcast", "TritonBench/cheating/batchnorm/wrong_running_stats_broadcast.py"),
    ],
    "scaled_dot_product_attention": [
        ("scaled_dot_product_attention/wrong_mask", "TritonBench/cheating/scaled_dot_product_attention/wrong_mask.py"),
    ],
    "causal_flash_attention": [
        ("causal_flash_attention/wrong_causal_mask", "TritonBench/cheating/causal_flash_attention/wrong_causal_mask.py"),
    ],

    # Deferred -- listed for completeness, not reachable via OPERATORS
    "cross_entropy": [
        ("cross_entropy/missing_max_subtraction", "TritonBench/cheating/cross_entropy/missing_max_subtraction.py"),
    ],
    "groupnorm": [
        ("groupnorm/ignore_affine", "TritonBench/cheating/groupnorm/ignore_affine.py"),
    ],
    "max_pool1d": [("max_pool1d/wrong_padding", "TritonBench/cheating/max_pool1d/wrong_padding.py")],
    "max_pool2d": [("max_pool2d/wrong_padding", "TritonBench/cheating/max_pool2d/wrong_padding.py")],
    "max_pool3d": [("max_pool3d/wrong_padding", "TritonBench/cheating/max_pool3d/wrong_padding.py")],
    "avg_pool1d": [("avg_pool1d/wrong_divisor", "TritonBench/cheating/avg_pool1d/wrong_divisor.py")],
    "avg_pool2d": [("avg_pool2d/wrong_divisor", "TritonBench/cheating/avg_pool2d/wrong_divisor.py")],
    "avg_pool3d": [("avg_pool3d/wrong_divisor", "TritonBench/cheating/avg_pool3d/wrong_divisor.py")],
}

# Only operators with a valid InputProposal representation AND confirmed
# executor/materializer support. groupnorm/pooling/cross_entropy are
# deliberately excluded -- see module docstring.
OPERATORS = [
    "softmax", "layernorm", "matmul", "flash_attention", "rmsnorm",
    "log_softmax", "swish", "gelu",
    "sum_reduction", "mean_reduction", "max_reduction", "min_reduction",
    "l1norm", "l2norm", "frobenius_norm",
    "argmax", "argmin",
    "instancenorm", "batchnorm",
    "scaled_dot_product_attention", "causal_flash_attention",
]

BLOCKED_OPERATORS = {
    "groupnorm":       "num_groups is a scalar hyperparameter -- InputProposal/TensorDescriptor has no slot for it",
    "max_pool1d":       "kernel_size/stride/padding are scalar hyperparameters -- same schema blocker",
    "max_pool2d":       "same",
    "max_pool3d":       "same",
    "avg_pool1d":       "same",
    "avg_pool2d":       "same",
    "avg_pool3d":       "same",
    "cross_entropy":   "targets needs dtype='int64' -- materializer support unconfirmed",
}


# ---------------------------------------------------------------------------
# Shape/scale sampling primitives
# ---------------------------------------------------------------------------

FILLS = ["randn", "ones", "zeros", "arange"]

# Broad, unbiased ranges -- NOT a curated shortlist of known-adversarial
# values. Odd/prime/non-power-of-two dimensions occur here at whatever
# rate uniform sampling produces them, same as any other value.
DIM_RANGE = (32, 1024)
SMALL_DIM_RANGE = (4, 32)     # for 4D tensors (instancenorm/batchnorm) -- keeps N*C*H*W reasonable
SCALE_RANGE_LOG10 = (-2, 4)   # sampled in log space: 1e-2 .. 1e4 -- default, used everywhere
                              # EXCEPT norm-family primary tensors (see below)
NORM_SCALE_RANGE_LOG10 = (0, 4)   # 1e0 .. 1e4 -- avoids near-zero-variance inputs where
                                   # positive_scale_invariance breaks down against a fixed eps.
                                   # ONLY applied at explicit call sites in norm-family
                                   # generators (layernorm/rmsnorm/instancenorm/batchnorm),
                                   # NOT inside the shared _make_tensor helper.

# Attention head dims must be a compile-time power of 2 -- the Triton
# kernels use tl.arange(0, D). Confirmed inherited from flash_attention's
# original kernel (not new to the 2 new attention operators), so this is
# a real deployment constraint, not an artifact of these two kernels.
ATTENTION_HEAD_DIMS = [32, 64, 128, 256, 512]


def _random_dim(rng: random.Random, dim_range=DIM_RANGE) -> int:
    return rng.randint(*dim_range)


def _random_scale(rng: random.Random) -> float:
    exponent = rng.uniform(*SCALE_RANGE_LOG10)
    return 10 ** exponent


def _random_scale_norm(rng: random.Random) -> float:
    exponent = rng.uniform(*NORM_SCALE_RANGE_LOG10)
    return 10 ** exponent


def _random_shift(rng: random.Random) -> float:
    return rng.choice([0.0, 0.0, 0.0, rng.uniform(-10, 10)])  # mostly centered, occasionally shifted


def _make_tensor(rng: random.Random, shape: List[int]) -> TensorDescriptor:
    """Default tensor generator -- broad scale range. Norm-family
    generators below do NOT use this for their primary tensor; they
    build a TensorDescriptor directly with _random_scale_norm instead."""
    return TensorDescriptor(
        shape=shape, dtype="float32",
        fill=rng.choice(FILLS), scale=_random_scale(rng), shift=_random_shift(rng),
    )


# ---------------------------------------------------------------------------
# Per-operator proposal generators
# ---------------------------------------------------------------------------

def _gen_softmax_like(rng, keys: List[str]) -> Dict[str, TensorDescriptor]:
    """Single 2D tensor: softmax, log_softmax, sum/mean/max/min_reduction,
    l1norm, l2norm, argmax, argmin -- all just ["x"] at (n_rows, n_cols).
    NOTE: l1norm/l2norm's positive_scale_invariance check may have the
    same eps-vs-tiny-variance issue found in instancenorm -- unconfirmed.
    If so, this generator will need to be split for those two operators
    rather than patched globally again."""
    shape = [_random_dim(rng), _random_dim(rng)]
    return {keys[0]: _make_tensor(rng, shape)}


def _gen_elementwise_1d(rng, keys: List[str]) -> Dict[str, TensorDescriptor]:
    """Single 1D tensor: swish, gelu. No normalization -- broad scale
    range is fine, unaffected by the eps/scale-invariance finding."""
    n = _random_dim(rng)
    return {keys[0]: _make_tensor(rng, [n])}


def _gen_frobenius_norm(rng, keys: List[str]) -> Dict[str, TensorDescriptor]:
    """Single 2D tensor, kept small -- reference uses O(n) atomic_add
    over the whole tensor, unlike every other operator here."""
    shape = [rng.randint(4, 128), rng.randint(4, 128)]
    return {keys[0]: _make_tensor(rng, shape)}


def _gen_layernorm(rng, keys: List[str]) -> Dict[str, TensorDescriptor]:
    n_rows, n_cols = _random_dim(rng), _random_dim(rng)
    return {
        "x": TensorDescriptor(shape=[n_rows, n_cols], dtype="float32",
                               fill=rng.choice(FILLS), scale=_random_scale_norm(rng), shift=_random_shift(rng)),
        "gamma": TensorDescriptor(shape=[n_cols], dtype="float32",
                                   fill=rng.choice(["ones", "randn"]), scale=_random_scale_norm(rng), shift=0.0),
        "beta": TensorDescriptor(shape=[n_cols], dtype="float32",
                                  fill="zeros", scale=1.0, shift=rng.uniform(-5, 5)),
    }


def _gen_rmsnorm(rng, keys: List[str]) -> Dict[str, TensorDescriptor]:
    n_rows, n_cols = _random_dim(rng), _random_dim(rng)
    return {
        "x": TensorDescriptor(shape=[n_rows, n_cols], dtype="float32",
                               fill=rng.choice(FILLS), scale=_random_scale_norm(rng), shift=_random_shift(rng)),
        "gamma": TensorDescriptor(shape=[n_cols], dtype="float32",
                                   fill=rng.choice(["ones", "randn"]), scale=_random_scale_norm(rng), shift=0.0),
    }


def _gen_matmul(rng, keys: List[str]) -> Dict[str, TensorDescriptor]:
    """A:[M,K], B:[K,N] -- K SHARED, M/N independent. See module history:
    giving A and B the same [dim1,dim2] shape (the original bug) makes
    the multiplication valid only ~0.1% of the time by coincidence.
    Broad scale range -- no normalization involved, unaffected by the
    eps/scale-invariance finding."""
    M, K, N = _random_dim(rng), _random_dim(rng), _random_dim(rng)
    return {
        "A": _make_tensor(rng, [M, K]),
        "B": _make_tensor(rng, [K, N]),
    }


def _gen_attention_like(rng, keys: List[str]) -> Dict[str, TensorDescriptor]:
    """Q,K,V all share one (N,D) -- self-attention requires this,
    confirmed against the reference kernels' own signatures. Covers
    flash_attention, scaled_dot_product_attention, causal_flash_attention.
    D is drawn from ATTENTION_HEAD_DIMS (power-of-2 only), NOT DIM_RANGE
    -- the Triton kernels use tl.arange(0, D), which requires D to be a
    compile-time power of 2. Confirmed this constraint exists in the
    original flash_attention kernel too, so it's a real deployment
    constraint being modeled correctly here, not worked around."""
    N = _random_dim(rng)
    D = rng.choice(ATTENTION_HEAD_DIMS)
    shape = [N, D]
    return {k: _make_tensor(rng, shape) for k in keys}


def _gen_instancenorm(rng, keys: List[str]) -> Dict[str, TensorDescriptor]:
    """x: (N,C,H,W), weight/bias: (C,). Small dims -- N*C*H*W grows fast
    in 4D; SMALL_DIM_RANGE keeps proposals a reasonable size."""
    N, C, H, W = (_random_dim(rng, SMALL_DIM_RANGE) for _ in range(4))
    return {
        "x": TensorDescriptor(shape=[N, C, H, W], dtype="float32",
                               fill=rng.choice(FILLS), scale=_random_scale_norm(rng), shift=_random_shift(rng)),
        "weight": TensorDescriptor(shape=[C], dtype="float32",
                                    fill=rng.choice(["ones", "randn"]), scale=_random_scale_norm(rng), shift=0.0),
        "bias": TensorDescriptor(shape=[C], dtype="float32",
                                  fill="zeros", scale=1.0, shift=rng.uniform(-5, 5)),
    }


def _gen_batchnorm(rng, keys: List[str]) -> Dict[str, TensorDescriptor]:
    """x: (N,C,H,W), running_mean/running_var/weight/bias: (C,)."""
    N, C, H, W = (_random_dim(rng, SMALL_DIM_RANGE) for _ in range(4))
    return {
        "x": TensorDescriptor(shape=[N, C, H, W], dtype="float32",
                               fill=rng.choice(FILLS), scale=_random_scale_norm(rng), shift=_random_shift(rng)),
        "running_mean": TensorDescriptor(shape=[C], dtype="float32", fill="randn", scale=1.0, shift=0.0),
        "running_var": TensorDescriptor(shape=[C], dtype="float32", fill="ones", scale=1.0, shift=0.0),
        "weight": TensorDescriptor(shape=[C], dtype="float32",
                                    fill=rng.choice(["ones", "randn"]), scale=_random_scale_norm(rng), shift=0.0),
        "bias": TensorDescriptor(shape=[C], dtype="float32",
                                  fill="zeros", scale=1.0, shift=rng.uniform(-5, 5)),
    }


# operator -> (generator_fn, tensor_keys)
GENERATORS = {
    "softmax":         (_gen_softmax_like, ["x"]),
    "log_softmax":     (_gen_softmax_like, ["x"]),
    "sum_reduction":   (_gen_softmax_like, ["x"]),
    "mean_reduction":  (_gen_softmax_like, ["x"]),
    "max_reduction":   (_gen_softmax_like, ["x"]),
    "min_reduction":   (_gen_softmax_like, ["x"]),
    "l1norm":          (_gen_softmax_like, ["x"]),
    "l2norm":          (_gen_softmax_like, ["x"]),
    "argmax":          (_gen_softmax_like, ["x"]),
    "argmin":          (_gen_softmax_like, ["x"]),

    "swish": (_gen_elementwise_1d, ["x"]),
    "gelu":  (_gen_elementwise_1d, ["x"]),

    "frobenius_norm": (_gen_frobenius_norm, ["x"]),

    "layernorm": (_gen_layernorm, ["x", "gamma", "beta"]),
    "rmsnorm":   (_gen_rmsnorm, ["x", "gamma"]),

    "matmul": (_gen_matmul, ["A", "B"]),

    "flash_attention":              (_gen_attention_like, ["Q", "K", "V"]),
    "scaled_dot_product_attention": (_gen_attention_like, ["Q", "K", "V"]),
    "causal_flash_attention":       (_gen_attention_like, ["Q", "K", "V"]),

    "instancenorm": (_gen_instancenorm, ["x", "weight", "bias"]),
    "batchnorm":    (_gen_batchnorm, ["x", "running_mean", "running_var", "weight", "bias"]),
}


def random_proposal(operator: str, worker_id: str, iteration: int, rng: random.Random) -> InputProposal:
    """Generate a random InputProposal with no LLM and no curated
    edge-case shortlist -- shape dims and scale are drawn from broad
    uniform/log-uniform ranges, not hand-picked gotcha values (except
    where a real deployment/numerical constraint requires otherwise --
    see ATTENTION_HEAD_DIMS and NORM_SCALE_RANGE_LOG10 above).
    Dispatches to a per-operator generator (GENERATORS above) rather than
    forcing every operator through one shared shape-generation path --
    the 24 new operators span 1D (swish/gelu), 2D (most), and 4D
    (instancenorm/batchnorm) tensor shapes, which a single "primary_shape"
    variable can't represent correctly."""
    if operator not in GENERATORS:
        raise ValueError(
            f"No proposal generator registered for operator={operator!r}. "
            f"If this is one of BLOCKED_OPERATORS, it's excluded from "
            f"OPERATORS deliberately -- see module docstring, not a bug."
        )
    gen_fn, keys = GENERATORS[operator]
    tensors = gen_fn(rng, keys)

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
    """Prints progress per proposal, with wall-clock time for that
    proposal -- silence for an entire operator's full sweep is
    indistinguishable from a hang."""
    rng = random.Random(seed)
    worker_id = f"random-baseline-{operator}-{seed}"

    for i in range(1, budget + 1):
        t0 = time.perf_counter()
        proposal = random_proposal(operator, worker_id, i, rng)
        hit = _evaluate_proposal(proposal, operator, timeout_seconds)
        elapsed = time.perf_counter() - t0
        print(f"    [{operator}] seed={seed} proposal={i}/{budget} "
              f"({elapsed:.1f}s) {'HIT' if hit else ''}", flush=True)
        if hit:
            return SeedResult(seed=seed, hit=True, proposals_to_hit=i)

    return SeedResult(seed=seed, hit=False, proposals_to_hit=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=25,
                         help="proposals per seed -- match your LLM system's budget for the headline comparison")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=30,
                         help="per-execute_proposal subprocess timeout, seconds")
    parser.add_argument("--checker-root", default=".",
                         help="project root added to sys.path inside each executor subprocess "
                              "(executor.py reads this from the CHECKER_ROOT env var)")
    parser.add_argument("--operators", nargs="*", default=None,
                         help="subset of OPERATORS to run, e.g. --operators softmax gelu -- "
                              "useful for smoke-testing one new operator before a full sweep")
    parser.add_argument("--out", default="random_baseline_report.json")
    args = parser.parse_args()

    os.environ["CHECKER_ROOT"] = args.checker_root

    operators_to_run = args.operators if args.operators else OPERATORS
    unknown = [op for op in operators_to_run if op not in OPERATORS]
    if unknown:
        blocked = [op for op in unknown if op in BLOCKED_OPERATORS]
        msg = f"Unknown operator(s): {unknown}."
        if blocked:
            msg += f" {blocked} are in BLOCKED_OPERATORS -- {[BLOCKED_OPERATORS[b] for b in blocked]}"
        raise ValueError(msg)

    report = {"budget": args.budget, "n_seeds": args.n_seeds, "operators": {}}

    for operator in operators_to_run:
        print(f"\n=== {operator} ({args.n_seeds} seeds x {args.budget} proposals) ===", flush=True)
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