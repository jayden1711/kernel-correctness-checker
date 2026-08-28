"""
verification/adversarial_search/schemas.py

Typed JSON contracts shared across every stage of the adversarial
input search pipeline.  All inter-stage communication is validated
against these schemas — LLM output, worker proposals, execution
results, coordinator decisions.

Design principle: the schema IS the interface.  No stage should ever
inspect raw exception text or raw tensor values directly; everything
goes through a parse step that produces one of these types.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional, Tuple
import json
import uuid


# ── Tensor descriptors ────────────────────────────────────────────────────────

@dataclass
class TensorDescriptor:
    """
    A JSON-serialisable description of a single input tensor.
    Workers never send raw tensors — they send descriptors, and the
    executor reconstructs tensors from them deterministically.
    """
    shape: List[int]
    dtype: str                         # "float32", "float16", etc.
    fill: str                          # "randn", "ones", "zeros", "arange", "literal"
    scale: float = 1.0
    shift: float = 0.0
    literal_values: Optional[List[float]] = None
    patches: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TensorDescriptor":
        return cls(**d)


@dataclass
class InputProposal:
    """
    One candidate adversarial input from a worker.
    All tensor arguments are described symbolically.
    """
    proposal_id: str
    worker_id: str
    iteration: int
    operator: str
    tensors: Dict[str, TensorDescriptor]
    rationale: str
    predicted_failure_mode: str
    # Score assigned by the strategy (higher = more promising beam member)
    score: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "InputProposal":
        d = dict(d)
        d["tensors"] = {k: TensorDescriptor.from_dict(v) for k, v in d["tensors"].items()}
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── Execution results ─────────────────────────────────────────────────────────

@dataclass
class ExecutionError:
    error_type: str
    message: str
    layer: Optional[str]
    check_name: Optional[str]
    max_err: Optional[float]
    traceback_snippet: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionError":
        return cls(**d)


@dataclass
class KernelExecutionResult:
    """
    Result of running one kernel on one proposal.
    Produced by the executor; consumed by the coordinator.
    """
    proposal_id: str
    kernel_id: str
    passed_checker: bool
    passed_naive: bool
    error: Optional[ExecutionError]
    check_results: List[Dict]
    wall_time_ms: float
    # Total wall time of execute_proposal(), measured in the PARENT around
    # Process.start()/join(). `wall_time_ms` above is measured INSIDE the
    # subprocess and covers only the kernel call, so the difference between the
    # two is subprocess spawn + `import torch`/triton + CUDA init.
    #
    # That gap is not a rounding detail: on the 2026-08-20 causal_flash_attention
    # run the median in-kernel time was 0.03s while the median spawn-to-result
    # interval was 10.25s, and 160 executions of that made process startup ~71%
    # of each worker's wall time -- the single largest cost in the search, and
    # completely invisible in the persisted data because nothing measured it.
    # Defaults to None so results built by older code (or loaded from a
    # pre-migration DB) stay constructible.
    #
    # BATCHED EXECUTIONS LEAVE THIS NULL, ON PURPOSE. When N+1 kernels share
    # one subprocess there is no parent-observed spawn-to-result interval for
    # any individual kernel -- the interval genuinely does not exist, and
    # inventing one by dividing the batch's would be a fabricated number of
    # exactly the kind this field was added to expose. Read `batch_spawn_ms`
    # plus `kernel_wall_time_ms` for batched rows; check `exec_mode` to know
    # which applies. NULL still means "never measured", never "free".
    total_wall_time_ms: Optional[float] = None

    # Which executor path produced this result: "single" (one subprocess per
    # kernel, the original path, still used as the batch fallback) or "batched"
    # (one subprocess per proposal). This is a DISCRIMINATOR, not decoration:
    # total_wall_time_ms and batch_spawn_ms are populated on mutually exclusive
    # paths, so any analysis that averages across both must group by this first.
    exec_mode: str = "single"

    # Child-measured startup for the whole batch: process entry through imports,
    # CUDA init and materialization, i.e. everything paid ONCE no matter how
    # many kernels the batch runs. Identical on every row of the same batch, so
    # a per-proposal spawn cost is `SELECT DISTINCT` per proposal_id -- summing
    # it across rows double-counts. NULL on the single path.
    batch_spawn_ms: Optional[float] = None

    # Child-measured interval for THIS kernel alone: module load, the naive
    # allclose pair, and the full KernelChecker run. Populated on BOTH paths,
    # which is what makes batched and single executions comparable at all.
    # Distinct from wall_time_ms, which times only the candidate call.
    kernel_wall_time_ms: Optional[float] = None

    # Decomposition of the startup that `batch_spawn_ms` totals: interpreter +
    # multiprocessing bootstrap, `import torch`, the per-operator spec imports,
    # CUDA context init, and materialization. Added because the ~10.3s above was
    # a single opaque number, and which of process-reuse strategies is worth its
    # risk depends entirely on the import-vs-CUDA-init split inside it.
    startup_phases: Optional[Dict[str, float]] = None

    # Which multiprocessing start method actually created this execution's
    # process: "spawn" or "forkserver". STAMPED BY THE PARENT, which is the only
    # party that knows what it asked for AND what it got.
    #
    # A SEPARATE FIELD, NOT AN `exec_mode` SUFFIX, DELIBERATELY. `exec_mode` is
    # read with an exact string compare (`scripts/analyze_spawn_cost.py:144`
    # does `exec_mode == "batched"`), so widening its vocabulary to
    # "batched:forkserver" would make that tool silently report ZERO batched
    # executions -- the silent exact-match lookup this project has now hit four
    # times. The two axes are orthogonal anyway: batching sets how many
    # processes per proposal, the start method sets how each one is created.
    #
    # It exists mainly to make a SILENT FALLBACK visible. `_mp_context` drops to
    # spawn when forkserver is unavailable, and without this field such a run
    # would report "forkserver made no difference" for a run in which forkserver
    # never ran -- structurally the same defect as a subprocess timing its own
    # startup. NULL means a pre-migration row, never "spawn".
    start_method: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.error:
            d["error"] = self.error.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "KernelExecutionResult":
        d = dict(d)
        if d.get("error"):
            d["error"] = ExecutionError.from_dict(d["error"])
        return cls(**d)


@dataclass
class ProposalVerdict:
    """
    Coordinator decision on one proposal.

    HIT requires ALL of:
      1. reference_passed  — input is semantically valid
      2. hit_mutants non-empty — bug exposed by checker
      3. gap_confirmed — at least one hit mutant ALSO passed naive allclose
         (the bug is invisible to naive testing — this is the publishable claim)
    """
    proposal_id: str
    is_hit: bool
    hit_mutants: List[str]
    # NOTE: `missed_mutants` conflates two OPPOSITE outcomes and is kept only
    # for backward compatibility -- it equals not_caught + caught_no_gap. Every
    # pre-existing consumer (the history DB column, the beam/greedy strategies,
    # the worker feedback template) keeps reading it unchanged, so stored runs
    # stay comparable. For any new diagnosis use the two precise lists below.
    missed_mutants: List[str]
    reference_passed: bool
    gap_confirmed: bool
    failure_summary: str
    # Numeric score for beam search ranking (higher = better)
    beam_score: float = 0.0

    # ── added: the split that makes a non-hit interpretable ───────────────
    # A mutant landed in `missed_mutants` when EITHER the checker failed to
    # catch it, OR the checker caught it but naive allclose caught it too (so
    # there was no gap worth reporting). Those are opposite results for this
    # project's central claim and were recorded identically, which is why the
    # causal_flash_attention non-hit could not be diagnosed from its own output
    # and had to be reconstructed from proposal shapes months later. See
    # adversarial_results/CFA_NONHIT_ROOTCAUSE.md section 4.
    not_caught: List[str] = field(default_factory=list)
    caught_no_gap: List[str] = field(default_factory=list)
    # Per-mutant raw outcomes: {kernel_id, passed_checker, passed_naive,
    # outcome}. The executor computed both booleans on every run and the
    # verdict threw them away; this is the additive record that keeps them,
    # same mechanism as the per-check records in the benchmark harness.
    mutant_records: List[Dict] = field(default_factory=list)

    # WHY the reference failed, when it did. None when reference_passed.
    #   "domain"    — input out of the kernel's contract, or execution error.
    #                 Expected; not a bug.
    #   "invariant" — the reference ran and its own output violates an
    #                 operator invariant: REFERENCE-SUSPECT. A single
    #                 "reference failed" bucket hid the flash_attention
    #                 masking bug for a month (three N=130 verdicts,
    #                 2026-07-23); see reference_failure.py and
    #                 verification_runs/attention_mask_bug_impact_2026-08-27/.
    # Classification lives in reference_failure.classify_reference_failure —
    # the single source of truth also used by
    # scripts/review_reference_failures.py.
    reference_failure_kind: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProposalVerdict":
        # Tolerant of both directions of schema drift: verdicts stored BEFORE
        # the fields above existed simply get the defaults, and unknown keys
        # from a newer writer are dropped instead of raising TypeError. The
        # history store round-trips these from SQLite, so a hard failure here
        # would make every pre-existing run unreadable.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class WorkerFeedback:
    """Structured feedback from coordinator back to one worker."""
    proposal_id: str
    verdict: ProposalVerdict
    hints: List[str]
    # Memory items relevant to this operator (injected by coordinator)
    memory_items: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchResult:
    """Final output for one operator × mutant set search run."""
    run_id: str
    operator: str
    strategy: str
    total_proposals: int
    total_iterations: int
    n_workers: int
    winning_proposal: Optional[InputProposal]
    winning_verdict: Optional[ProposalVerdict]
    all_verdicts: List[ProposalVerdict]
    wall_time_s: float
    model: str

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.winning_proposal:
            d["winning_proposal"] = self.winning_proposal.to_dict()
        if self.winning_verdict:
            d["winning_verdict"] = self.winning_verdict.to_dict()
        d["all_verdicts"] = [v.to_dict() for v in self.all_verdicts]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── Validation ────────────────────────────────────────────────────────────────

REQUIRED_TENSOR_KEYS = {
    "softmax":         ["x"],
    "layernorm":       ["x", "gamma", "beta"],
    "matmul":          ["A", "B"],
    "flash_attention": ["Q", "K", "V"],
    "rmsnorm":         ["x", "gamma"],

    # Front-door wiring for the 16 operators executor.py/materializer.py
    # already support at the execution layer (see materializer.py's
    # _OPERATOR_TENSOR_KEYS, which this must match exactly) -- without an
    # entry here, validate_proposal() hard-rejects every proposal for
    # these operators with "Unknown operator", even though the executor
    # could run them.
    "log_softmax":                  ["x"],
    "swish":                        ["x"],
    "gelu":                         ["x"],
    "sum_reduction":                ["x"],
    "mean_reduction":               ["x"],
    "max_reduction":                ["x"],
    "min_reduction":                ["x"],
    "l1norm":                       ["x"],
    "l2norm":                       ["x"],
    "frobenius_norm":               ["x"],
    "argmax":                       ["x"],
    "argmin":                       ["x"],
    "instancenorm":                 ["x", "weight", "bias"],
    "batchnorm":                    ["x", "running_mean", "running_var", "weight", "bias"],
    "scaled_dot_product_attention": ["Q", "K", "V"],
    "causal_flash_attention":       ["Q", "K", "V"],
}

# ── Shape constraints ─────────────────────────────────────────────────────────
#
# WHY THIS EXISTS. Before 2026-08-21, validate_proposal() checked tensor KEYS and
# fill validity and nothing else. A proposal could therefore ask for a rank-4 Q
# on causal_flash_attention, pass validation, materialise fine, and only explode
# inside the reference kernel -- after a ~10s subprocess spawn per kernel. On the
# 2026-08-21 causal_flash_attention run that was 12 of 74 proposals (16% of the
# run): reference AND mutant both crashed, so the proposal yielded no comparison
# at all. Worse, those crashes were recorded as `nan_inf` + `dtype_preserved`
# check failures against the REFERENCE, which published a false 17.1%
# false-positive rate (BENCHMARK_RESULTS.md §8.3.1, since corrected).
#
# ─────────────────────────────────────────────────────────────────────────────
# THE RULE FOR EDITING THIS TABLE. Read it before adding an entry.
#
#   Constraints are DERIVED FROM REFERENCE-KERNEL SOURCE. Historical data can
#   only FALSIFY a constraint, never confirm one.
#
# It is tempting to infer a constraint from observed proposals -- "every
# layernorm proposal that worked used powers of two, so layernorm must require
# them". That inference is invalid and this table must never contain one. When
# it was checked, layernorm had exactly ONE historical passing proposal, matmul
# two, rmsnorm two. "Always a power of two" in that data reflects what the LLM
# happened to propose, nothing about the kernel.
#
# An ABSENT constraint is always safe -- it costs at worst a wasted iteration.
# An INVENTED constraint silently suppresses legitimate adversarial inputs and
# hides real bugs, which is the opposite of what this search exists to do. A
# blanket "all dims must be powers of two" rule, for instance, would have
# rejected 6 CONFIRMED HITS: softmax [512,777] and [512,333] (first_tile,
# wrong_reduction), flash_attention [96,64] (approx_denom), and gelu [33,33] /
# [128,160] (sigmoid_approx). Softmax's non-power-of-two reduction dim is
# precisely what exposes partial-tile bugs.
#
# So: every entry cites the source line that justifies it. No citation, no entry.
# tests/instrumentation/check_shape_constraints.py replays every historical
# proposal through this table and FAILS if any input whose reference actually
# passed would now be rejected.
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT IS **NOT** CONSTRAINED, AND WHY -- do not "helpfully" add these:
#
#   * n_cols for every row-wise operator (softmax, log_softmax, layernorm,
#     rmsnorm, l1norm, l2norm, the four reductions, argmax, argmin). These
#     kernels do `BLOCK_SIZE = triton.next_power_of_2(n_cols)` and mask with
#     `col_offsets < n_cols` -- they compute the power of two FOR you. A
#     non-power-of-two column count is the intended adversarial case.
#   * N (sequence length) for the three attention kernels. BLOCK_M/BLOCK_N are
#     compile-time 32, so N never enters a tl.dot shape; it is masked. In
#     particular N >= 16 is NOT a real constraint, despite prompts/base.py
#     having claimed so until 2026-08-21.
#   * Any upper bound on D. Large D exceeds register/shared-memory limits at
#     some point, but the threshold is hardware- and Triton-version-specific.
#     No number is invented here.
#   * dtype. materializer.py builds every tensor float32, so non-fp32 inputs are
#     currently unreachable from the search.

def _is_pow2(n: int) -> bool:
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


# Per-tensor rules. Keys are tensor names; values describe that tensor.
#   rank      : int (exact) | (lo, None) for ">= lo" | None for unconstrained
#   pow2_dims : tuple of dim indices that MUST be powers of two (-1 = last)
#   min_dims  : {dim_index: minimum}
# Cross-tensor rules live in `cross`, evaluated after per-tensor rules pass.
#
# CRASH vs SILENT-GARBAGE. Most rules below prevent an outright raise. Three
# kinds prevent something subtler -- inputs the kernel ACCEPTS but computes
# nonsense from, making the reference/mutant comparison meaningless. They are
# marked `# silent-garbage` so they can be relaxed as a group if they ever prove
# too strict; they are NOT needed to stop crashes.
SHAPE_CONSTRAINTS: Dict[str, Dict[str, Any]] = {

    # ---- rank-2 row-wise ops: `n_rows, n_cols = x.shape`, no pow2 anywhere ----
    # softmax.py:35, log_softmax.py:30, sum_reduction.py:25, mean_reduction.py:23,
    # max_reduction.py:24, min_reduction.py:24, l1norm.py:25, l2norm.py:26,
    # argmax.py:32, argmin.py:27 -- all unpack exactly two, all mask n_cols.
    "softmax":        {"tensors": {"x": {"rank": 2}}},
    "log_softmax":    {"tensors": {"x": {"rank": 2}}},
    "sum_reduction":  {"tensors": {"x": {"rank": 2}}},
    "mean_reduction": {"tensors": {"x": {"rank": 2}}},
    "max_reduction":  {"tensors": {"x": {"rank": 2}}},
    "min_reduction":  {"tensors": {"x": {"rank": 2}}},
    "l1norm":         {"tensors": {"x": {"rank": 2}}},
    "l2norm":         {"tensors": {"x": {"rank": 2}}},
    "argmax":         {"tensors": {"x": {"rank": 2}}},
    "argmin":         {"tensors": {"x": {"rank": 2}}},

    # ---- rank-agnostic elementwise / global ops ----
    # swish.py:20, gelu.py:22, frobenius_norm.py:45 all do
    # `x.contiguous().view(-1)` with a hardcoded BLOCK_SIZE=1024. ANY rank is
    # legal. Deliberately no rank entry -- see the editing rule above.
    "swish":          {"tensors": {"x": {}}},
    "gelu":           {"tensors": {"x": {}}},
    "frobenius_norm": {"tensors": {"x": {}}},

    # ---- normalisations with per-column parameter vectors ----
    # layernorm.py:37 / rmsnorm.py:37 `n_rows, n_cols = x.shape`.
    # gamma/beta are loaded at layernorm.py:27 with mask=mask, other=1.0 -- but
    # a SHORT gamma still reads out of bounds within the masked lanes, so the
    # element count is a real requirement.
    "layernorm": {
        "tensors": {"x": {"rank": 2}, "gamma": {"rank": 1}, "beta": {"rank": 1}},
        "cross": [("numel_ge_dim", "gamma", "x", 1),      # silent-garbage
                  ("numel_ge_dim", "beta",  "x", 1)],     # silent-garbage
    },
    "rmsnorm": {
        "tensors": {"x": {"rank": 2}, "gamma": {"rank": 1}},
        "cross": [("numel_ge_dim", "gamma", "x", 1)],     # silent-garbage
    },

    # ---- matmul ----
    # mat_mult.py:48-49 does `M, K = A.shape` then `K, N = B.shape`, DISCARDING
    # the first K with no assert. Mismatched inner dims therefore do not raise --
    # they read out of bounds or truncate the reduction, so both reference and
    # mutant return garbage and the comparison is meaningless.
    # M/N/K are all masked and BLOCK_* are compile-time 32, so tl.dot's >=16 rule
    # is satisfied regardless of input: there is NO min-dim constraint here.
    "matmul": {
        "tensors": {"A": {"rank": 2}, "B": {"rank": 2}},
        "cross": [("dim_eq", "A", 1, "B", 0)],            # silent-garbage
    },

    # ---- instancenorm / batchnorm ----
    # instancenorm.py:42-43 `N, C = x.shape[0], x.shape[1]`, then x.shape[2:] is
    # flattened -- so rank >= 2, with no upper bound enforced by the wrapper.
    # instancenorm.py:30 loads gamma UNMASKED, so weight/bias really must cover C.
    "instancenorm": {
        "tensors": {"x": {"rank": (2, None)}, "weight": {"rank": 1}, "bias": {"rank": 1}},
        "cross": [("numel_ge_dim", "weight", "x", 1),     # silent-garbage
                  ("numel_ge_dim", "bias",   "x", 1)],    # silent-garbage
    },
    "batchnorm": {
        "tensors": {"x": {"rank": (2, None)}, "running_mean": {"rank": 1},
                    "running_var": {"rank": 1}, "weight": {"rank": 1},
                    "bias": {"rank": 1}},
        "cross": [("numel_ge_dim", "running_mean", "x", 1),   # silent-garbage
                  ("numel_ge_dim", "running_var",  "x", 1),   # silent-garbage
                  ("numel_ge_dim", "weight",       "x", 1),   # silent-garbage
                  ("numel_ge_dim", "bias",         "x", 1)],  # silent-garbage
    },

    # ---- attention family: the ONLY operators with a power-of-two rule ----
    # flash_attention.py:75, scaled_dot_product_attention.py:64,
    # causal_flash_attention.py:69 all do `N, D = Q.shape` -> Q is exactly rank 2.
    # D is passed as `D: tl.constexpr` and reaches `d_offsets = tl.arange(0, D)`
    # (causal_flash_attention.py:24) -- Triton REQUIRES an arange range to be a
    # power of two, so a non-pow2 D fails to COMPILE. tl.dot at :45/:56 forces
    # D >= 16. N is masked and unconstrained.
    # K/V rank: the wrapper validates only Q; a rank-3 K silently mis-reads via
    # K.stride(1) rather than raising, hence the silent-garbage marking.
    **{
        op: {
            "tensors": {
                "Q": {"rank": 2, "pow2_dims": (-1,), "min_dims": {-1: 16}},
                "K": {"rank": 2},                          # silent-garbage
                "V": {"rank": 2},                          # silent-garbage
            },
            "cross": [("dim_eq", "K", -1, "Q", -1),        # silent-garbage
                      ("dim_eq", "V", -1, "Q", -1)],       # silent-garbage
        }
        for op in ("flash_attention", "scaled_dot_product_attention",
                   "causal_flash_attention")
    },
}


def validate_shape_constraints(proposal: InputProposal) -> Optional[str]:
    """
    Check a proposal's shapes against SHAPE_CONSTRAINTS.

    Returns None if acceptable, else a SPECIFIC human-readable reason. The
    reason string is fed back to the model verbatim on the retry turn
    (worker.py -> prompts.format_rejection_turn), so it must say what is wrong
    AND what the operator actually wants -- "invalid shape" teaches nothing.

    An operator with no entry is NOT rejected. Absence means "no constraint
    derived from source", which is the safe default.
    """
    spec = SHAPE_CONSTRAINTS.get(proposal.operator)

    # Universal, and deliberately NOT source-derived: a dimension of 0 or a
    # negative dimension. Whether triton.next_power_of_2(0) raises is
    # version-specific and was not verified, so this is a judgement call rather
    # than a derived rule -- justified because a zero-element tensor carries no
    # adversarial signal, so rejecting it cannot suppress anything meaningful.
    for name, desc in proposal.tensors.items():
        for i, d in enumerate(desc.shape):
            if not isinstance(d, int) or d < 1:
                return (f"Tensor '{name}' has dimension {i} = {d}; every "
                        f"dimension must be a positive integer.")

    if spec is None:
        return None

    for name, rules in spec.get("tensors", {}).items():
        desc = proposal.tensors.get(name)
        if desc is None:          # missing keys are validate_proposal's job
            continue
        shape = list(desc.shape)
        rank = rules.get("rank")
        if isinstance(rank, int) and len(shape) != rank:
            return (f"Tensor '{name}' must be rank {rank} for operator "
                    f"'{proposal.operator}', got rank {len(shape)} {shape}.")
        if isinstance(rank, tuple) and len(shape) < rank[0]:
            return (f"Tensor '{name}' must be rank >= {rank[0]} for operator "
                    f"'{proposal.operator}', got rank {len(shape)} {shape}.")
        for idx in rules.get("pow2_dims", ()):
            if idx < len(shape) or idx < 0:
                val = shape[idx]
                if not _is_pow2(val):
                    return (f"Tensor '{name}' dimension {idx} = {val} must be a "
                            f"power of two for operator '{proposal.operator}' "
                            f"(it is passed as a tl.constexpr into tl.arange, "
                            f"which requires a power-of-two range). Use e.g. "
                            f"16, 32, 64 or 128.")
        for idx, lo in rules.get("min_dims", {}).items():
            if idx < len(shape) or idx < 0:
                val = shape[idx]
                if val < lo:
                    return (f"Tensor '{name}' dimension {idx} = {val} must be "
                            f">= {lo} for operator '{proposal.operator}' "
                            f"(tl.dot requires every participating dimension "
                            f">= 16).")

    for rule in spec.get("cross", ()):
        kind = rule[0]
        if kind == "dim_eq":
            _, a, ai, b, bi = rule
            da, db = proposal.tensors.get(a), proposal.tensors.get(b)
            if da is None or db is None:
                continue
            if abs(ai) <= len(da.shape) and abs(bi) <= len(db.shape):
                if da.shape[ai] != db.shape[bi]:
                    return (f"Tensor '{a}' dimension {ai} ({da.shape[ai]}) must "
                            f"equal tensor '{b}' dimension {bi} "
                            f"({db.shape[bi]}) for operator "
                            f"'{proposal.operator}'. Mismatched dimensions here "
                            f"do not raise -- they silently produce incorrect "
                            f"output, so the comparison would be meaningless.")
        elif kind == "numel_ge_dim":
            _, vec, ref, ref_dim = rule
            dv, dr = proposal.tensors.get(vec), proposal.tensors.get(ref)
            if dv is None or dr is None:
                continue
            if abs(ref_dim) <= len(dr.shape):
                need = dr.shape[ref_dim]
                have = 1
                for d in dv.shape:
                    have *= d
                if have < need:
                    return (f"Tensor '{vec}' has {have} element(s) but must "
                            f"have at least {need} (matching '{ref}' dimension "
                            f"{ref_dim}) for operator '{proposal.operator}'. "
                            f"It is indexed without a bounds mask, so a shorter "
                            f"vector reads out of bounds instead of raising.")
    return None


def validate_shape_constraint_coverage() -> List[str]:
    """
    Assert SHAPE_CONSTRAINTS covers exactly the operators REQUIRED_TENSOR_KEYS
    does. Returns the symmetric difference, empty when aligned.

    This is a deliberately NARROW check over two tables. It is not §2.3's B2
    item (enforcing agreement across all seven parallel operator tables), which
    remains open and unscoped -- but keeping these two in lockstep makes B2
    strictly easier later rather than pre-empting it. Same pattern as
    prompts.validate_bug_pattern_hints().
    """
    missing = set(REQUIRED_TENSOR_KEYS) - set(SHAPE_CONSTRAINTS)
    extra = set(SHAPE_CONSTRAINTS) - set(REQUIRED_TENSOR_KEYS)
    return sorted([f"missing from SHAPE_CONSTRAINTS: {m}" for m in missing] +
                  [f"unknown operator in SHAPE_CONSTRAINTS: {e}" for e in extra])


_coverage_gaps = validate_shape_constraint_coverage()
if _coverage_gaps:
    raise ValueError(
        "SHAPE_CONSTRAINTS and REQUIRED_TENSOR_KEYS disagree: "
        + "; ".join(_coverage_gaps)
        + ". Every wired operator needs an explicit entry -- use an empty "
          "'tensors' dict to say 'no constraints derived from source', so that "
          "an absent constraint is visibly deliberate rather than an oversight."
    )


def validate_proposal(proposal: InputProposal) -> Tuple[bool, str]:
    expected = REQUIRED_TENSOR_KEYS.get(proposal.operator)
    if expected is None:
        return False, f"Unknown operator: {proposal.operator}"
    missing = [k for k in expected if k not in proposal.tensors]
    if missing:
        return False, f"Missing tensor keys: {missing}"
    for name, desc in proposal.tensors.items():
        if not desc.shape:
            return False, f"Tensor '{name}' has empty shape"
        valid_fills = {"randn", "ones", "zeros", "arange", "literal"}
        if desc.fill not in valid_fills:
            return False, f"Tensor '{name}' has invalid fill: {desc.fill!r}"
        if desc.fill == "literal" and desc.literal_values is None:
            return False, f"Tensor '{name}' fill=literal but no literal_values"

    # Shape contract. Runs last so the cheaper key/fill errors surface first.
    shape_err = validate_shape_constraints(proposal)
    if shape_err:
        return False, shape_err
    return True, ""