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
from dataclasses import dataclass, field, asdict
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
    missed_mutants: List[str]
    reference_passed: bool
    gap_confirmed: bool
    failure_summary: str
    # Numeric score for beam search ranking (higher = better)
    beam_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProposalVerdict":
        return cls(**d)


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
}

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
    return True, ""