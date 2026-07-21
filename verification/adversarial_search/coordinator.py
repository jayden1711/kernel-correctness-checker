"""
verification/adversarial_search/coordinator.py

Search coordinator.

Manages N parallel worker threads, routes proposals to the executor,
evaluates verdicts via the pluggable strategy, persists everything to
SQLite, cancels all workers the instant a hit is confirmed, and
optionally resumes a previous run.

Architecture decisions:
  - Workers in threads (not processes): LLM calls are I/O-bound, not
    CPU-bound.  subprocess isolation is in the executor, not here.
  - Stop event: threading.Event(), checked before every LLM call and
    every executor call.  Instant propagation without polling.
  - Strategy is injected at construction time — coordinator is strategy-agnostic.
  - History store is shared across threads (SQLite WAL mode handles concurrency).
  - Memory items are distilled from confirmed hits and injected into
    subsequent worker prompts — same mechanism as AccelOpt's optimization memory.

Hit invariant (all three must hold):
  1. reference_result.passed_checker  — input is semantically valid
  2. at least one mutant fails checker — bug is exposed
  3. that mutant ALSO passed naive allclose — gap is real (checker found
     what naive testing missed; this is the publishable claim)
"""

from __future__ import annotations
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from verification.adversarial_search.schemas import (
    InputProposal,
    KernelExecutionResult,
    ProposalVerdict,
    SearchResult,
    WorkerFeedback,
)
from verification.adversarial_search.executor import (
    execute_proposal,
    build_feedback_hints,
)
from verification.adversarial_search.worker import AdversarialWorker
from verification.adversarial_search.strategy import SearchStrategy, get_strategy
from verification.adversarial_search.history.store import SearchHistoryStore


# Seed bug patterns per operator — assigned round-robin to workers
_BUG_PATTERNS: Dict[str, List[str]] = {
    "softmax":         ["partial_tile", "wrong_reduction", "missing_max_shift", "boundary_mask"],
    "layernorm":       ["numerical_instab", "ignore_gamma_beta", "wrong_variance", "wrong_axis"],
    "matmul":          ["partial_k_reduct", "skip_boundary", "swapped_strides", "wrong_dtype"],
    "flash_attention": ["drop_last_tile", "skip_rescaling", "approx_denom", "wrong_mask"],
    "rmsnorm":         ["ignore_gamma", "wrong_norm", "partial_reduction", "missing_eps"],
}


class SearchCoordinator:
    """
    Full adversarial input search coordinator.

    Args:
        operator:            "softmax", "layernorm", etc.
        reference_src_path:  Path to reference kernel .py
        mutant_src_paths:    {kernel_id: path} for each mutant
        model:               LiteLLM model string
        strategy:            "greedy" | "beam" | "diverse"
        n_workers:           Parallel worker count (= beam width for beam/diverse)
        max_iterations:      Max proposals per worker before giving up
        timeout_per_exec:    Subprocess timeout in seconds
        output_dir:          Where to write SearchResult JSON + SQLite DB
        resume_run_id:       Resume a previously interrupted run by ID
        diversity_weight:    λ for diverse beam strategy (default 3.0)
        temperature:         LLM temperature (default 0.7)
    """

    def __init__(
        self,
        operator: str,
        reference_src_path: str,
        mutant_src_paths: Dict[str, str],
        model: str,
        strategy: str = "beam",
        n_workers: int = 4,
        max_iterations: int = 20,
        timeout_per_exec: int = 30,
        output_dir: str = "adversarial_results",
        resume_run_id: Optional[str] = None,
        diversity_weight: float = 3.0,
        temperature: float = 0.7,
    ):
        self.operator = operator
        self.reference_src_path = reference_src_path
        self.mutant_src_paths = mutant_src_paths
        self.model = model
        self.n_workers = n_workers
        self.max_iterations = max_iterations
        self.timeout_per_exec = timeout_per_exec
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = temperature

        strategy_kwargs = {}
        if strategy == "diverse":
            strategy_kwargs["diversity_weight"] = diversity_weight
        self.strategy: SearchStrategy = get_strategy(strategy, **strategy_kwargs)

        db_path = self.output_dir / "search_history.db"
        self.store = SearchHistoryStore(str(db_path))

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._all_verdicts: List[ProposalVerdict] = []
        self._winning_proposal: Optional[InputProposal] = None
        self._winning_verdict: Optional[ProposalVerdict] = None
        self._total_proposals = 0

        # Beam: current beam members (proposal, verdict) per worker seed
        self._beam: List[Tuple[InputProposal, ProposalVerdict]] = []

        # Resume context
        self._resume_context: Optional[dict] = None
        if resume_run_id:
            ctx = self.store.resume_run(resume_run_id)
            if ctx:
                self._resume_context = ctx
                self._all_verdicts = ctx["verdicts"]
                self._total_proposals = ctx["n_proposals"]
                print(f"[coordinator] Resuming run {resume_run_id} "
                      f"({self._total_proposals} proposals already done)")
            else:
                print(f"[coordinator] Run {resume_run_id} not found or already finished; "
                      f"starting fresh.")

        self._run_id = (
            resume_run_id
            if (resume_run_id and self._resume_context)
            else str(uuid.uuid4())[:8]
        )

    def run(self) -> SearchResult:
        t_start = time.perf_counter()
        patterns = _BUG_PATTERNS.get(self.operator, ["partial_tile"] * self.n_workers)

        print(f"\n[coordinator] run_id={self._run_id} "
              f"operator={self.operator} strategy={self.strategy.name} "
              f"workers={self.n_workers} model={self.model}")
        print(f"[coordinator] Mutants: {list(self.mutant_src_paths.keys())}")

        if not self._resume_context:
            self.store.create_run(
                run_id=self._run_id,
                operator=self.operator,
                strategy=self.strategy.name,
                model=self.model,
                n_workers=self.n_workers,
                max_iter=self.max_iterations,
            )

        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = [
                pool.submit(
                    self._worker_loop,
                    worker_id=f"w{i}",
                    seed_pattern=patterns[i % len(patterns)],
                    resume_proposal=self._get_resume_proposal(f"w{i}"),
                )
                for i in range(self.n_workers)
            ]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"[coordinator] Worker raised: {e}")

        wall_s = time.perf_counter() - t_start

        result = SearchResult(
            run_id=self._run_id,
            operator=self.operator,
            strategy=self.strategy.name,
            total_proposals=self._total_proposals,
            total_iterations=self.max_iterations * self.n_workers,
            n_workers=self.n_workers,
            winning_proposal=self._winning_proposal,
            winning_verdict=self._winning_verdict,
            all_verdicts=self._all_verdicts,
            wall_time_s=round(wall_s, 2),
            model=self.model,
        )

        self.store.finish_run(self._run_id, result)

        # If we found a hit, distil a memory item for future runs
        if self._winning_proposal:
            self._distil_memory_item(self._winning_proposal, self._winning_verdict)

        out_path = self.output_dir / f"{self.operator}_search_result.json"
        out_path.write_text(result.to_json())

        self._print_summary(result, wall_s)
        return result

    # ── Worker loop ───────────────────────────────────────────────────────────

    def _worker_loop(
        self,
        worker_id: str,
        seed_pattern: str,
        resume_proposal: Optional[InputProposal] = None,
    ):
        memory_items = self.store.get_memory_items(self.operator, limit=4)
        memory_strs = [f"[{m['bug_pattern']}] {m['summary']}" for m in memory_items]

        worker = AdversarialWorker(
            worker_id=worker_id,
            operator=self.operator,
            model=self.model,
            seed_bug_pattern=seed_pattern,
            temperature=self.temperature,
        )

        print(f"[{worker_id}] Started, seed={seed_pattern}, "
              f"memory_items={len(memory_strs)}")

        # Cold start or resume
        if resume_proposal is not None:
            proposal = resume_proposal
            print(f"[{worker_id}] Resuming from proposal {proposal.proposal_id[:8]}")
        else:
            try:
                proposal = worker.propose()
            except Exception as e:
                print(f"[{worker_id}] Initial proposal failed: {e}")
                return

        for iteration in range(self.max_iterations):
            if self._stop_event.is_set():
                print(f"[{worker_id}] Stop at iteration {iteration}.")
                return

            print(f"[{worker_id}] iter={iteration:03d} "
                  f"pid={proposal.proposal_id[:8]} "
                  f"target={proposal.predicted_failure_mode}")

            # Persist proposal before executing (crash-safe)
            self.store.save_proposal(self._run_id, proposal)

            # Execute
            reference_result = execute_proposal(
                proposal=proposal,
                kernel_id="reference",
                candidate_src_path=self.reference_src_path,
                reference_src_path=self.reference_src_path,
                operator=self.operator,
                timeout_seconds=self.timeout_per_exec,
            )

            mutant_results = []
            for kernel_id, path in self.mutant_src_paths.items():
                if self._stop_event.is_set():
                    return
                mr = execute_proposal(
                    proposal=proposal,
                    kernel_id=kernel_id,
                    candidate_src_path=path,
                    reference_src_path=self.reference_src_path,
                    operator=self.operator,
                    timeout_seconds=self.timeout_per_exec,
                )
                mutant_results.append(mr)

            # Score and evaluate
            verdict = self._evaluate_verdict(proposal, reference_result, mutant_results)
            verdict.beam_score = self.strategy.score(proposal, verdict)
            proposal.score = verdict.beam_score

            # Persist verdict
            self.store.save_verdict(self._run_id, verdict)

            with self._lock:
                self._all_verdicts.append(verdict)
                self._total_proposals += 1
                # Update beam
                self._beam.append((proposal, verdict))
                self._beam = self.strategy.select(self._beam, self.n_workers)

            self._log_iteration(worker_id, iteration, verdict)

            # Hit?
            if verdict.is_hit:
                with self._lock:
                    if not self._stop_event.is_set():
                        self._winning_proposal = proposal
                        self._winning_verdict = verdict
                        self._stop_event.set()
                        print(f"[{worker_id}] ✓ HIT — stopping all workers.")
                return

            if self._stop_event.is_set():
                return

            # Get memory items for refinement prompt
            fresh_memory = self.store.get_memory_items(self.operator, limit=3)
            fresh_memory_strs = [f"[{m['bug_pattern']}] {m['summary']}" for m in fresh_memory]

            # Build feedback and refine
            hints = build_feedback_hints(reference_result, mutant_results)
            feedback = WorkerFeedback(
                proposal_id=proposal.proposal_id,
                verdict=verdict,
                hints=hints,
                memory_items=fresh_memory_strs,
            )

            # For beam search: refine from the best beam member, not just our own last proposal
            beam_seed = self._pick_beam_seed(worker_id)
            if beam_seed is not None and beam_seed.proposal_id != proposal.proposal_id:
                print(f"[{worker_id}] Beam: switching seed to {beam_seed.proposal_id[:8]}")
                feedback = WorkerFeedback(
                    proposal_id=beam_seed.proposal_id,
                    verdict=self._verdict_for(beam_seed.proposal_id),
                    hints=hints,
                    memory_items=fresh_memory_strs,
                )

            try:
                proposal = worker.refine(feedback)
            except Exception as e:
                print(f"[{worker_id}] Refinement failed at iteration {iteration}: {e}")
                return

        print(f"[{worker_id}] Exhausted {self.max_iterations} iterations.")

    # ── Verdict evaluation ────────────────────────────────────────────────────

    def _evaluate_verdict(
        self,
        proposal: InputProposal,
        reference_result: KernelExecutionResult,
        mutant_results: List[KernelExecutionResult],
    ) -> ProposalVerdict:
        """
        HIT requires:
          1. reference passes full checker (valid input)
          2. at least one mutant fails full checker (bug exposed)
          3. that mutant passed naive allclose (gap confirmed — checker
             found what naive testing missed)
        """
        reference_passed = reference_result.passed_checker
        hit_mutants = []
        missed_mutants = []
        gap_confirmed = False

        for mr in mutant_results:
            if not mr.passed_checker and mr.passed_naive:
                hit_mutants.append(mr.kernel_id)
                gap_confirmed = True
            else:
                missed_mutants.append(mr.kernel_id)

        is_hit = reference_passed and bool(hit_mutants) and gap_confirmed

        parts = []
        if not reference_passed:
            fails = [r["check_name"] for r in reference_result.check_results
                     if not r["passed"]][:3]
            parts.append(f"Reference failed: {fails}")
        if hit_mutants:
            parts.append(f"Caught (gap confirmed): {hit_mutants}")
        if missed_mutants:
            parts.append(f"Missed: {missed_mutants}")
        if not parts:
            parts.append("All checks passed, no mutants caught.")

        return ProposalVerdict(
            proposal_id=proposal.proposal_id,
            is_hit=is_hit,
            hit_mutants=hit_mutants,
            missed_mutants=missed_mutants,
            reference_passed=reference_passed,
            gap_confirmed=gap_confirmed,
            failure_summary=" | ".join(parts),
        )

    # ── Beam helpers ──────────────────────────────────────────────────────────

    def _pick_beam_seed(self, worker_id: str) -> Optional[InputProposal]:
        """
        For beam search: return the current top beam member's proposal.
        Workers assigned to lower-scoring beam slots seed from the best.
        """
        with self._lock:
            if not self._beam:
                return None
            best = self._beam[0]  # already sorted descending by beam_score
            return best[0]

    def _verdict_for(self, proposal_id: str) -> Optional[ProposalVerdict]:
        with self._lock:
            for v in self._all_verdicts:
                if v.proposal_id == proposal_id:
                    return v
        return None

    # ── Memory distillation ───────────────────────────────────────────────────

    def _distil_memory_item(
        self,
        proposal: InputProposal,
        verdict: Optional[ProposalVerdict],
    ):
        """
        When a hit is confirmed, distil a 1–2 sentence insight and store it.
        This is the "optimization memory" pattern from AccelOpt — future runs
        for the same operator get this injected into their prompts.
        """
        if verdict is None:
            return
        summary = (
            f"Caught {verdict.hit_mutants} using pattern "
            f"'{proposal.predicted_failure_mode}': {proposal.rationale}"
        )
        self.store.add_memory_item(
            operator=self.operator,
            bug_pattern=proposal.predicted_failure_mode,
            summary=summary,
            source_run=self._run_id,
        )

    # ── Resume helpers ────────────────────────────────────────────────────────

    def _get_resume_proposal(self, worker_id: str) -> Optional[InputProposal]:
        if self._resume_context is None:
            return None
        return self._resume_context["last_per_worker"].get(worker_id)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_iteration(
        self,
        worker_id: str,
        iteration: int,
        verdict: ProposalVerdict,
    ):
        status = (
            "HIT     " if verdict.is_hit
            else "ref_fail" if not verdict.reference_passed
            else "miss    "
        )
        print(
            f"[{worker_id}] iter={iteration:03d} "
            f"status={status} score={verdict.beam_score:6.1f} "
            f"hit={verdict.hit_mutants} miss={verdict.missed_mutants}"
        )

    def _print_summary(self, result: SearchResult, wall_s: float):
        if result.winning_proposal:
            print(f"\n[coordinator] ✓ HIT in {self._total_proposals} proposals, "
                  f"{wall_s:.1f}s")
            print(f"[coordinator] Caught: {result.winning_verdict.hit_mutants}")
            print(f"[coordinator] Pattern: {result.winning_proposal.predicted_failure_mode}")
            print(f"[coordinator] Rationale: {result.winning_proposal.rationale}")
        else:
            print(f"\n[coordinator] ✗ No hit after {self._total_proposals} proposals, "
                  f"{wall_s:.1f}s")

        out = self.output_dir / f"{self.operator}_search_result.json"
        print(f"[coordinator] Result: {out}")
        db = self.output_dir / "search_history.db"
        print(f"[coordinator] History DB: {db}")