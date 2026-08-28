"""
verification/adversarial_search/worker.py

One stateful search worker (one per thread in the coordinator).

Responsibilities:
  - Maintain conversation history (sliding window)
  - Call the LLM via litellm (model is a CLI arg)
  - Parse and validate LLM responses into InputProposals
  - Retry failed schema validation with a targeted correction prompt
  - Accept WorkerFeedback and produce refined proposals

Worker is pure from the coordinator's perspective:
  propose() → InputProposal
  refine(feedback) → InputProposal

All prompt content is in prompts/base.py.
All schema validation is in schemas.py.
No tensor construction here — that's the materializer's job.

FIXED: litellm used to be imported at module level. litellm's own
import chain pulls in fastapi -> pydantic -> the full Anthropic SDK
transitively -- a genuinely heavy, multi-second cold import. Every
executor.py subprocess spawn does `from
verification.adversarial_search.schemas import InputProposal`, which
(via this module being imported along the same package chain) paid that
full cost EVERY SINGLE SUBPROCESS, thousands of times across a full
search or random-baseline run, even though neither the random baseline
nor most executor subprocess work ever calls an LLM at all. litellm is
now imported lazily, inside _llm_call, only on the path that actually
needs it.
"""

from __future__ import annotations
import json
import re
import uuid
from typing import Optional

from verification.adversarial_search.schemas import (
    InputProposal,
    TensorDescriptor,
    WorkerFeedback,
    validate_proposal,
)
from verification.adversarial_search.prompts.base import (
    SYSTEM_PROMPT,
    format_first_turn,
    format_refine_turn,
    format_rejection_turn,
)


class ProposalRejected(Exception):
    """
    The model produced output, but it could not be turned into a valid proposal
    within MAX_RETRIES + 1 attempts (bad JSON, missing keys, or a shape-contract
    violation).

    RECOVERABLE, and distinct from an LLM outage on purpose. A transport or API
    failure raises out of _llm_call uncaught and still kills the worker, which is
    correct -- there is nothing to recover from. This exception means the worker
    is healthy and merely produced a bad proposal, so the coordinator resets it
    and keeps going rather than forfeiting the rest of its budget.

    Why this exists: before 2026-08-21 exhaustion raised a bare RuntimeError and
    coordinator._worker_loop caught it with a plain `return`, abandoning every
    remaining iteration. That was measured -- in the 2026-08-21
    causal_flash_attention run, worker w0 died at iteration 13 and forfeited 6
    iterations, which is why that run produced 74 proposals instead of 80. Adding
    shape constraints raises the rejection rate, so leaving that path in place
    would have made 1b a net throughput LOSS.
    """


class AdversarialWorker:
    """
    Stateful LLM agent.  One instance per coordinator thread.
    """

    MAX_HISTORY_TURNS = 6   # last 3 proposal/feedback pairs
    MAX_RETRIES = 2         # retries on JSON schema validation failure

    def __init__(
        self,
        worker_id: str,
        operator: str,
        model: str,
        seed_bug_pattern: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.worker_id = worker_id
        self.operator = operator
        self.model = model
        self.seed_bug_pattern = seed_bug_pattern
        self.temperature = temperature
        self.iteration = 0
        self._history: list[dict] = []

    def propose(self) -> InputProposal:
        """Generate the first proposal (cold start)."""
        user_msg = format_first_turn(self.operator, self.seed_bug_pattern)
        return self._call_and_parse(user_msg)

    def refine(self, feedback: WorkerFeedback) -> InputProposal:
        """Generate a refined proposal based on coordinator feedback."""
        user_msg = format_refine_turn(feedback)
        return self._call_and_parse(user_msg)

    # ── Private ───────────────────────────────────────────────────────────────

    def _call_and_parse(self, user_msg: str) -> InputProposal:
        self._history.append({"role": "user", "content": user_msg})
        self._trim_history()

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        last_error = None
        raw = ""
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                raw = self._llm_call(messages)
                proposal = self._parse(raw)
                self._history.append({"role": "assistant", "content": raw})
                self._trim_history()
                self.iteration += 1
                return proposal

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": format_rejection_turn(e, self.operator),
                        },
                    ]

        # Leave the worker in a CLEAN state so the caller can retry.
        #
        # _call_and_parse appends the user message to self._history on entry but
        # appends the assistant reply only on SUCCESS. Without this pop, a failed
        # call leaves a dangling user turn with no reply, and the next call
        # appends a second user message straight after it -- two user turns in a
        # row, which is malformed for most chat APIs and poisons every subsequent
        # attempt by this worker. Recovery in coordinator._worker_loop depends on
        # the worker still being usable after a rejection, so this matters.
        if self._history and self._history[-1].get("role") == "user":
            self._history.pop()

        raise ProposalRejected(
            f"Worker {self.worker_id} failed to produce a valid proposal after "
            f"{self.MAX_RETRIES + 1} attempts. Last error: {last_error}"
        )

    def _llm_call(self, messages: list[dict]) -> str:
        import litellm  # lazy -- see module docstring
        response = litellm.completion(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def _parse(self, raw: str) -> InputProposal:
        # Strip markdown fences if model adds them despite instructions
        clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        clean = re.sub(r"\s*```$", "", clean.strip(), flags=re.MULTILINE)

        data = json.loads(clean.strip())

        tensors = {}
        for name, td in data["tensors"].items():
            tensors[name] = TensorDescriptor(
                shape=td["shape"],
                dtype=td.get("dtype", "float32"),
                fill=td["fill"],
                scale=float(td.get("scale", 1.0)),
                shift=float(td.get("shift", 0.0)),
                literal_values=td.get("literal_values"),
                patches=td.get("patches", []),
            )

        proposal = InputProposal(
            proposal_id=str(uuid.uuid4()),
            worker_id=self.worker_id,
            iteration=self.iteration,
            operator=self.operator,
            tensors=tensors,
            rationale=data.get("rationale", ""),
            predicted_failure_mode=data.get("predicted_failure_mode", ""),
        )

        ok, err = validate_proposal(proposal)
        if not ok:
            raise ValueError(f"Proposal failed schema validation: {err}")

        return proposal

    def _trim_history(self):
        max_msgs = self.MAX_HISTORY_TURNS
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]