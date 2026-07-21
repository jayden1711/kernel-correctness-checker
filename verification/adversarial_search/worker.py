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
"""

from __future__ import annotations
import json
import re
import uuid
from typing import Optional

import litellm

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
)


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
                            "content": (
                                f"Your response failed schema validation: {e}\n"
                                f"Respond with ONLY the JSON proposal schema. "
                                f"No markdown, no text outside the JSON object."
                            ),
                        },
                    ]

        raise RuntimeError(
            f"Worker {self.worker_id} failed to produce valid JSON after "
            f"{self.MAX_RETRIES + 1} attempts. Last error: {last_error}"
        )

    def _llm_call(self, messages: list[dict]) -> str:
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