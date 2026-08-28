"""
verification/adversarial_search/reference_failure.py

Classify WHY a reference kernel failed the checker on a proposed input.

Until 2026-08-27 the search had exactly one vocabulary item for this:
"reference failed = invalid input". That conflated two opposite situations:

  domain     The input is outside the kernel's contract (wrong rank, bad
             dtype, a shape the kernel documents it cannot take) or the
             execution errored outright. Expected; not a bug. The historical
             carriers of this outcome are the checks that return a plain
             False on any exception (nan_inf, dtype_preserved) plus
             kernel-execution/shape checks.

  invariant  The reference EXECUTED, produced finite output of the right
             shape and dtype, and that output violates a mathematical
             invariant of the operator itself (rows summing to one, bounds,
             conservation). The reference cannot be wrong about its own
             invariant on an in-contract input unless the reference is buggy.
             This is a REFERENCE-SUSPECT signal, and it is exactly what
             happened three times on 2026-07-23: flash_attention proposals at
             N=130 failed `attention_weights_sum_to_one` because of the
             padded-column masking bug (fixed 2026-08-27), and the search
             filed all three as "invalid input" for a month. See
             verification_runs/attention_mask_bug_impact_2026-08-27/.

The split is deliberately asymmetric: only a curated list of checks may
classify a failure as "domain". Any OTHER failing check -- including checks
added in the future -- classifies as "invariant" and gets surfaced loudly.
Silently absorbing an unknown failure into "invalid input" is the exact
failure mode this module exists to prevent, so the default direction errs
toward a false alarm, never toward silence.

`scripts/review_reference_failures.py` applies the same classification to
accumulated history databases (including records written before this module
existed, via their failure summaries), so the two can never drift apart.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Checks whose failure on the REFERENCE indicates the INPUT is out of
# contract (or the execution errored), not that the reference is wrong.
# Curated and closed on purpose -- see module docstring before adding to it.
DOMAIN_CHECKS = frozenset({
    "nan_inf",             # returns plain False on ANY exception; also fires
                           # on inputs that legitimately overflow
    "dtype_preserved",     # same exception behaviour
    "output_shape",
    "kernel_executed",
})

_SUMMARY_RE = re.compile(r"Reference failed: \[(.*?)\]")


def classify_reference_failure(reference_result) -> Optional[str]:
    """None if the reference passed; else "domain" or "invariant"."""
    if getattr(reference_result, "passed_checker", False):
        return None
    if getattr(reference_result, "error", None) is not None:
        return "domain"
    failed = [r["check_name"] for r in
              (getattr(reference_result, "check_results", None) or [])
              if not r.get("passed", True)]
    return classify_failed_checks(failed)


def classify_failed_checks(failed_check_names: List[str]) -> str:
    """"domain" iff every failed check is a curated domain check."""
    if not failed_check_names:
        # Failed with no per-check record: an execution-level failure.
        return "domain"
    if all(name in DOMAIN_CHECKS for name in failed_check_names):
        return "domain"
    return "invariant"


def invariant_failures(failed_check_names: List[str]) -> List[str]:
    return [n for n in failed_check_names if n not in DOMAIN_CHECKS]


def failed_checks_from_summary(failure_summary: str) -> Optional[List[str]]:
    """Recover the failed-check list from a stored failure_summary string.

    Pre-2026-08-27 verdicts carry only the summary; its format has been
    stable since the first commit: "Reference failed: ['a', 'b']" (truncated
    to three names -- fine for classification, since one non-domain name is
    enough). Returns None when the summary records no reference failure.
    """
    m = _SUMMARY_RE.search(failure_summary or "")
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return []
    return [p.strip().strip("'\"") for p in inner.split(",")]
