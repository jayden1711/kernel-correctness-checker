#!/usr/bin/env python3
"""Review accumulated reference-failed search verdicts for patterns that look
like a reference-kernel bug rather than expected domain rejection.

Usage:
    python scripts/review_reference_failures.py [db ...]

With no arguments, scans every adversarial_results/**/search_history.db.

For every verdict where the reference failed the checker, the failed-check
list is recovered (from the stored `reference_failure_kind` when present, and
for pre-2026-08-27 records from the failure summary) and classified with the
SAME rule the coordinator applies (reference_failure.classify_failed_checks):
a failure is "domain" only if every failing check is on the curated
domain-check list; anything else is "invariant" = REFERENCE-SUSPECT.

Exit status: 0 when nothing is reference-suspect, 2 otherwise -- so this can
run as a post-search assertion or in CI. Rationale and history:
verification_runs/attention_mask_bug_impact_2026-08-27/FINDINGS.md §5
(three real reference-bug detections sat mislabeled as "invalid input" for a
month).
"""

import glob
import json
import os
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from verification.adversarial_search.reference_failure import (  # noqa: E402
    classify_failed_checks,
    failed_checks_from_summary,
    invariant_failures,
)


def scan(db_path):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT v.proposal_id, v.run_id, v.operator, v.verdict_json, "
        "       p.proposal_json "
        "FROM verdicts v LEFT JOIN proposals p ON v.proposal_id = p.proposal_id"
    ).fetchall()
    con.close()
    groups = defaultdict(list)
    for r in rows:
        vj = json.loads(r["verdict_json"])
        if vj.get("reference_passed", True):
            continue
        kind = vj.get("reference_failure_kind")
        failed = failed_checks_from_summary(vj.get("failure_summary", ""))
        if kind is None:
            kind = classify_failed_checks(failed if failed is not None else [])
        shape = None
        if r["proposal_json"]:
            pj = json.loads(r["proposal_json"])
            shapes = {tuple(t["shape"]) for t in pj.get("tensors", {}).values()}
            shape = sorted(shapes)
        groups[(r["operator"], kind, tuple(failed or []))].append(
            (r["proposal_id"][:8], shape))
    return groups


def main():
    dbs = sys.argv[1:] or sorted(
        glob.glob("adversarial_results/**/search_history.db", recursive=True))
    suspect = 0
    for db in dbs:
        groups = scan(db)
        if not groups:
            continue
        print(f"== {db}")
        for (op, kind, failed), items in sorted(groups.items()):
            tag = "REFERENCE-SUSPECT" if kind == "invariant" else "domain"
            if kind == "invariant":
                suspect += len(items)
            print(f"  [{tag:17s}] {op:28s} n={len(items):3d} "
                  f"failed={list(failed)}"
                  + (f" invariants={invariant_failures(list(failed))}"
                     if kind == "invariant" else ""))
            for pid, shape in items[:5]:
                print(f"      {pid} shapes={shape}")
    if suspect:
        print(f"\n{suspect} REFERENCE-SUSPECT verdict(s): the reference kernel "
              f"violated its own operator invariant on an executed input. "
              f"Investigate the reference before trusting these runs' "
              f"'invalid input' bookkeeping.")
        return 2
    print("\nNo reference-suspect verdicts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
