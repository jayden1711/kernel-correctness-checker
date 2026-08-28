"""Extract the 42 non-flash reference-suspect verdicts with full tensor
configurations (shapes, fills, scales, shifts, patch counts) for the
claims-affected sweep (CLAIMS_SWEEP.md). Writes to stdout; banked output in
data/flagged42_detail.log."""

import json
import os
import sqlite3
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
sys.path.insert(0, ROOT)

from verification.adversarial_search.reference_failure import (  # noqa: E402
    classify_failed_checks,
    failed_checks_from_summary,
)


def main():
    con = sqlite3.connect(os.path.join(ROOT, "adversarial_results/search_history.db"))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT v.proposal_id, r.operator, p.iteration, v.verdict_json, "
        "       p.proposal_json "
        "FROM verdicts v JOIN proposals p ON v.proposal_id = p.proposal_id "
        "JOIN runs r ON v.run_id = r.run_id "
        "ORDER BY r.operator, p.created_at").fetchall()
    n = 0
    for r in rows:
        vj = json.loads(r["verdict_json"])
        if vj.get("reference_passed", True):
            continue
        failed = failed_checks_from_summary(vj.get("failure_summary", "")) or []
        if classify_failed_checks(failed) != "invariant":
            continue
        if r["operator"] == "flash_attention":
            continue          # the three known (relabelled) records
        pj = json.loads(r["proposal_json"])
        n += 1
        print(f"{r['operator']:14s} it{r['iteration']} "
              f"{r['proposal_id'][:8]} failed={failed}")
        for k, t in pj["tensors"].items():
            print(f"    {k}: shape={t['shape']} fill={t['fill']} "
                  f"scale={t['scale']:g} shift={t['shift']:g} "
                  f"patches={len(t['patches'])}")
    print(f"\ntotal non-flash reference-suspect records: {n}")


if __name__ == "__main__":
    main()
