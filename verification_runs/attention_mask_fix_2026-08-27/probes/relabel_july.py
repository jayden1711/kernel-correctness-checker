"""Retroactively relabel the three 2026-07-23 flash_attention N=130 verdicts
under the reference-failure split, additively.

What changes in each verdict_json:
  + reference_failure_kind: "invariant"
  + relabel_2026_08_27: provenance note (original summary preserved verbatim
    in original_failure_summary; the recorded booleans are NOT altered --
    the recorded run really did see the reference fail, because the reference
    really was buggy. The relabel records what that failure MEANT and what
    the corrected-world verdict is, per the impact round's counterfactual and
    the post-fix GPU replay: valid non-hit, no gap.)

Idempotent: skips rows already carrying relabel_2026_08_27.
"""

import json
import os
import sqlite3

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "../../../adversarial_results/search_history.db")

NOTE = {
    "kind": "invariant",
    "date": "2026-08-27",
    "meaning": ("TRUE detection of the flash_attention padded-column masking "
                "bug (reference kernel, N % 32 != 0), not an invalid input. "
                "Bug fixed 2026-08-27; post-fix GPU replay classifies this "
                "proposal as VALID NON-HIT (reference passes "
                "attention_weights_sum_to_one; no mutant passes naive "
                "allclose, so no gap)."),
    "evidence": ("verification_runs/attention_mask_bug_impact_2026-08-27/ and "
                 "verification_runs/attention_mask_fix_2026-08-27/ (stage C)"),
}


def main():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT v.proposal_id, v.verdict_json FROM verdicts v "
        "JOIN proposals p ON v.proposal_id = p.proposal_id "
        "JOIN runs r ON v.run_id = r.run_id "
        "WHERE r.operator = 'flash_attention'").fetchall()
    changed = 0
    for r in rows:
        vj = json.loads(r["verdict_json"])
        if vj.get("reference_passed", True):
            continue
        if "attention_weights_sum_to_one" not in vj.get("failure_summary", ""):
            continue
        if "relabel_2026_08_27" in vj:
            print(f"{r['proposal_id'][:8]}: already relabelled, skipping")
            continue
        print(f"{r['proposal_id'][:8]}: BEFORE: {vj['failure_summary'][:110]}")
        vj["original_failure_summary"] = vj["failure_summary"]
        vj["reference_failure_kind"] = "invariant"
        vj["relabel_2026_08_27"] = NOTE
        vj["failure_summary"] = (
            "REFERENCE-SUSPECT (relabelled 2026-08-27: reference kernel "
            "masking bug, fixed; corrected-world verdict = valid non-hit) | "
            + vj["failure_summary"])
        con.execute("UPDATE verdicts SET verdict_json = ? WHERE proposal_id = ?",
                    (json.dumps(vj), r["proposal_id"]))
        changed += 1
        print(f"{r['proposal_id'][:8]}: AFTER : {vj['failure_summary'][:110]}")
    con.commit()
    con.close()
    print(f"\nrelabelled {changed} verdict(s)")
    assert changed in (0, 3), f"expected exactly 3 (or 0 on re-run), got {changed}"


if __name__ == "__main__":
    main()
