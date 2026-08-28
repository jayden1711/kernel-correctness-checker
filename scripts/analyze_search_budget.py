"""
scripts/analyze_search_budget.py

Two separate anchors, matching the two legitimate justifications:
  1. EMPIRICAL: for runs that actually found a hit, how many total
     proposals did the search consume? This is "budget matched to what
     guided search typically needed" -- the number for the headline
     same-budget Table 3 comparison.
  2. CEILING: n_workers x max_iter, the hard cap the coordinator is
     configured to allow before giving up entirely, independent of what
     any particular run happened to use. This is "worst-case
     compute-matched" -- the number for a second, explicitly separate
     "does random catch up given more tries" pass.

Run from project root:
    python scripts/analyze_search_budget.py --db adversarial_results/search_history.db
"""

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="adversarial_results/search_history.db")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # --- Anchor 1: EMPIRICAL, from runs that found a hit ---
    rows = conn.execute(
        "SELECT run_id, operator, n_workers, max_iter, result_json "
        "FROM runs WHERE result_json IS NOT NULL"
    ).fetchall()

    by_operator = defaultdict(list)
    ceilings = {}  # operator -> (n_workers, max_iter) as configured, for anchor 2
    n_no_result = 0
    n_no_hit = 0

    for row in rows:
        try:
            result = json.loads(row["result_json"])
        except (json.JSONDecodeError, TypeError):
            n_no_result += 1
            continue

        op = row["operator"]
        ceilings[op] = (row["n_workers"], row["max_iter"])

        if result.get("winning_proposal") is not None:
            total = result.get("total_proposals")
            if total is not None:
                by_operator[op].append(total)
        else:
            n_no_hit += 1

    print("=" * 70)
    print("ANCHOR 1 -- EMPIRICAL: proposals consumed by runs that found a hit")
    print("=" * 70)
    if not by_operator:
        print("No completed runs with a confirmed hit found in this DB yet.")
    else:
        all_totals = []
        for op in sorted(by_operator):
            totals = by_operator[op]
            all_totals.extend(totals)
            print(f"\n{op}  (n={len(totals)} hit-runs)")
            print(f"  mean:   {statistics.mean(totals):.1f}")
            print(f"  median: {statistics.median(totals):.1f}")
            print(f"  min:    {min(totals)}")
            print(f"  max:    {max(totals)}")
            print(f"  raw:    {sorted(totals)}")

        print(f"\n--- ALL OPERATORS COMBINED (n={len(all_totals)} hit-runs) ---")
        print(f"  mean:   {statistics.mean(all_totals):.1f}")
        print(f"  median: {statistics.median(all_totals):.1f}")
        print(f"  min:    {min(all_totals)}")
        print(f"  max:    {max(all_totals)}")

    print(f"\n(runs with no result_json / still running: {n_no_result})")
    print(f"(completed runs that never found a hit: {n_no_hit} -- these aren't")
    print(f" in the anchor-1 numbers above; they'd only inflate the mean")
    print(f" without telling you anything about typical successful budget)")

    print("\n" + "=" * 70)
    print("ANCHOR 2 -- CEILING: configured n_workers x max_iter per operator")
    print("=" * 70)
    if not ceilings:
        print("No runs found to read configuration from.")
    else:
        for op in sorted(ceilings):
            n_workers, max_iter = ceilings[op]
            print(f"  {op:20s} n_workers={n_workers} x max_iter={max_iter} "
                  f"= {n_workers * max_iter} max proposals")

    conn.close()


if __name__ == "__main__":
    main()