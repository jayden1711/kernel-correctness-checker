"""Replay the two runs that banked per-kernel execution data (incl.
error_type) under the DOCSTRING-INTENDED beam scoring, and measure whether
any seed decision the coordinator actually made would have changed.

Intended beam scoring, per beam.py's docstring:
  +10 reference passed | -5 reference failed
  +8  per mutant caught with gap confirmed
  +3  per mutant caught without gap        (dead code in the shipped scorer)
  +2  reference passed and nothing caught  (shipped grants it whenever
                                            hit_mutants is empty, i.e. also on
                                            caught-no-gap and errored verdicts)
  -2  per mutant that errored              (never implemented; an errored
                                            mutant has passed_checker=False,
                                            passed_naive=False and lands in
                                            caught_no_gap scoring 0 + the
                                            spurious +2)

Errored mutants are identified from executions.error_type. A mutant row with
an error is scored -2 and excluded from "caught" credit; caught-no-gap =
passed_checker=0, passed_naive=0, no error; caught-with-gap = passed_checker=0,
passed_naive=1.
"""

import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "../../..")

DBS = [
    "adversarial_results/cfa_rerun_2026-08-20/search_history.db",
    "adversarial_results/cfa_rerun_postfix_2026-08-21/search_history.db",
]


def main():
    for db in DBS:
        con = sqlite3.connect(os.path.join(ROOT, db))
        con.row_factory = sqlite3.Row
        run = con.execute("SELECT * FROM runs").fetchone()
        rows = con.execute(
            "SELECT proposal_id, beam_score, created_at FROM verdicts "
            "WHERE run_id=? ORDER BY created_at", (run["run_id"],)).fetchall()
        print(f"== {db}  run {run['run_id'][:8]} {run['operator']} "
              f"strat={run['strategy']} B={run['n_workers']} n={len(rows)}")

        n_err_verdicts = 0
        n_cng_verdicts = 0
        diverge_steps = 0
        top_shipped = top_intended = None
        details = []
        for r in rows:
            ex = con.execute(
                "SELECT kernel_id, passed_checker, passed_naive, error_type "
                "FROM executions WHERE proposal_id=?", (r["proposal_id"],)
            ).fetchall()
            ref = [e for e in ex if e["kernel_id"] == "reference"]
            muts = [e for e in ex if e["kernel_id"] != "reference"]
            ref_passed = bool(ref and ref[0]["passed_checker"])
            errored = [m for m in muts if m["error_type"]]
            gap = [m for m in muts
                   if not m["passed_checker"] and m["passed_naive"]]
            cng = [m for m in muts if not m["passed_checker"]
                   and not m["passed_naive"] and not m["error_type"]]
            if errored:
                n_err_verdicts += 1
            if cng:
                n_cng_verdicts += 1

            intended = (10.0 if ref_passed else -5.0) + 8.0 * len(gap) \
                + 3.0 * len(cng) - 2.0 * len(errored)
            if ref_passed and not gap and not cng and not errored:
                intended += 2.0

            if errored:
                details.append(
                    (r["proposal_id"][:8], len(errored), r["beam_score"], intended))

            # running argmax under each scoring (ties: first wins, matching
            # stable sort + list order in the coordinator)
            if top_shipped is None or r["beam_score"] > top_shipped[1]:
                top_shipped = (r["proposal_id"], r["beam_score"])
            if top_intended is None or intended > top_intended[1]:
                top_intended = (r["proposal_id"], intended)
            if top_shipped[0] != top_intended[0]:
                diverge_steps += 1

        print(f"   verdicts with errored mutants: {n_err_verdicts}   "
              f"with caught-no-gap mutants: {n_cng_verdicts}")
        for pid, ne, s, i in details:
            print(f"   errored-proposal {pid}: {ne} errored, shipped score {s} "
                  f"-> intended {i}")
        print(f"   steps where the consumed seed (running argmax) differs "
              f"shipped vs intended: {diverge_steps} / {len(rows)}")
        con.close()


if __name__ == "__main__":
    main()
