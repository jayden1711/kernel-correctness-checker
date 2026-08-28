"""Replay every recorded adversarial-search run through both selection
strategies and measure whether the diversity penalty ever changed anything
that the coordinator actually consumes.

Facts this probe establishes on the REAL recorded pools (not synthetic):
  1. score replication: recomputing the shipped beam score from each stored
     verdict reproduces the stored beam_score (validates the replay).
  2. slot-0 identity: beam[0] -- the ONLY beam slot the coordinator reads
     (_pick_beam_seed ignores worker_id and returns _beam[0]) -- is identical
     under beam vs diverse at the shipped lambda=3.0, at lambda=100, and at
     lambda=infinity (hard exclusion), at the run's real beam width.
  3. how often the diverse beam differed AT ALL (set or order) from plain
     top-B -- the ceiling on what any consumer of slots 1..B-1 could have seen.

Chronology: verdicts are replayed in created_at order, reproducing the
coordinator's incremental pool exactly (append -> select -> truncate).
"""

import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "../../..")
sys.path.insert(0, ROOT)

from verification.adversarial_search.strategy.beam import BeamSearchStrategy
from verification.adversarial_search.strategy.diverse import DiverseBeamStrategy

DBS = [
    "adversarial_results/search_history.db",
    "adversarial_results/cfa_rerun_2026-08-20/search_history.db",
    "adversarial_results/cfa_rerun_postfix_2026-08-21/search_history.db",
]


class P:
    def __init__(self, pid, mode):
        self.proposal_id = pid
        self.predicted_failure_mode = mode


class V:
    def __init__(self, score):
        self.beam_score = score


def shipped_score(vj):
    s = 10.0 if vj["reference_passed"] else -5.0
    for _ in vj["hit_mutants"]:
        s += 8.0 if vj["gap_confirmed"] else 3.0
    if vj["reference_passed"] and not vj["hit_mutants"]:
        s += 2.0
    return s


def main():
    beam = BeamSearchStrategy()
    div3 = DiverseBeamStrategy(3.0)
    div100 = DiverseBeamStrategy(100.0)
    div_inf = DiverseBeamStrategy(1e9)

    tot = dict(verdicts=0, score_mismatch=0, slot0_diff3=0, slot0_diff100=0,
               slot0_diffinf=0, beamset_diff3=0, beamorder_diff3=0,
               beamset_diff100=0)

    for db in DBS:
        path = os.path.join(ROOT, db)
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        runs = con.execute(
            "SELECT run_id, operator, strategy, n_workers FROM runs").fetchall()
        for run in runs:
            rows = con.execute(
                "SELECT v.proposal_id, v.beam_score, v.verdict_json, v.created_at,"
                "       p.proposal_json "
                "FROM verdicts v JOIN proposals p ON v.proposal_id = p.proposal_id "
                "WHERE v.run_id = ? ORDER BY v.created_at", (run["run_id"],)
            ).fetchall()
            B = run["n_workers"]
            pool_beam, pool_d3, pool_d100, pool_dinf = [], [], [], []
            n_mismatch = s0_3 = s0_100 = s0_inf = set3 = ord3 = set100 = 0
            for r in rows:
                vj = json.loads(r["verdict_json"])
                pj = json.loads(r["proposal_json"])
                mode = pj.get("predicted_failure_mode", "?")
                rescored = shipped_score(vj)
                if abs(rescored - r["beam_score"]) > 1e-9:
                    n_mismatch += 1
                item = (P(r["proposal_id"], mode), V(r["beam_score"]))
                pool_beam = beam.select(pool_beam + [item], B)
                pool_d3 = div3.select(pool_d3 + [item], B)
                pool_d100 = div100.select(pool_d100 + [item], B)
                pool_dinf = div_inf.select(pool_dinf + [item], B)
                if pool_beam[0][0].proposal_id != pool_d3[0][0].proposal_id:
                    s0_3 += 1
                if pool_beam[0][0].proposal_id != pool_d100[0][0].proposal_id:
                    s0_100 += 1
                if pool_beam[0][0].proposal_id != pool_dinf[0][0].proposal_id:
                    s0_inf += 1
                ids_b = [x[0].proposal_id for x in pool_beam]
                ids_3 = [x[0].proposal_id for x in pool_d3]
                ids_100 = [x[0].proposal_id for x in pool_d100]
                if set(ids_b) != set(ids_3):
                    set3 += 1
                elif ids_b != ids_3:
                    ord3 += 1
                if set(ids_b) != set(ids_100):
                    set100 += 1
            print(f"{run['run_id'][:8]} {run['operator']:24s} "
                  f"strat={run['strategy']:8s} B={B} n={len(rows):3d} "
                  f"score_mismatch={n_mismatch} slot0(l3/l100/linf)="
                  f"{s0_3}/{s0_100}/{s0_inf} beamset_l3={set3} "
                  f"beamorder_l3={ord3} beamset_l100={set100}")
            tot["verdicts"] += len(rows)
            tot["score_mismatch"] += n_mismatch
            tot["slot0_diff3"] += s0_3
            tot["slot0_diff100"] += s0_100
            tot["slot0_diffinf"] += s0_inf
            tot["beamset_diff3"] += set3
            tot["beamorder_diff3"] += ord3
            tot["beamset_diff100"] += set100
        con.close()

    print("\nTOTALS:", tot)


if __name__ == "__main__":
    main()
