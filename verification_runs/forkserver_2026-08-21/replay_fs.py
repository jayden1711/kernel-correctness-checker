"""
forkserver A/B — replay recorded proposals through spawn and forkserver.

Adapted from `verification_runs/batch_executor_2026-08-21/replay_ab.py`, and for
the same reason: an LLM-driven search generates a DIFFERENT proposal set every
run, so two searches' wall times measure different amounts of work and cannot be
compared. Here the proposals come from a recorded `search_history.db`, so the
work is held constant and only the executor varies.

PASS ORDER, and why it is what it is:

    A1, A2   unbatched + unseeded, on a SUBSET.
             NOT part of the measurement. These exist to prove the verdict
             comparator below can SEE a disagreement at all. "forkserver changed
             0 verdicts" is unfalsifiable on its own -- a comparator that never
             reports a difference prints exactly the same number (§5 instance
             11). The unseeded path is known to disagree with itself at ~6%, so
             it is the natural positive control, and running it through THIS
             comparator on REAL data is the only thing that establishes the
             instrument works. Synthetic validation would prove only that the
             synthetic cases are detectable (§5 instance 12).

    B1 -> C -> B2   the measurement, order-controlled.
             B1 and B2 are the same arm. If they agree, warm page/JIT caches are
             not what produced C's difference. Without a second B the whole
             result would be confounded by run order -- the defect that made the
             checker's per-layer latency table wrong (#7a step 2).

WHAT THE VERDICT BAR IS. Batching carried a declared semantic change and so was
judged against a ~6% floor. forkserver carries NONE: both B and C batch, both
seed from the proposal id, and only the start method differs. So the bar is
zero -- but "zero" only means something once B1-vs-B2 has established what the
seeded path's own run-to-run floor actually is, which nothing has measured yet.
That comparison is computed first and reported as the resolution limit.

Run on the VM:
    PYTHONPATH=/content python3 /content/verification_runs/forkserver_2026-08-21/replay_fs.py
"""
import json
import os
import sqlite3
import statistics as st
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/content")
os.environ.setdefault("CHECKER_ROOT", "/content")

from verification.adversarial_search.schemas import InputProposal
from verification.adversarial_search.executor import (
    execute_proposal, execute_proposal_batch)

ROOT = "/content"
N_WORKERS = 4
OUT = "/content/verification_runs/forkserver_2026-08-21/replay_fs.json"

# The unbatched control is a subset: it costs ~28s per proposal per arm and its
# only job is to show the comparator is not blind. The measurement itself uses
# the full set.
CONTROL_N = 20

REFS = {
    "causal_flash_attention": "TritonBench/reference/causal_flash_attention.py",
    "flash_attention": "TritonBench/reference/flash_attention.py",
}
MUTANTS = {
    "causal_flash_attention": {
        "wrong_causal_mask": "TritonBench/cheating/causal_flash_attention/wrong_causal_mask.py",
    },
    "flash_attention": {
        "approx_denom":   "TritonBench/cheating/flash_attention/approx_denom.py",
        "drop_last_tile": "TritonBench/cheating/flash_attention/drop_last_tile.py",
        "skip_rescaling": "TritonBench/cheating/flash_attention/skip_rescaling.py",
        "wrong_mask":     "TritonBench/cheating/flash_attention/wrong_mask.py",
    },
}


def load_proposals(db, operator, limit):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT proposal_json FROM proposals WHERE operator=? "
        "ORDER BY created_at LIMIT ?", (operator, limit)).fetchall()
    con.close()
    return [InputProposal.from_dict(json.loads(r[0])) for r in rows]


def kernels_for(operator):
    ks = [("reference", os.path.join(ROOT, REFS[operator]))]
    ks += [(k, os.path.join(ROOT, v)) for k, v in MUTANTS[operator].items()]
    return ks


def run_arm(arm, proposals, operator, timeout=30):
    """arm: 'single' | 'batched_spawn' | 'batched_forkserver'."""
    ks = kernels_for(operator)
    ref = ks[0][1]

    def one(p):
        t0 = time.perf_counter()
        if arm == "single":
            res = [execute_proposal(
                proposal=p, kernel_id=kid, candidate_src_path=path,
                reference_src_path=ref, operator=operator,
                timeout_seconds=timeout) for kid, path in ks]
        else:
            res = execute_proposal_batch(
                proposal=p, kernels=ks, reference_src_path=ref,
                operator=operator, timeout_seconds=timeout,
                use_forkserver=(arm == "batched_forkserver"))
        dt = time.perf_counter() - t0
        return {
            "proposal_id": p.proposal_id,
            "proposal_s": dt,
            "kernels": [{
                "kernel_id": r.kernel_id,
                "passed_checker": r.passed_checker,
                "passed_naive": r.passed_naive,
                "error": r.error.error_type if r.error else None,
                "exec_mode": r.exec_mode,
                "start_method": r.start_method,
                "kernel_wall_time_ms": r.kernel_wall_time_ms,
                "total_wall_time_ms": r.total_wall_time_ms,
                "batch_spawn_ms": r.batch_spawn_ms,
                "startup_phases": r.startup_phases,
            } for r in res],
        }

    out = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for rec in pool.map(one, proposals):
            out.append(rec)
            print(f"    [{arm}] {rec['proposal_id'][:8]} {rec['proposal_s']:6.2f}s",
                  flush=True)
    return out


def verdict_map(recs):
    """(proposal_id, kernel_id) -> passed_checker, the quantity a verdict rests on."""
    return {(r["proposal_id"], k["kernel_id"]): k["passed_checker"]
            for r in recs for k in r["kernels"]}


def compare(a, b):
    """Disagreeing (proposal, kernel) pairs between two passes.

    Compared over the INTERSECTION of keys: a pass cut short by a reclamation
    would otherwise report every missing pair as a disagreement, which is a
    different fact entirely.
    """
    ma, mb = verdict_map(a), verdict_map(b)
    shared = set(ma) & set(mb)
    diffs = sorted(k for k in shared if ma[k] != mb[k])
    return {"n_compared": len(shared), "n_diff": len(diffs),
            "pairs": [{"proposal_id": p, "kernel_id": k,
                       "left": ma[(p, k)], "right": mb[(p, k)]}
                      for p, k in diffs]}


def med(xs):
    return st.median(xs) if xs else float("nan")


def phase_medians(recs):
    phases = defaultdict(list)
    for r in recs:
        for k in r["kernels"]:
            for name, v in (k.get("startup_phases") or {}).items():
                phases[name].append(v)
    return {n: med(v) for n, v in sorted(phases.items())}


def main():
    jobs = [
        ("causal_flash_attention",
         "/content/adversarial_results/cfa_rerun_2026-08-20/search_history.db", 40),
        ("flash_attention",
         "/content/adversarial_results/search_history.db", 12),
    ]
    report = {}

    for operator, db, limit in jobs:
        props = load_proposals(db, operator, limit)
        n_k = len(kernels_for(operator))
        print(f"\n{'='*70}\n{operator}: {len(props)} proposals x {n_k} kernels\n{'='*70}",
              flush=True)

        passes = {}
        plan = [
            ("A1", "single", props[:CONTROL_N]),
            ("A2", "single", props[:CONTROL_N]),
            ("B1", "batched_spawn", props),
            ("C",  "batched_forkserver", props),
            ("B2", "batched_spawn", props),
        ]
        for label, arm, subset in plan:
            print(f"\n-- pass {label} ({arm}, n={len(subset)}) --", flush=True)
            t0 = time.time()
            passes[label] = run_arm(arm, subset, operator)
            print(f"   pass {label} total {time.time()-t0:.1f}s", flush=True)
            report[operator] = {"passes": passes}
            with open(OUT, "w") as f:
                json.dump(report, f)      # after EVERY pass: a reclamation costs
                                          # one pass, not the whole run

        summary = {}
        for label in ("A1", "A2", "B1", "C", "B2"):
            recs = passes[label]
            summary[label] = {
                "n": len(recs),
                "per_proposal_median_s": med([r["proposal_s"] for r in recs]),
                "phases": phase_medians(recs),
                "start_methods": sorted({k["start_method"] for r in recs
                                         for k in r["kernels"]}),
                "exec_modes": sorted({k["exec_mode"] for r in recs
                                      for k in r["kernels"]}),
            }
        # Instrument control FIRST, then the seeded floor, then the result.
        summary["cmp_A1_A2_instrument_control"] = compare(passes["A1"], passes["A2"])
        summary["cmp_B1_B2_seeded_floor"] = compare(passes["B1"], passes["B2"])
        summary["cmp_C_B2_the_result"] = compare(passes["C"], passes["B2"])
        summary["cmp_C_B1"] = compare(passes["C"], passes["B1"])
        report[operator]["summary"] = summary

        print(f"\n  --- {operator} ---")
        for label in ("A1", "A2", "B1", "C", "B2"):
            s = summary[label]
            print(f"  {label:3s} n={s['n']:3d}  median {s['per_proposal_median_s']:7.2f}s"
                  f"  start_method={s['start_methods']}  exec_mode={s['exec_modes']}")
        b1, b2 = summary["B1"]["per_proposal_median_s"], summary["B2"]["per_proposal_median_s"]
        c = summary["C"]["per_proposal_median_s"]
        print(f"  order drift B1 vs B2: {abs(b1-b2)/max(b1,b2)*100:.1f}%")
        print(f"  effect C vs B2:       {c/b2:.3f}x  ({(c/b2-1)*100:+.1f}%)")
        print(f"  INSTRUMENT CONTROL  A1 vs A2 (unseeded): "
              f"{summary['cmp_A1_A2_instrument_control']['n_diff']} of "
              f"{summary['cmp_A1_A2_instrument_control']['n_compared']} disagree")
        print(f"  SEEDED FLOOR        B1 vs B2:            "
              f"{summary['cmp_B1_B2_seeded_floor']['n_diff']} of "
              f"{summary['cmp_B1_B2_seeded_floor']['n_compared']} disagree")
        print(f"  RESULT              C  vs B2:            "
              f"{summary['cmp_C_B2_the_result']['n_diff']} of "
              f"{summary['cmp_C_B2_the_result']['n_compared']} disagree")

        with open(OUT, "w") as f:
            json.dump(report, f)

    with open(OUT, "w") as f:
        json.dump(report, f)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
