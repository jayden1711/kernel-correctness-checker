"""
Replay real recorded proposals through BOTH executor arms and time them.

No LLM, no API key, no network: the proposals come from the recorded
`search_history.db`, so the input sequence is fixed and identical across arms.
That is the point -- an LLM-driven run generates a DIFFERENT proposal set each
time, which is exactly what makes total-wall-time comparisons between two search
runs meaningless. Here the work is held constant and only the executor changes.

Order control: arm A runs, then arm B, then arm A AGAIN. If A1 and A2 agree,
warm page/JIT caches are not what produced the difference. Without that second A
pass the whole comparison would be confounded by run order -- which is precisely
the defect that made the checker's per-layer latency table wrong (#7a step 2).

Concurrency matches the real search (4 worker threads), because four processes
contending to initialise CUDA on one T4 is part of the cost being measured.
"""
import json
import os
import sqlite3
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/content")
os.environ.setdefault("CHECKER_ROOT", "/content")

from verification.adversarial_search.schemas import InputProposal
from verification.adversarial_search.executor import (
    execute_proposal, execute_proposal_batch)

ROOT = "/content"
N_WORKERS = 4

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
    ks = kernels_for(operator)
    ref = ks[0][1]
    out = []

    def one(p):
        t0 = time.perf_counter()
        if arm == "batched":
            res = execute_proposal_batch(
                proposal=p, kernels=ks, reference_src_path=ref,
                operator=operator, timeout_seconds=timeout)
        else:
            res = [execute_proposal(
                proposal=p, kernel_id=kid, candidate_src_path=path,
                reference_src_path=ref, operator=operator,
                timeout_seconds=timeout) for kid, path in ks]
        dt = time.perf_counter() - t0
        return {
            "proposal_id": p.proposal_id,
            "proposal_s": dt,
            "spawns": 1 if arm == "batched" and all(
                r.exec_mode == "batched" for r in res) else len(
                [r for r in res if r.exec_mode != "batched"]) + (
                1 if any(r.exec_mode == "batched" for r in res) else 0),
            "kernels": [{
                "kernel_id": r.kernel_id,
                "passed_checker": r.passed_checker,
                "passed_naive": r.passed_naive,
                "error": r.error.error_type if r.error else None,
                "exec_mode": r.exec_mode,
                "kernel_wall_time_ms": r.kernel_wall_time_ms,
                "total_wall_time_ms": r.total_wall_time_ms,
                "batch_spawn_ms": r.batch_spawn_ms,
                "startup_phases": r.startup_phases,
            } for r in res],
        }

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for rec in pool.map(one, proposals):
            out.append(rec)
            print(f"    [{arm}] {rec['proposal_id'][:8]} "
                  f"{rec['proposal_s']:6.2f}s", flush=True)
    return out


def med(xs):
    return st.median(xs) if xs else float("nan")


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
        print(f"\n{'='*66}\n{operator}: {len(props)} proposals x {n_k} kernels\n{'='*66}",
              flush=True)

        passes = {}
        for label, arm in (("A1", "single"), ("B", "batched"), ("A2", "single")):
            print(f"\n-- pass {label} ({arm}) --", flush=True)
            t0 = time.time()
            passes[label] = run_arm(arm, props, operator)
            print(f"   pass {label} total {time.time()-t0:.1f}s", flush=True)
            report[operator] = passes
            with open("/content/replay_ab.json", "w") as f:
                json.dump(report, f)          # written after EVERY pass: a VM
                                              # reclamation costs one pass, not all
        for label in ("A1", "B", "A2"):
            recs = passes[label]
            print(f"  {label:3s} per-proposal median {med([r['proposal_s'] for r in recs]):7.2f}s"
                  f"   spawns/proposal {med([r['spawns'] for r in recs]):.2f}", flush=True)

    with open("/content/replay_ab.json", "w") as f:
        json.dump(report, f)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
