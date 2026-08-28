"""
Does forkserver change the delegation-detector race rate? Powered version.

THE PILOT COULD NOT ANSWER THIS AND SAID SO: 1 of 3 spawn passes flipped against
3 of 3 forkserver passes, which at n=3 per arm resolves nothing. This runs the
sample size that does.

TWO ENDPOINTS, because the binary one is expensive and the continuous one is not:

  BINARY      flip = the detector fires = `delegation_ratio > 10`.
              This is the thing that actually costs a verdict. It is also a rare
              event (~1.7% pooled in the pilot), so distinguishing 0.83% from
              2.50% at 80% power needs ~921 trials per arm -- computed BEFORE
              the run, not after, and the reason the target is 26 passes/arm.

  CONTINUOUS  `delegation_ratio` itself, now recorded on every reference
              execution rather than only when it trips (see runtime_guards.py).
              Comparing distributions rather than counting tail crossings gives
              far more power for the same GPU time: ~700/arm resolves a 0.15-SD
              shift in log(ratio). If the two arms' distributions are
              indistinguishable through the bulk AND the upper quantiles, the
              tail rates cannot meaningfully differ either -- and that argument
              is what makes a null result here mean something rather than being
              another underpowered zero.

DESIGN NOTES

  * ARMS ARE INTERLEAVED (spawn, forkserver, spawn, ...), not blocked. The pilot
    ran each arm's passes contiguously, so a thermal or noisy-neighbour drift
    would land entirely on one arm. Interleaving makes drift common-mode, and
    the spawn-vs-spawn comparison across the run is what detects it.
  * ONE JSONL LINE PER TRIAL, flushed immediately. A Colab VM can be reclaimed
    mid-run; this costs the in-flight pass, not the experiment. Re-running
    appends, so the budget can be extended after an interim look without
    discarding anything.
  * Every reference execution is recorded, including the ~5 of 40 whose input is
    out of domain and which therefore never reach the detector. They are marked
    `reached=False` and excluded from rates -- counting them in the denominator
    would dilute both arms and understate the effect.

Run on the VM:
    PYTHONPATH=/content python3 .../race_rate.py [n_pairs]
"""
import json
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/content")
os.environ.setdefault("CHECKER_ROOT", "/content")

from verification.adversarial_search.schemas import InputProposal
from verification.adversarial_search.executor import execute_proposal_batch

ROOT = "/content"
OP = "causal_flash_attention"
REF = os.path.join(ROOT, "TritonBench/reference/causal_flash_attention.py")
MUT = os.path.join(ROOT, "TritonBench/cheating/causal_flash_attention/wrong_causal_mask.py")
DB = "/content/adversarial_results/cfa_rerun_2026-08-20/search_history.db"
N_WORKERS = 4
OUT = "/content/verification_runs/forkserver_2026-08-21/race_rate.jsonl"

_RATIO = re.compile(r"delegation_ratio=([0-9.eE+\-]+|inf)")


def parse_ratio(details):
    m = _RATIO.search(details or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def main():
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 13

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute("SELECT proposal_json FROM proposals WHERE operator=? "
                       "ORDER BY created_at LIMIT 40", (OP,)).fetchall()
    con.close()
    props = [InputProposal.from_dict(json.loads(r[0])) for r in rows]
    ks = [("reference", REF), ("wrong_causal_mask", MUT)]

    # Resume: continue numbering from whatever is already on disk so a relaunch
    # extends the experiment rather than colliding with it.
    start_pass = 0
    if os.path.exists(OUT):
        with open(OUT) as f:
            seen = [json.loads(ln) for ln in f if ln.strip()]
        if seen:
            start_pass = max(r["pass_idx"] for r in seen) + 1
            print(f"resuming: {len(seen)} trials already on disk, "
                  f"starting at pass {start_pass}", flush=True)

    def one(p, fs):
        res = execute_proposal_batch(
            proposal=p, kernels=ks, reference_src_path=REF, operator=OP,
            timeout_seconds=60, use_forkserver=fs)
        ref = [r for r in res if r.kernel_id == "reference"][0]
        ke = [c for c in ref.check_results if c["check_name"] == "kernel_executed"]
        ratio = parse_ratio(ke[0]["details"]) if ke else None
        return {
            "proposal_id": p.proposal_id,
            "ref_passed": ref.passed_checker,
            "start_method": ref.start_method,
            "reached": ratio is not None,
            "ratio": ratio,
            "ke_passed": (ke[0]["passed"] if ke else None),
            "failed_checks": [c["check_name"] for c in ref.check_results
                              if not c["passed"]],
            "n_checks": len(ref.check_results),
            "mut_passed": [r.passed_checker for r in res
                           if r.kernel_id != "reference"],
        }

    fh = open(OUT, "a")
    for pair in range(n_pairs):
        # Interleaved, and the ORDER WITHIN EACH PAIR ALTERNATES too, so that
        # "spawn always runs first" cannot itself become the explanation.
        arms = [("spawn", False), ("forkserver", True)]
        if pair % 2 == 1:
            arms = arms[::-1]
        for arm, fs in arms:
            pidx = start_pass + pair * 2 + (0 if arm == "spawn" else 1)
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                recs = list(pool.map(lambda p: one(p, fs), props))
            wall = time.time() - t0
            for r in recs:
                r.update({"arm": arm, "pass_idx": pidx, "pair": pair,
                          "pass_wall_s": wall})
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

            reached = [r for r in recs if r["reached"]]
            flips = [r for r in reached if not r["ke_passed"]]
            ratios = sorted(r["ratio"] for r in reached)
            med = ratios[len(ratios) // 2] if ratios else float("nan")
            p90 = ratios[int(0.9 * (len(ratios) - 1))] if ratios else float("nan")
            print(f"[pair {pair:2d} {arm:11s}] {wall:6.1f}s  "
                  f"reached {len(reached):2d}/{len(recs)}  "
                  f"flips {len(flips)}  "
                  f"ratio med {med:6.2f}  p90 {p90:6.2f}", flush=True)
    fh.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
