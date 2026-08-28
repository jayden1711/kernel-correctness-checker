"""
Is the 2-of-80 flip a forkserver effect, or a contention-sensitive check?

The A/B ran ONE pass per arm. That is enough to say two verdicts moved; it is not
enough to say which arm moved them, because it cannot separate "forkserver did
this" from "this check flips under load and one pass happened to catch it".
`diag_flip.py` already showed both proposals pass 6/6 in BOTH arms when run
sequentially, so whatever this is, it needs concurrency to appear.

So: repeat both arms under the SAME 4-thread contention, and capture the full
per-check detail this time -- the A/B recorded only `passed_checker`, which is
why the failing check was unknown.

Reports the flip rate per arm. If spawn also flips, the "seeded floor = 0 of 80"
from a single B1/B2 pair was underpowered rather than exact, and forkserver's 2
is inside a band that comparison could not resolve (§5 instance 12).
"""
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
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
OUT = "/content/verification_runs/forkserver_2026-08-21/diag_contention.json"


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute("SELECT proposal_json FROM proposals WHERE operator=? "
                       "ORDER BY created_at LIMIT 40", (OP,)).fetchall()
    con.close()
    props = [InputProposal.from_dict(json.loads(r[0])) for r in rows]
    ks = [("reference", REF), ("wrong_causal_mask", MUT)]

    def one(p, fs):
        res = execute_proposal_batch(
            proposal=p, kernels=ks, reference_src_path=REF, operator=OP,
            timeout_seconds=60, use_forkserver=fs)
        ref = [r for r in res if r.kernel_id == "reference"][0]
        return {"proposal_id": p.proposal_id,
                "passed": ref.passed_checker,
                "failed": [(c["check_name"], str(c.get("details"))[:200])
                           for c in ref.check_results if not c["passed"]]}

    report = {}
    plan = [("spawn", False), ("forkserver", True)] * 3
    for i, (arm, fs) in enumerate(plan):
        label = f"{arm}_{i}"
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            recs = list(pool.map(lambda p: one(p, fs), props))
        bad = [r for r in recs if not r["passed"]]
        report[label] = {"arm": arm, "wall_s": time.time() - t0,
                         "n": len(recs), "n_ref_failed": len(bad), "failures": bad}
        print(f"[{label}] {time.time()-t0:6.1f}s  reference failed "
              f"{len(bad)}/{len(recs)}", flush=True)
        for b in bad:
            print(f"    {b['proposal_id'][:8]} -> "
                  f"{[c[0] for c in b['failed']]}", flush=True)
            for name, det in b["failed"]:
                print(f"       {name}: {det}", flush=True)
        with open(OUT, "w") as f:
            json.dump(report, f, indent=2)

    print("\n=== flip rate by arm ===", flush=True)
    by_arm = defaultdict(list)
    for label, r in report.items():
        by_arm[r["arm"]].append(r["n_ref_failed"])
    for arm, counts in by_arm.items():
        print(f"  {arm:11s} reference failures per pass: {counts} "
              f"(total {sum(counts)} of {sum(len(props) for _ in counts)})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
