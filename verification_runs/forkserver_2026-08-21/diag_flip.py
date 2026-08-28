"""
Which check flips between spawn and forkserver, and by how much.

The A/B recorded only `passed_checker`, so it can say two reference verdicts
moved but not why. This re-runs just those proposals through both arms with the
full per-check detail, repeated N times per arm so a flip that is intermittent
can be told from one that is deterministic -- a distinction the A/B's single
pass per arm cannot make and which decides whether this is a forkserver defect
or a marginal input.
"""
import json
import os
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, "/content")
os.environ.setdefault("CHECKER_ROOT", "/content")

from verification.adversarial_search.schemas import InputProposal
from verification.adversarial_search.executor import execute_proposal_batch

ROOT = "/content"
OP = "causal_flash_attention"
REF = os.path.join(ROOT, "TritonBench/reference/causal_flash_attention.py")
MUT = os.path.join(ROOT, "TritonBench/cheating/causal_flash_attention/wrong_causal_mask.py")
DB = "/content/adversarial_results/cfa_rerun_2026-08-20/search_history.db"
TARGETS = ("2e429b94", "981ae6d1")
REPEATS = 3


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute("SELECT proposal_json FROM proposals WHERE operator=? "
                       "ORDER BY created_at LIMIT 40", (OP,)).fetchall()
    con.close()
    props = [InputProposal.from_dict(json.loads(r[0])) for r in rows]
    props = [p for p in props if p.proposal_id[:8] in TARGETS]
    print(f"{len(props)} target proposals\n", flush=True)

    ks = [("reference", REF), ("wrong_causal_mask", MUT)]
    out = defaultdict(list)

    for p in props:
        print("=" * 70)
        print(f"proposal {p.proposal_id[:8]}   tensors="
              f"{ {k: (v.shape, v.dtype, v.fill) for k, v in p.tensors.items()} }")
        print("=" * 70, flush=True)
        for arm, fs in (("spawn", False), ("forkserver", True)):
            for rep in range(REPEATS):
                res = execute_proposal_batch(
                    proposal=p, kernels=ks, reference_src_path=REF, operator=OP,
                    timeout_seconds=60, use_forkserver=fs)
                ref = [r for r in res if r.kernel_id == "reference"][0]
                failed = [c for c in ref.check_results if not c["passed"]]
                out[(p.proposal_id, arm)].append(
                    {"passed": ref.passed_checker,
                     "n_checks": len(ref.check_results),
                     "failed": [(c["check_name"], str(c.get("details"))[:160])
                                for c in failed]})
                print(f"  [{arm:10s} rep{rep}] passed={ref.passed_checker} "
                      f"n_checks={len(ref.check_results)} "
                      f"failed={[c['check_name'] for c in failed]}", flush=True)
                for c in failed:
                    print(f"       {c['check_name']}: {str(c.get('details'))[:200]}",
                          flush=True)
        print(flush=True)

    with open("/content/verification_runs/forkserver_2026-08-21/diag_flip.json", "w") as f:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in out.items()}, f, indent=2)
    print("DONE")

# The spawn arm re-imports this module in every child, so the driver MUST sit
# behind a __main__ guard -- without it each child re-runs the driver and
# multiprocessing refuses to start, which is how the first attempt produced
# `n_checks=0` rows that looked like a checker result and were not one.
if __name__ == "__main__":
    main()
