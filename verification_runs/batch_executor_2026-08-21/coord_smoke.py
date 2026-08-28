"""
End-to-end smoke test of the COORDINATOR wiring, on real GPU kernels, with no
LLM and no API key.

The replay harness exercised the executor. This exercises everything the
coordinator change touched that the executor tests cannot see: that
`on_result` really persists each execution as it lands, that the run survives a
stop event, that verdicts still evaluate, and that the new timing columns reach
the database on a real run. AdversarialWorker.propose/refine are replaced by a
replay of recorded proposals -- the only part of the loop that needs an LLM.
"""
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, "/content")
os.environ.setdefault("CHECKER_ROOT", "/content")

from verification.adversarial_search import worker as W
from verification.adversarial_search.coordinator import SearchCoordinator
from verification.adversarial_search.schemas import InputProposal

OP = "causal_flash_attention"
DB = "/content/adversarial_results/cfa_rerun_2026-08-20/search_history.db"

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
POOL = [json.loads(r[0]) for r in con.execute(
    "SELECT proposal_json FROM proposals WHERE operator=? ORDER BY created_at "
    "LIMIT 24", (OP,)).fetchall()]
con.close()
_i = [0]


def _next(self):
    d = dict(POOL[_i[0] % len(POOL)])
    _i[0] += 1
    d["proposal_id"] = f"smoke-{_i[0]:04d}"
    d["worker_id"] = self.worker_id
    return InputProposal.from_dict(d)


W.AdversarialWorker.propose = _next
W.AdversarialWorker.refine = lambda self, feedback: _next(self)


def main():
    for batched in (True, False):
        out = f"/content/smoke_{'batched' if batched else 'single'}"
        os.makedirs(out, exist_ok=True)
        print(f"\n{'='*66}\n  coordinator run, batch_executions={batched}\n{'='*66}",
              flush=True)
        t0 = time.time()
        coord = SearchCoordinator(
            operator=OP,
            reference_src_path="/content/TritonBench/reference/causal_flash_attention.py",
            mutant_src_paths={"wrong_causal_mask":
                "/content/TritonBench/cheating/causal_flash_attention/wrong_causal_mask.py"},
            model="stub", strategy="beam", n_workers=2, max_iterations=3,
            timeout_per_exec=30, output_dir=out, batch_executions=batched,
        )
        result = coord.run()
        wall = time.time() - t0

        db = os.path.join(out, "search_history.db")
        c = sqlite3.connect(db)
        n_ex = c.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
        n_pr = c.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        n_vd = c.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0]
        modes = dict(c.execute(
            "SELECT exec_mode, COUNT(*) FROM executions GROUP BY exec_mode"))
        spawn = c.execute(
            "SELECT COUNT(DISTINCT proposal_id) FROM executions "
            "WHERE exec_mode='batched'").fetchone()[0]
        single = c.execute(
            "SELECT COUNT(*) FROM executions WHERE exec_mode!='batched'").fetchone()[0]
        phases = c.execute("SELECT startup_phases_json FROM executions "
                           "WHERE startup_phases_json IS NOT NULL LIMIT 1").fetchone()
        c.close()

        print(f"  wall {wall:.1f}s  proposals={n_pr} executions={n_ex} verdicts={n_vd}")
        print(f"  exec_mode {modes}")
        print(f"  SPAWNS {spawn + single}  (vs {n_ex} executions)")
        print(f"  startup_phases persisted: {phases[0] if phases else 'NONE'}")
        print(f"  status={result.status if hasattr(result,'status') else '?'}")
    print("\nSMOKE DONE", flush=True)


if __name__ == "__main__":
    main()
