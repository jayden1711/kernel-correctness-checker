"""
Re-verify the three forkserver gates AT THE NEW DEFAULT (flipped 2026-08-28).

The A/B in ../../forkserver_ab/ passed `use_forkserver` explicitly per arm;
it validated the MECHANISM but never exercised the default. This probe runs
the same order-controlled protocol with the middle arm calling
`execute_proposal_batch` WITH THE KWARG OMITTED -- what every caller gets
after the flip -- so the gates are re-established for the path users
actually take:

  gate 1  arm D (default) records start_method == 'forkserver' on EVERY
          kernel result -- the flip landed AND nothing silently fell back;
  gate 2  order drift A1 vs A2 < 2% (spawn passes bracketing D, so warm
          caches cannot masquerade as the effect);
  gate 3  timeout semantics: per-kernel timeout counts equal across arms,
          and the forced-timeout probe (1s budget both arms must miss)
          returns identical result/error sets for spawn-explicit vs
          default-omitted.

Proposals are replayed from the recorded search_history.db exactly as the
A/B did (no LLM, identical work across arms). Reduced replay sizes
(--cfa-limit 16 --fa-limit 8 by default here) keep the re-verification
affordable; the medians they feed are per-proposal statistics, and the
validity/timeout gates are per-record, not per-median.

MUST RUN ON THE GPU BOX:
    python verification_runs/forkserver_default_2026-08-28/probes/verify_default.py \
        --root /content --out /content/forkserver_default.json
"""
import argparse
import collections
import json
import os
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor

N_WORKERS = 4

REFS = {
    "causal_flash_attention": "TritonBench/reference/causal_flash_attention.py",
    "flash_attention": "TritonBench/reference/flash_attention.py",
}
MUTANTS = {
    "causal_flash_attention": {
        "wrong_causal_mask":
            "TritonBench/cheating/causal_flash_attention/wrong_causal_mask.py",
    },
    "flash_attention": {
        "approx_denom":   "TritonBench/cheating/flash_attention/approx_denom.py",
        "drop_last_tile": "TritonBench/cheating/flash_attention/drop_last_tile.py",
        "skip_rescaling": "TritonBench/cheating/flash_attention/skip_rescaling.py",
        "wrong_mask":     "TritonBench/cheating/flash_attention/wrong_mask.py",
    },
}


def load_proposals(db, operator, limit, InputProposal):
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT proposal_json FROM proposals WHERE operator=? "
        "ORDER BY created_at LIMIT ?", (operator, limit)).fetchall()
    con.close()
    return [InputProposal.from_dict(json.loads(r[0])) for r in rows]


def kernels_for(root, operator):
    ks = [("reference", os.path.join(root, REFS[operator]))]
    ks += [(k, os.path.join(root, v)) for k, v in MUTANTS[operator].items()]
    return ks


def run_arm(root, mode, proposals, operator, timeout, execute_batch):
    """mode: 'spawn' (explicit False) or 'default' (kwarg OMITTED)."""
    ks = kernels_for(root, operator)
    ref = ks[0][1]
    out = []

    def one(p):
        kw = dict(proposal=p, kernels=ks, reference_src_path=ref,
                  operator=operator, timeout_seconds=timeout)
        if mode == "spawn":
            kw["use_forkserver"] = False
        # mode == 'default': deliberately NOT passed -- the point of the run.
        t0 = time.perf_counter()
        res = execute_batch(**kw)
        dt = time.perf_counter() - t0
        return {"proposal_id": p.proposal_id, "proposal_s": dt,
                "kernels": [{
                    "kernel_id": r.kernel_id,
                    "passed_checker": r.passed_checker,
                    "error": r.error.error_type if r.error else None,
                    "start_method": r.start_method,
                    "torch_import_ms": (r.startup_phases or {}).get(
                        "torch_import_ms"),
                } for r in res]}

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for rec in pool.map(one, proposals):
            out.append(rec)
            print(f"    [{mode}] {rec['proposal_id'][:8]} "
                  f"{rec['proposal_s']:6.2f}s", flush=True)
    return out


def med(xs):
    return st.median(xs) if xs else float("nan")


def summarise(passes):
    stats = {}
    for label in ("A1", "D", "A2"):
        recs = passes[label]
        ks = [k for r in recs for k in r["kernels"]]
        methods = collections.Counter(k["start_method"] for k in ks)
        ti = [k["torch_import_ms"] for k in ks
              if k["torch_import_ms"] is not None]
        stats[label] = {
            "median_s": med([r["proposal_s"] for r in recs]),
            "methods": dict(methods),
            "torch_import_p50": med(ti),
            "timeouts": sum(1 for k in ks if k["error"] == "TimeoutError"),
            "n_kernels": len(ks)}
        print(f"  {label:3s} median {stats[label]['median_s']:6.2f}s  "
              f"{stats[label]['methods']}  "
              f"torch_import_p50 {stats[label]['torch_import_p50']:.1f}ms  "
              f"timeouts {stats[label]['timeouts']}")

    bad = {k: v for k, v in stats["D"]["methods"].items() if k != "forkserver"}
    stats["gate1_default_all_forkserver"] = not bad
    a1, a2 = stats["A1"]["median_s"], stats["A2"]["median_s"]
    drift = abs(a1 - a2) / max(a1, a2)
    stats["order_drift"] = drift
    stats["gate2_drift_lt_2pct"] = drift < 0.02
    stats["gate3_timeouts_equal"] = (
        stats["A1"]["timeouts"] == stats["D"]["timeouts"]
        == stats["A2"]["timeouts"])
    eff = stats["D"]["median_s"] / ((a1 + a2) / 2)
    stats["effect_vs_spawn"] = eff
    print(f"  gate1 default-all-forkserver: {stats['gate1_default_all_forkserver']}"
          f"   gate2 drift {drift*100:.1f}% (<2%: {stats['gate2_drift_lt_2pct']})"
          f"   gate3 timeouts equal: {stats['gate3_timeouts_equal']}")
    print(f"  effect: {eff:.2f}x ({(eff-1)*100:+.0f}%) vs spawn mean")
    return stats


def forced_timeout_probe(root, operator, proposals, execute_batch):
    print("\n  -- forced-timeout probe (timeout=1s) --")
    ks = kernels_for(root, operator)
    out = {}
    for mode in ("spawn", "default"):
        kw = dict(proposal=proposals[0], kernels=ks,
                  reference_src_path=ks[0][1], operator=operator,
                  timeout_seconds=1)
        if mode == "spawn":
            kw["use_forkserver"] = False
        t0 = time.perf_counter()
        res = execute_batch(**kw)
        dt = time.perf_counter() - t0
        errs = collections.Counter(
            (r.error.error_type if r.error else None) for r in res)
        out[mode] = {"elapsed_s": dt, "n_results": len(res),
                     "errors": {str(k): v for k, v in errs.items()},
                     "start_methods": dict(collections.Counter(
                         r.start_method for r in res))}
        print(f"     {mode:8s} {dt:6.2f}s  {len(res)}/{len(ks)}  {dict(errs)}")
    out["identical"] = (
        out["spawn"]["n_results"] == out["default"]["n_results"]
        and out["spawn"]["errors"] == out["default"]["errors"])
    print(f"     => {'IDENTICAL' if out['identical'] else 'DIFFERS -- blocker'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content")
    ap.add_argument("--out", default="/content/forkserver_default.json")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--cfa-limit", type=int, default=16)
    ap.add_argument("--fa-limit", type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, args.root)
    import torch
    assert torch.cuda.is_available(), "needs the GPU box"
    import multiprocessing as mp
    assert "forkserver" in mp.get_all_start_methods()

    from verification.adversarial_search.schemas import InputProposal
    from verification.adversarial_search.executor import execute_proposal_batch
    import inspect
    default = inspect.signature(execute_proposal_batch) \
        .parameters["use_forkserver"].default
    print(f"execute_proposal_batch use_forkserver default = {default}")
    assert default is True, "default not flipped -- nothing to verify"

    jobs = [
        ("causal_flash_attention",
         os.path.join(args.root, "adversarial_results/cfa_rerun_2026-08-20/"
                                 "search_history.db"), args.cfa_limit),
        ("flash_attention",
         os.path.join(args.root, "adversarial_results/search_history.db"),
         args.fa_limit),
    ]
    report = {"executor_default": default}
    for operator, db, limit in jobs:
        if not os.path.exists(db):
            print(f"SKIP {operator}: no {db}")
            continue
        props = load_proposals(db, operator, limit, InputProposal)
        print(f"\n== {operator}: {len(props)} proposals x "
              f"{len(kernels_for(args.root, operator))} kernels ==", flush=True)
        passes = {}
        for label, mode in (("A1", "spawn"), ("D", "default"),
                            ("A2", "spawn")):
            print(f"\n-- pass {label} ({mode}) --", flush=True)
            passes[label] = run_arm(args.root, mode, props, operator,
                                    args.timeout, execute_proposal_batch)
            report[operator] = {"passes": passes}
            with open(args.out, "w") as f:
                json.dump(report, f, default=str)
        report[operator]["summary"] = summarise(passes)
        report[operator]["timeout_probe"] = forced_timeout_probe(
            args.root, operator, props, execute_proposal_batch)
        with open(args.out, "w") as f:
            json.dump(report, f, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
