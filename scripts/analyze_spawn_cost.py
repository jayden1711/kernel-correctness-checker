"""
scripts/analyze_spawn_cost.py

Where does an adversarial-search run's wall time go, and what did batching
change? Reads one or two `search_history.db` files and prints the comparison.

    python3 scripts/analyze_spawn_cost.py --db adversarial_results/.../search_history.db
    python3 scripts/analyze_spawn_cost.py --baseline A/search_history.db --after B/search_history.db

READ PER-EXECUTION AND PER-PROPOSAL MEDIANS, NOT TOTAL RUN WALL TIME.
The search is LLM-driven: a run stops the moment a hit is confirmed, and a
worker that loses proposals to rejected JSON produces fewer of them. Two runs
therefore contain different amounts of work, and their totals are not a
like-for-like quantity. Total wall is printed for context and is NOT the
headline. This is the same class of error as timing a subprocess from inside
itself (#7b) and as the dict-order confound in the checker latency table (#7a).

Works on databases written before the timing columns existed: it reconstructs
per-execution intervals from `created_at` gaps within a worker's timeline,
which is how the original 71% figure was derived.
"""

import argparse
import json
import sqlite3
import statistics as st
from collections import defaultdict


def _cols(con, table):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def load(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    have = _cols(con, "executions")

    runs = [dict(r) for r in con.execute(
        "SELECT run_id, operator, n_workers, max_iter, status, "
        "started_at, finished_at FROM runs ORDER BY started_at")]
    props = {r["proposal_id"]: dict(r) for r in con.execute(
        "SELECT proposal_id, run_id, worker_id, created_at FROM proposals")}

    sel = ["proposal_id", "kernel_id", "created_at", "wall_time_ms",
           "passed_checker", "passed_naive"]
    for opt in ("total_wall_time_ms", "exec_mode", "batch_spawn_ms",
                "kernel_wall_time_ms", "startup_phases_json", "start_method"):
        if opt in have:
            sel.append(opt)
    execs = [dict(r) for r in con.execute(
        f"SELECT {', '.join(sel)} FROM executions ORDER BY created_at")]

    # Verdict columns differ across schema generations (not_caught /
    # caught_no_gap were split out of missed_mutants later), so select only
    # what this database actually has rather than failing on an older one.
    vcols = _cols(con, "verdicts")
    want = [c for c in ("proposal_id", "is_hit", "not_caught", "caught_no_gap",
                        "missed_mutants") if c in vcols]
    verdicts = [dict(r) for r in con.execute(
        f"SELECT {', '.join(want)} FROM verdicts")] if want else []
    con.close()
    return runs, props, execs, verdicts


def per_execution_intervals(props, execs):
    """Spawn-to-result interval per execution, from the worker's own timeline.

    The first execution of a proposal is timed from when the proposal row was
    written; each later one from the previous execution of the same proposal.
    Reconstructed rather than read from a column so runs recorded before the
    timing columns existed stay comparable.
    """
    by_prop = defaultdict(list)
    for e in execs:
        by_prop[e["proposal_id"]].append(e)

    first, later, per_proposal = [], [], []
    for pid, lst in by_prop.items():
        if pid not in props:
            continue
        lst.sort(key=lambda e: e["created_at"])
        prev = props[pid]["created_at"]
        total = 0.0
        for i, e in enumerate(lst):
            d = e["created_at"] - prev
            (first if i == 0 else later).append(d)
            total += d
            prev = e["created_at"]
        per_proposal.append(total)
    return first, later, per_proposal


def summarize(name, db_path):
    runs, props, execs, verdicts = load(db_path)
    first, later, per_prop = per_execution_intervals(props, execs)

    print(f"\n{'=' * 72}\n  {name}\n  {db_path}\n{'=' * 72}")
    for r in runs:
        wall = (r["finished_at"] - r["started_at"]) if r["finished_at"] else None
        print(f"  run {r['run_id']}  {r['operator']}  workers={r['n_workers']} "
              f"max_iter={r['max_iter']}  status={r['status']}"
              + (f"  wall={wall:.1f}s" if wall else "  wall=unfinished"))

    modes = defaultdict(int)
    methods = defaultdict(int)
    for e in execs:
        modes[e.get("exec_mode") or "unrecorded"] += 1
        methods[e.get("start_method") or "unrecorded"] += 1

    n_props = len({e["proposal_id"] for e in execs})
    print(f"\n  proposals             {n_props}")
    print(f"  executions            {len(execs)}")
    print(f"  SUBPROCESS SPAWNS     {_spawn_count(execs, n_props)}")
    print(f"  exec_mode             {dict(modes)}")
    # As USED, not as requested: forkserver silently drops to spawn where it is
    # unavailable, and a run that never forked must not be read as evidence that
    # forking did not help.
    print(f"  start_method          {dict(methods)}")

    if per_prop:
        print(f"\n  per-PROPOSAL execute phase   median {st.median(per_prop):8.2f}s"
              f"   mean {st.mean(per_prop):7.2f}s   n={len(per_prop)}")
    if first:
        print(f"  first execution of proposal  median {st.median(first):8.2f}s"
              f"   mean {st.mean(first):7.2f}s   n={len(first)}")
    if later:
        print(f"  each SUBSEQUENT execution    median {st.median(later):8.2f}s"
              f"   mean {st.mean(later):7.2f}s   n={len(later)}")
        print("     ^ this is the number batching attacks: with one subprocess "
              "per proposal\n       these no longer pay startup at all.")
    ik = [e["wall_time_ms"] for e in execs if e.get("wall_time_ms") is not None]
    if ik:
        print(f"\n  in-kernel (wall_time_ms)     median {st.median(ik):8.2f}ms")

    _startup_breakdown(execs)
    _verdicts(verdicts)
    return {"per_prop": per_prop, "first": first, "later": later,
            "n_props": n_props, "n_execs": len(execs),
            "spawns": _spawn_count(execs, n_props), "runs": runs}


def _spawn_count(execs, n_props):
    """One spawn per execution, except that a whole batch shares one.

    A fallback re-run is a genuine extra spawn and is counted as one -- if
    batching were quietly falling back all the time, the spawn count would show
    it even though the exec_mode histogram might be skimmed past.
    """
    batched = [e for e in execs if (e.get("exec_mode") or "") == "batched"]
    if not batched:
        return len(execs)
    n_batches = len({e["proposal_id"] for e in batched})
    n_single = len(execs) - len(batched)
    return n_batches + n_single


def _startup_breakdown(execs):
    phases = defaultdict(list)
    spawn = []
    for e in execs:
        raw = e.get("startup_phases_json")
        if raw:
            for k, v in json.loads(raw).items():
                phases[k].append(v)
        if e.get("batch_spawn_ms") is not None:
            spawn.append(e["batch_spawn_ms"])
    if not phases:
        print("\n  startup breakdown            not recorded in this run")
        return
    print("\n  STARTUP BREAKDOWN (median per subprocess)")
    order = ["pre_module_ms", "torch_import_ms", "spec_import_ms",
             "cuda_init_ms", "materialize_ms"]
    # Costs paid ONCE in another process and inherited by every child. Real, but
    # not per-execution: adding them to the running total would charge one
    # forkserver boot to every fork and report a startup several times the one
    # actually paid. Reported on their own line, outside the sum.
    amortised = {"startup_stamps_inherited_ms"}
    total = 0.0
    for k in order + [k for k in phases if k not in order]:
        if k not in phases:
            continue
        m = st.median(phases[k])
        if k in amortised:
            print(f"    {k:22s} {m:9.1f} ms   "
                  f"<- paid ONCE in another process; NOT in the sum below")
            continue
        total += m
        print(f"    {k:22s} {m:9.1f} ms")
    print(f"    {'(sum of phases)':22s} {total:9.1f} ms")
    if spawn:
        print(f"    {'batch_spawn_ms':22s} {st.median(spawn):9.1f} ms   "
              f"<- measured end to end; the gap to the sum above is unattributed")


def _verdicts(verdicts):
    if not verdicts:
        return
    def _n(field):
        if field not in verdicts[0]:
            return "n/a"
        return sum(1 for v in verdicts if json.loads(v[field] or "[]"))

    hits = sum(1 for v in verdicts if v.get("is_hit"))
    print(f"\n  VERDICTS  hits={hits}  with-not_caught={_n('not_caught')}  "
          f"with-caught_no_gap={_n('caught_no_gap')}  "
          f"with-missed={_n('missed_mutants')}  n={len(verdicts)}")
    print("     ^ batching also SHARES and SEEDS the input tensors, so this "
          "distribution\n       is expected to move a little. It is a declared "
          "change, not a latency artifact.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--baseline")
    ap.add_argument("--after")
    args = ap.parse_args()

    if args.db:
        summarize("run", args.db)
        return
    if not (args.baseline and args.after):
        ap.error("pass --db, or both --baseline and --after")

    a = summarize("BASELINE  (one subprocess per kernel)", args.baseline)
    b = summarize("AFTER     (one subprocess per proposal)", args.after)

    print(f"\n{'=' * 72}\n  COMPARISON\n{'=' * 72}")
    # SPAWNS PER PROPOSAL, not total spawns. A run that stopped early or lost
    # proposals to rejected JSON has fewer of both, and the raw totals would
    # then report a "reduction" that is only a shorter run -- precisely the
    # unnormalised-quantity error this project has made twice before.
    spp_a = a["spawns"] / a["n_props"]
    spp_b = b["spawns"] / b["n_props"]
    print(f"  proposals             {a['n_props']:>8}  ->{b['n_props']:>8}"
          f"   (different runs: totals below are normalised by this)")
    print(f"  spawns (total)        {a['spawns']:>8}  ->{b['spawns']:>8}"
          f"   (NOT comparable on its own)")
    print(f"  SPAWNS PER PROPOSAL   {spp_a:>8.2f}  ->{spp_b:>8.2f}   "
          f"({spp_b / spp_a:.2f}x)")
    for key, label in (("per_prop", "per-proposal execute (median)"),
                       ("later", "subsequent execution (median)")):
        if a[key] and b[key]:
            ma, mb = st.median(a[key]), st.median(b[key])
            print(f"  {label:34s} {ma:7.2f}s ->{mb:7.2f}s   "
                  f"({mb / ma:.2f}x, {100 * (1 - mb / ma):+.0f}%)")
    print("\n  Totals are deliberately not compared: the two runs contain "
          "different\n  numbers of proposals, so their wall times measure "
          "different amounts of work.")


if __name__ == "__main__":
    main()
