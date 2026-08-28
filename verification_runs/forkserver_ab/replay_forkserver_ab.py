"""
Forkserver A/B: batched+spawn vs batched+forkserver, order-controlled.

MUST BE RUN ON THE GPU BOX. It was authored but NOT executed -- the machine it
was written on has no CUDA device and cannot even install triton (`pip index
versions triton` -> "No matching distribution found"; there is no darwin build).
Every number it prints is therefore absent from the report that ships with it,
deliberately: the point of this harness is that the numbers come from hardware,
not from a projection.

WHAT IS HELD CONSTANT
---------------------
Batching is ALREADY the default and stays on in both arms. The only variable is
HOW each child is created. That is the whole comparison: `_mp_context`'s
docstring is explicit that batching and start method are orthogonal axes, so
mixing them into one A/B would reproduce exactly the confound the batching A/B
was built to avoid.

Proposals are replayed from a recorded `search_history.db` -- no LLM, no network
-- so the work is identical across arms. An LLM-driven run would generate a
different proposal set per arm, which is what makes search-to-search wall-time
comparisons meaningless.

ORDER CONTROL: A1 (spawn) -> B (forkserver) -> A2 (spawn). If A1 and A2 agree,
warm page/JIT caches did not produce the difference. This is the same protocol
as the batching A/B, and it exists because run-order confounding is precisely
what made the checker's per-layer latency table wrong (#7a step 2).

THE FALLBACK CHECK IS NOT OPTIONAL
----------------------------------
`_mp_context` returns the start method ACTUALLY used, because forkserver is
unavailable on some platforms and silently degrades to spawn. A run that never
forked but reports "forkserver made no difference" is worse than no run: it
looks like a measurement and is an absence of one. `summarise()` treats any
forkserver-arm record whose `start_method` is not "forkserver" as a hard failure
of the experiment, not a footnote.

    python verification_runs/forkserver_ab/replay_forkserver_ab.py \
        --root /content --out /content/forkserver_ab.json
"""
import argparse
import collections
import json
import os
import sqlite3
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor

N_WORKERS = 4          # matches the real search: 4 children contending on 1 GPU

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


def run_arm(root, use_forkserver, proposals, operator, timeout, execute_batch):
    ks = kernels_for(root, operator)
    ref = ks[0][1]
    label = "forkserver" if use_forkserver else "spawn"
    out = []

    def one(p):
        t0 = time.perf_counter()
        res = execute_batch(
            proposal=p, kernels=ks, reference_src_path=ref,
            operator=operator, timeout_seconds=timeout,
            use_forkserver=use_forkserver)
        dt = time.perf_counter() - t0
        return {
            "proposal_id": p.proposal_id,
            "proposal_s": dt,
            "kernels": [{
                "kernel_id": r.kernel_id,
                "passed_checker": r.passed_checker,
                "error": r.error.error_type if r.error else None,
                "exec_mode": r.exec_mode,
                # The experiment's validity hinges on this field, not on timing.
                "start_method": r.start_method,
                "kernel_wall_time_ms": r.kernel_wall_time_ms,
                "batch_spawn_ms": r.batch_spawn_ms,
                "startup_phases": r.startup_phases,
            } for r in res],
        }

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for rec in pool.map(one, proposals):
            out.append(rec)
            print(f"    [{label}] {rec['proposal_id'][:8]} "
                  f"{rec['proposal_s']:6.2f}s", flush=True)
    return out


def preflight(root, torch_module):
    """
    Record WHERE torch and the repo actually live, and prove the import was not
    served over a network filesystem.

    This exists because a network-mounted venv silently inflates
    `torch_import_ms` -- the exact quantity this A/B measures -- by orders of
    magnitude. Observed live: an `import torch` from a Google Drive
    CloudStorage mount took ~7 minutes against a few seconds warm, with 1644 of
    1676 stack samples blocked in read()/open() rather than computing.

    A run whose headline number is "forkserver removes the torch import" is
    worthless if that import was measuring a network round-trip, so the run
    records its own filesystem rather than leaving a later reader to infer it
    from paths. The 2026-08-21 batching A/B had to be re-established this way
    after the fact; this makes that unnecessary next time.

    NETWORK_FS lists filesystem types that would invalidate the measurement.
    Anything unrecognised is reported verbatim rather than assumed local --
    silence here would defeat the point.
    """
    NETWORK_FS = {"nfs", "smbfs", "cifs", "fuse", "fuse.gdrive", "fuse.drivefs",
                  "afpfs", "webdav", "9p", "osxfuse", "macfuse", "drivefs"}
    # Path markers catch what fstype lookup cannot. macOS Google Drive presents
    # through FileProvider and is INVISIBLE to both `df` and `mount` -- verified
    # on the machine this was written on, where `df -P` reported the ordinary
    # data volume for a path whose imports were demonstrably going over the
    # network. Colab's Drive mount at /content/drive is likewise a fuse mount
    # that a naive `df` parse can miss.
    PATH_MARKERS = ("cloudstorage", "mydrive", "my drive", "googledrive",
                    "google drive", "gdrive", "drivefs", "/content/drive",
                    "/drive/", "/net/", "/mnt/nfs")

    torch_dir = os.path.dirname(os.path.abspath(torch_module.__file__))
    repo_dir = os.path.abspath(root)
    info = {"torch_path": torch_dir, "repo_root": repo_dir,
            "torch_version": getattr(torch_module, "__version__", "?")}

    probe = os.path.join(torch_dir, "__init__.py")
    try:
        t0 = time.perf_counter()
        with open(probe, "rb") as f:
            f.read()
        info["reread_ms"] = 1000.0 * (time.perf_counter() - t0)
    except OSError as e:
        info["reread_ms"] = None
        info["reread_error"] = str(e)

    def fstype(path):
        """Filesystem type, or None if it genuinely cannot be determined.

        None is a THIRD outcome, not a synonym for "local". An earlier version
        collapsed the two and reported a Google Drive mount as clean, which is
        the precise failure this whole function exists to prevent.
        """
        import subprocess
        try:                                    # GNU: type is column 2
            out = subprocess.run(["df", "-P", "-T", path], capture_output=True,
                                 text=True, timeout=10, check=True).stdout
            lines = [l for l in out.strip().splitlines() if l.strip()]
            if len(lines) > 1 and len(lines[-1].split()) > 1:
                return lines[-1].split()[1]
        except Exception:
            pass
        try:                                    # BSD/macOS: match the mountpoint
            out = subprocess.run(["df", "-P", path], capture_output=True,
                                 text=True, timeout=10, check=True).stdout
            lines = [l for l in out.strip().splitlines() if l.strip()]
            if len(lines) < 2:
                return None
            mp = lines[-1].split()[-1]
            mnt = subprocess.run(["mount"], capture_output=True, text=True,
                                 timeout=10, check=True).stdout
            for line in mnt.splitlines():
                m = re.search(r" on (.+?) \((\w[\w.]*)", line)
                if m and m.group(1) == mp:
                    return m.group(2)
        except Exception:
            pass
        return None

    fs = {"torch": fstype(torch_dir), "repo": fstype(repo_dir)}
    info["filesystems"] = {k: (v or "undetermined") for k, v in fs.items()}

    marker_hits = sorted({m for m in PATH_MARKERS
                          for p in (torch_dir.lower(), repo_dir.lower())
                          if m in p})
    fs_hits = sorted({f"{k}={v}" for k, v in fs.items()
                      if v and v.lower() in NETWORK_FS})
    info["path_markers"] = marker_hits
    info["network_fstypes"] = fs_hits

    print("\n-- preflight --")
    print(f"   torch {info['torch_version']}  at {torch_dir}")
    print(f"   filesystems: torch={info['filesystems']['torch']}  "
          f"repo={info['filesystems']['repo']}")
    if info["reread_ms"] is not None:
        print(f"   warm re-read of torch/__init__.py: {info['reread_ms']:.1f}ms")

    if marker_hits or fs_hits:
        info["network_fs_suspected"] = True
        why = ", ".join(fs_hits + [f"path contains {m!r}" for m in marker_hits])
        print(f"   *** NETWORK FILESYSTEM SUSPECTED: {why}.")
        print(f"       torch_import_ms would measure the network, not the start")
        print(f"       method. Move the repo AND the interpreter to local disk")
        print(f"       before trusting this A/B. ***")
    elif None in fs.values():
        info["network_fs_suspected"] = None
        print("   *** FILESYSTEM UNDETERMINED -- `df`/`mount` could not classify")
        print("       these paths, and this platform may hide network mounts from")
        print("       both (macOS FileProvider does). NOT the same as 'local'.")
        print("       Confirm by hand before trusting torch_import_ms. ***")
    else:
        info["network_fs_suspected"] = False
        print("   filesystems look local; torch_import_ms is trustworthy.")
    return info


def med(xs):
    return st.median(xs) if xs else float("nan")


def _all_kernels(recs):
    return [k for r in recs for k in r["kernels"]]


def summarise(operator, passes):
    print(f"\n  {'pass':6s} {'median s/proposal':>18s} {'start_method':>14s} "
          f"{'torch_import p50':>17s} {'timeouts':>9s}")
    stats = {}
    for label in ("A1", "B", "A2"):
        recs = passes[label]
        ks = _all_kernels(recs)
        methods = collections.Counter(k["start_method"] for k in ks)
        ti = [k["startup_phases"].get("torch_import_ms")
              for k in ks if k.get("startup_phases")
              and k["startup_phases"].get("torch_import_ms") is not None]
        touts = sum(1 for k in ks if k["error"] == "TimeoutError")
        m = med([r["proposal_s"] for r in recs])
        stats[label] = {"median_s": m, "methods": dict(methods),
                        "torch_import_p50": med(ti), "timeouts": touts,
                        "n_kernels": len(ks)}
        print(f"  {label:6s} {m:18.2f} "
              f"{'/'.join(f'{k}:{v}' for k, v in methods.items()):>14s} "
              f"{med(ti):17.1f} {touts:9d}")

    # Validity gate: did arm B actually fork?
    bad = {k: v for k, v in stats["B"]["methods"].items() if k != "forkserver"}
    if bad:
        print(f"\n  *** EXPERIMENT INVALID: arm B used {bad} -- forkserver was "
              f"requested but not used. Do not quote these numbers. ***")
        stats["valid"] = False
        return stats
    stats["valid"] = True

    a1, b, a2 = stats["A1"]["median_s"], stats["B"]["median_s"], stats["A2"]["median_s"]
    drift = abs(a1 - a2) / max(a1, a2) if max(a1, a2) else float("nan")
    print(f"\n  order drift A1 vs A2: {drift * 100:.1f}%  "
          f"({'ok' if drift < 0.10 else 'HIGH -- run order is confounding this'})")
    print(f"  effect: {b / ((a1 + a2) / 2):.2f}x "
          f"({(b / ((a1 + a2) / 2) - 1) * 100:+.0f}%) vs the mean of the two spawn passes")

    # Timeout semantics: the question is whether the START METHOD changed them.
    same = stats["A1"]["timeouts"] == stats["B"]["timeouts"] == stats["A2"]["timeouts"]
    print(f"  per-kernel timeouts: A1={stats['A1']['timeouts']} "
          f"B={stats['B']['timeouts']} A2={stats['A2']['timeouts']} "
          f"-- {'unchanged' if same else 'DIFFER, investigate before flipping default'}")
    return stats


def forced_timeout_probe(root, operator, proposals, execute_batch):
    """
    Timeout semantics are not established by observing zero timeouts in both
    arms -- that is equally consistent with the timeout never being exercised.
    So force one: a 1s budget that both arms must miss, then check that each
    still returns a TimeoutError result per kernel rather than hanging, raising,
    or losing the kernels it never reached.
    """
    print(f"\n  -- forced-timeout probe (timeout=1s, 1 proposal) --")
    ks = kernels_for(root, operator)
    out = {}
    for use_fs in (False, True):
        label = "forkserver" if use_fs else "spawn"
        t0 = time.perf_counter()
        res = execute_batch(
            proposal=proposals[0], kernels=ks, reference_src_path=ks[0][1],
            operator=operator, timeout_seconds=1, use_forkserver=use_fs)
        dt = time.perf_counter() - t0
        errs = collections.Counter(
            (r.error.error_type if r.error else None) for r in res)
        out[label] = {"elapsed_s": dt, "n_results": len(res),
                      "errors": {str(k): v for k, v in errs.items()},
                      "start_methods": collections.Counter(
                          r.start_method for r in res)}
        print(f"     {label:11s} {dt:6.2f}s  {len(res)}/{len(ks)} results  {dict(errs)}")
    same = (out["spawn"]["n_results"] == out["forkserver"]["n_results"]
            and out["spawn"]["errors"] == out["forkserver"]["errors"])
    print(f"     => timeout behaviour {'IDENTICAL' if same else 'DIFFERS -- blocker'}")
    out["identical"] = same
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content", help="repo root on the GPU box")
    ap.add_argument("--out", default="forkserver_ab.json")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--cfa-limit", type=int, default=40)
    ap.add_argument("--fa-limit", type=int, default=12)
    ap.add_argument("--skip-timeout-probe", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.root)
    os.environ.setdefault("CHECKER_ROOT", args.root)

    try:
        import torch
    except ImportError as e:
        sys.exit(f"{e} -- this harness must run on the GPU box.")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device; a forkserver A/B measured on CPU is meaningless.")

    import multiprocessing as mp
    if "forkserver" not in mp.get_all_start_methods():
        sys.exit("this platform has no forkserver start method; arm B cannot run.")

    from verification.adversarial_search.schemas import InputProposal
    from verification.adversarial_search.executor import execute_proposal_batch

    env = preflight(args.root, torch)

    jobs = [
        ("causal_flash_attention",
         os.path.join(args.root, "adversarial_results/cfa_rerun_2026-08-20/"
                                 "search_history.db"), args.cfa_limit),
        ("flash_attention",
         os.path.join(args.root, "adversarial_results/search_history.db"),
         args.fa_limit),
    ]

    report = {}
    for operator, db, limit in jobs:
        if not os.path.exists(db):
            print(f"SKIP {operator}: no {db}")
            continue
        props = load_proposals(db, operator, limit, InputProposal)
        n_k = len(kernels_for(args.root, operator))
        print(f"\n{'=' * 68}\n{operator}: {len(props)} proposals x {n_k} kernels"
              f"\n{'=' * 68}", flush=True)

        passes = {}
        for label, use_fs in (("A1", False), ("B", True), ("A2", False)):
            print(f"\n-- pass {label} "
                  f"({'forkserver' if use_fs else 'spawn'}) --", flush=True)
            t0 = time.time()
            passes[label] = run_arm(args.root, use_fs, props, operator,
                                    args.timeout, execute_proposal_batch)
            print(f"   pass {label} total {time.time() - t0:.1f}s", flush=True)
            # Written after EVERY pass: a reclaimed VM costs one pass, not all.
            report[operator] = {"passes": passes, "environment": env}
            with open(args.out, "w") as f:
                json.dump(report, f, default=str)

        report[operator]["summary"] = summarise(operator, passes)
        if not args.skip_timeout_probe:
            report[operator]["timeout_probe"] = forced_timeout_probe(
                args.root, operator, props, execute_proposal_batch)
        with open(args.out, "w") as f:
            json.dump(report, f, default=str)

    print(f"\nwrote {args.out}")
    print("\nREADY FOR A DEFAULT-FLIP DECISION only if, for BOTH operators:")
    print("  - summary.valid is true (arm B really forked)")
    print("  - order drift A1 vs A2 < 10%")
    print("  - timeout counts match across arms AND timeout_probe.identical")


if __name__ == "__main__":
    main()
