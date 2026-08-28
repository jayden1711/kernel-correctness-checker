"""
forkserver pre-flight — four questions that can only be answered on hardware.

This is a GATE, not a measurement. Two of its four probes can kill the change
outright, and it exists so that they kill it in 60 seconds rather than halfway
through an A/B whose numbers would then be unusable.

  1. CUDA AFTER FORK. The whole approach rests on `import torch` NOT initialising
     CUDA. If it does, the forkserver holds a CUDA context, every fork inherits a
     copy of it, and every child dies with an initialisation error. Probe 1 forks
     children off a torch-preloaded server and has each run a REAL Triton kernel
     from the corpus. Anything other than clean success ends the change here.

  2. FORK FROM A MULTITHREADED PARENT. torch starts threads at import. Forking a
     multithreaded process is safe only if the child avoids whatever locks those
     threads held; the failure mode is a HANG, not a crash, so this probe is
     bounded by a deadline and reports a timeout as a failure rather than
     waiting forever. Four concurrent threads, matching the search's worker
     count.

  3. WHAT A CHILD ACTUALLY INHERITS. Forks with NO seeding, each drawing
     `torch.randn`. If they come back identical, the inherited-RNG hazard is
     confirmed on real torch rather than argued from documentation -- and the
     offline control in `tests/instrumentation/check_forkserver_executor.py` is
     shown to be guarding a real thing. This project has twice shipped a control
     that could not observe what it asserted (§5 instances 11, 12), so the
     "obvious" answer gets measured too.

  4. WHERE THE STARTUP GOES. `import torch` was 5241ms of ~6185ms (85%) under
     spawn. Probe 4 reports the same decomposition under forkserver, separating
     the FIRST child (which pays the forkserver's own boot) from the rest.

Run on the VM:
    PYTHONPATH=/content python3 /content/verification_runs/forkserver_2026-08-21/preflight.py
Writes preflight.json next to itself and prints a PASS/FAIL gate verdict.
"""
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback

ROOT = os.environ.get("CHECKER_ROOT", "/content")
sys.path.insert(0, ROOT)

# A real corpus kernel, not a synthetic one: the question is whether Triton
# compiles and launches in a forked child, and only a real @triton.jit kernel
# answers it.
KERNEL = os.path.join(ROOT, "TritonBench/reference/softmax.py")

N_FORKS = 6
N_THREADS = 4
PER_THREAD = 5
DEADLINE_S = 180.0


def _load(path, name):
    spec = importlib.util.spec_from_file_location("_probe_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


# ── child bodies (module level: forkserver pickles them by reference) ─────────

def child_kernel(q, idx):
    """Probe 1: does a real Triton kernel run in a forked child?"""
    t0 = time.time()
    rec = {"idx": idx, "pid": os.getpid()}
    try:
        import torch
        rec["torch_already_imported"] = "torch" in sys.modules
        t_cuda = time.time()
        torch.zeros(1, device="cuda")
        rec["cuda_init_ms"] = 1000.0 * (time.time() - t_cuda)

        fn = _load(KERNEL, "softmax")
        x = torch.randn(128, 256, device="cuda", dtype=torch.float32)
        t_k = time.time()
        out = fn(x)
        torch.cuda.synchronize()
        rec["kernel_ms"] = 1000.0 * (time.time() - t_k)

        ref = torch.softmax(x, dim=-1)
        rec["max_err"] = float((out.float() - ref.float()).abs().max().item())
        rec["ok"] = True
    except Exception as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = "\n".join(traceback.format_exc().splitlines()[-5:])
    rec["total_ms"] = 1000.0 * (time.time() - t0)
    q.put(rec)


def child_rng(q, idx):
    """Probe 3: with NO seeding, what does a forked child draw?"""
    import torch
    q.put({"idx": idx, "pid": os.getpid(),
           "draw": [round(v, 8) for v in torch.randn(4).tolist()]})


def child_phases(q, idx, parent_t):
    """Probe 4: the startup decomposition, measured from inside the child."""
    t_entry = time.time()
    rec = {"idx": idx, "pid": os.getpid(),
           "pre_module_ms": 1000.0 * (t_entry - parent_t)}
    t0 = time.time()
    import torch                                             # noqa: F401
    rec["torch_import_ms"] = 1000.0 * (time.time() - t0)
    t0 = time.time()
    torch.zeros(1, device="cuda")
    rec["cuda_init_ms"] = 1000.0 * (time.time() - t0)
    rec["total_to_ready_ms"] = 1000.0 * (time.time() - parent_t)
    q.put(rec)


# ── driver ────────────────────────────────────────────────────────────────────

def make_ctx(method):
    ctx = mp.get_context(method)
    if method == "forkserver":
        ctx.set_forkserver_preload(["torch"])
    return ctx


def run_one(ctx, target, args, deadline=DEADLINE_S):
    """Start one child and collect its record, or report how it failed.

    A HANG is the failure mode probe 2 is looking for, so this never joins
    without a timeout -- an unbounded join would turn a detected deadlock into
    a test that simply never finishes.
    """
    q = ctx.Queue()
    t0 = time.time()
    p = ctx.Process(target=target, args=(q,) + args)
    p.start()
    p.join(timeout=deadline)
    if p.is_alive():
        p.kill()
        p.join()
        return {"ok": False, "error": "HANG: child exceeded deadline",
                "wall_ms": 1000.0 * (time.time() - t0)}
    if q.empty():
        return {"ok": False, "error": f"child exited ({p.exitcode}) with no result",
                "wall_ms": 1000.0 * (time.time() - t0)}
    rec = q.get()
    rec.setdefault("ok", True)
    rec["wall_ms"] = 1000.0 * (time.time() - t0)
    return rec


def main():
    report = {"env": {}}
    import torch
    report["env"] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "start_methods": mp.get_all_start_methods(),
        "python": sys.version.split()[0],
    }
    print(json.dumps(report["env"], indent=2), flush=True)

    if "forkserver" not in mp.get_all_start_methods():
        report["gate"] = "FAIL: forkserver unavailable on this platform"
        print(report["gate"])
        return report, False

    gate_ok = True

    # ── Probe 1 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*66}\n  PROBE 1: real Triton kernel in a forked child\n{'='*66}",
          flush=True)
    ctx = make_ctx("forkserver")
    p1 = [run_one(ctx, child_kernel, (i,)) for i in range(N_FORKS)]
    report["probe1_cuda_after_fork"] = p1
    n_ok = sum(1 for r in p1 if r.get("ok"))
    for r in p1:
        print(f"    fork {r['idx'] if 'idx' in r else '?'} pid={r.get('pid')} "
              f"ok={r.get('ok')} max_err={r.get('max_err')} "
              f"{r.get('error', '')}", flush=True)
    ok1 = n_ok == N_FORKS and all(
        r.get("max_err", 1.0) < 1e-2 for r in p1 if r.get("ok"))
    print(f"  -> {n_ok}/{N_FORKS} forks ran a real Triton kernel correctly")
    if not ok1:
        gate_ok = False
        print("  -> GATE FAILURE: CUDA is not usable after fork. STOP HERE.")

    # ── Probe 2 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*66}\n  PROBE 2: {N_THREADS} threads x {PER_THREAD} forks "
          f"(watching for a HANG, not a crash)\n{'='*66}", flush=True)
    results, errors = [], []

    def worker(wid):
        for i in range(PER_THREAD):
            try:
                results.append(run_one(ctx, child_kernel, (wid * 100 + i,),
                                       deadline=120.0))
            except Exception as e:
                errors.append(f"w{wid}: {type(e).__name__}: {e}")

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=DEADLINE_S)
    p2_wall = time.time() - t0
    stuck = [t for t in threads if t.is_alive()]
    n_ok2 = sum(1 for r in results if r.get("ok"))
    report["probe2_concurrency"] = {
        "wall_s": p2_wall, "completed": len(results), "ok": n_ok2,
        "expected": N_THREADS * PER_THREAD, "stuck_threads": len(stuck),
        "errors": errors,
        "hangs": [r for r in results if "HANG" in str(r.get("error", ""))],
    }
    print(f"    {n_ok2}/{N_THREADS * PER_THREAD} ok in {p2_wall:.1f}s, "
          f"{len(stuck)} threads still alive, {len(errors)} errors", flush=True)
    ok2 = not stuck and n_ok2 == N_THREADS * PER_THREAD
    if not ok2:
        gate_ok = False
        print("  -> GATE FAILURE: forking from torch's threaded runtime is not "
              "safe here. STOP HERE.")

    # ── Probe 3 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*66}\n  PROBE 3: what an UNSEEDED forked child inherits\n{'='*66}",
          flush=True)
    p3_fork = [run_one(ctx, child_rng, (i,)) for i in range(N_FORKS)]
    spawn_ctx = make_ctx("spawn")
    p3_spawn = [run_one(spawn_ctx, child_rng, (i,)) for i in range(3)]
    draws_f = [tuple(r["draw"]) for r in p3_fork if "draw" in r]
    draws_s = [tuple(r["draw"]) for r in p3_spawn if "draw" in r]
    report["probe3_rng"] = {
        "forkserver_draws": [list(d) for d in draws_f],
        "spawn_draws": [list(d) for d in draws_s],
        "forkserver_distinct": len(set(draws_f)),
        "spawn_distinct": len(set(draws_s)),
    }
    print(f"    forkserver: {len(set(draws_f))} distinct draws across "
          f"{len(draws_f)} unseeded children")
    print(f"    spawn:      {len(set(draws_s))} distinct draws across "
          f"{len(draws_s)} unseeded children")
    hazard_real = len(draws_f) > 1 and len(set(draws_f)) == 1
    print("    -> inherited-RNG hazard CONFIRMED on real torch"
          if hazard_real else
          "    -> NOTE: forks did NOT draw identically; re-read the seeding "
          "argument before trusting it")

    # ── Probe 4 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*66}\n  PROBE 4: startup decomposition, spawn vs forkserver\n{'='*66}",
          flush=True)
    phases = {}
    for label, c in (("spawn", spawn_ctx), ("forkserver", make_ctx("forkserver"))):
        rows = []
        for i in range(N_FORKS):
            t = time.time()
            rows.append(run_one(c, child_phases, (i, t)))
        phases[label] = rows
        # The FIRST child of a forkserver run pays the daemon's own boot; every
        # later one does not. Averaging them together would understate the
        # steady state and overstate the first proposal, so they are reported
        # separately rather than as one median.
        def med(key, rs):
            vals = sorted(r[key] for r in rs if key in r)
            return vals[len(vals) // 2] if vals else float("nan")
        print(f"    {label:11s} first: total {rows[0].get('total_to_ready_ms', float('nan')):8.1f} ms   "
              f"steady median: pre_module {med('pre_module_ms', rows[1:]):7.1f}  "
              f"torch_import {med('torch_import_ms', rows[1:]):7.1f}  "
              f"cuda_init {med('cuda_init_ms', rows[1:]):7.1f}  "
              f"total {med('total_to_ready_ms', rows[1:]):8.1f}", flush=True)
    report["probe4_phases"] = phases

    report["gate"] = "PASS" if gate_ok else "FAIL"
    report["probe_verdicts"] = {
        "cuda_after_fork": bool(ok1),
        "concurrent_forks": bool(ok2),
        "rng_hazard_confirmed": bool(hazard_real),
    }
    print(f"\n{'='*66}\n  GATE: {report['gate']}\n{'='*66}")
    return report, gate_ok


if __name__ == "__main__":
    rep, ok = main()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preflight.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {out}")
    sys.exit(0 if ok else 1)
