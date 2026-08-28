"""
Does the checker recompute the SAME reference output across candidates?

Wraps entry["torch_ref_fn"] at the probe level -- NO repo changes. Every
reference execution in the numeric layer funnels through that callable
(spec.run_reference, check_perturbation_tolerance, check_cross_shape,
check_weight_magnitude, the adversarial battery), so wrapping it captures all
of them.

For each execution records: op, trial identity, input fingerprint, elapsed ms.
The fingerprint is (shape, dtype, sum, sumsq, min, max) as float64 -- enough to
separate genuinely different draws, and cheap enough not to dominate. It is
computed AFTER the timed region so it never inflates the per-call timing.
"""
import json, os, sys, time, collections
sys.path.insert(0, "/content")
sys.path.insert(0, "/content/benchmarks/autokernel/files")
import torch, numpy as np
import my_corpus
import checker_adapter as ca

N_REF = int(os.environ.get("KCC_N_REF", "5"))
CALLS = []
CTX = {"op": None, "trial": None}


def _fingerprint(args):
    parts = []
    for a in (args if isinstance(args, tuple) else (args,)):
        if torch.is_tensor(a):
            f = a.detach().float()
            st = torch.stack([f.sum(), (f * f).sum(), f.min(), f.max()]).double()
            s, sq, mn, mx = [float(v) for v in st.cpu().tolist()]
            parts.append((tuple(a.shape), str(a.dtype), round(s, 6), round(sq, 6),
                          round(mn, 6), round(mx, 6)))
        else:
            parts.append(("scalar", repr(a)))
    return repr(parts)


_SITES = ("check_perturbation_tolerance", "check_cross_shape",
          "check_weight_magnitude", "check_kernel_executed", "_time_once",
          "check_output_shape", "check_backward_pass", "_check_exact_match",
          "check_all_tiles_visited_generic", "check_determinism")


def _callsite():
    """Nearest enclosing check function. `_time_once` is reported separately:
    it is the delegation detector's timing loop, whose verdict IS the elapsed
    time, so those executions can never be served from a cache."""
    f = sys._getframe(2)
    depth = 0
    while f is not None and depth < 25:
        n = f.f_code.co_name
        if n in _SITES:
            return n
        f = f.f_back; depth += 1
    return "other"


def wrap(fn, op):
    def inner(*args, **kwargs):
        a = args[0] if len(args) == 1 else args
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0)
        try:
            fp = _fingerprint(a)
        except Exception as e:
            fp = f"ERR{e}"
        CALLS.append({"op": op, "trial": CTX["trial"], "ms": ms, "fp": fp,
                      "site": _callsite()})
        return out
    return inner


rng = np.random.default_rng(0)
t_start = time.perf_counter()
for entry in my_corpus.CORPUS:
    op = entry["op"]
    e = dict(entry)
    e["torch_ref_fn"] = wrap(entry["torch_ref_fn"], op)
    # mutant fn left UNWRAPPED so candidate executions are excluded by
    # construction -- the question is reference-only cost.
    for idx in range(1 + N_REF):
        is_mut = (idx == 0)
        CTX["trial"] = f"{op}/{entry['mutant_name']}/{'M' if is_mut else 'R'+str(idx)}"
        ee = dict(e)
        if not is_mut:
            # reference trial: candidate IS the reference. Wrap it too, under a
            # separate tag, so the candidate-side reference executions are
            # visible but distinguishable.
            ee["torch_mutant_fn"] = e["torch_ref_fn"]
        try:
            ca.my_checker_system(ee, is_mut, rng)
        except Exception as ex:
            print("ERR", CTX["trial"], ex, flush=True)
    print(f"{op}/{entry['mutant_name']} done, {len(CALLS)} ref calls so far", flush=True)

wall = time.perf_counter() - t_start
tot_ms = sum(c["ms"] for c in CALLS)
fps = collections.Counter(c["fp"] for c in CALLS)
dup_calls = sum(v - 1 for v in fps.values() if v > 1)
print(f"\nTOTAL reference executions: {len(CALLS)}")
print(f"distinct fingerprints: {len(fps)}")
print(f"redundant executions (a fingerprint already seen): {dup_calls}")
print(f"reference-only time: {tot_ms/1000:.2f}s   probe wall: {wall:.1f}s")
bysite = collections.Counter(c["site"] for c in CALLS)
print("\nreference executions by call site:", dict(bysite.most_common()))
json.dump({"calls": CALLS, "wall_s": wall, "n_ref": N_REF},
          open("/content/refshare/calls.json", "w"))
