"""
Live Triton JIT cache hit/miss instrumentation for the checker pipeline.

MUST BE RUN ON THE GPU BOX. It is the empirical half of the pair whose static
half is `analyze_jit_specializations.py`; that script models what the cache
SHOULD do from the call sites and shapes, this one records what it actually
does. `triton` has no darwin build at all, so this file was authored but NOT
executed -- treat its output as unverified until someone runs it on CUDA.

WHAT IT RECORDS
---------------
For every candidate-kernel launch the checker triggers:
  - whether Triton served it from cache or compiled fresh
  - the specialization key that decided that
  - which checker layer and check name triggered it
  - how long the compile took, when it was a miss

HOW IT HOOKS
------------
Triton's `JITFunction.run` consults an in-process dict (`JITFunction.cache`,
keyed by device then by specialization key) before compiling. Rather than
depend on that internal layout -- it has moved across 2.x/3.x -- this wraps
`JITFunction.run` and infers hit vs miss by observing whether the cache
population GREW across the call. That works on any version exposing a dict-like
`.cache`, and degrades to a timing heuristic if it does not (see `_CacheProbe`).

The layer/check attribution comes from a contextvar the checker sets. Rather
than patch verification/checker.py (this is a measurement pass -- no production
edits), `_patch_run_check` wraps `KernelChecker._run_check` at runtime, which is
the single funnel every check already flows through.

USAGE
    python benchmarks/instrument_triton_cache.py                # whole corpus
    python benchmarks/instrument_triton_cache.py --op softmax   # one operator
    python benchmarks/instrument_triton_cache.py --json out.json

READING THE OUTPUT
------------------
The number that answers the question is "same-key recompiles". It should be
ZERO. Anything above zero means two call sites paid separately for an identical
specialization, which is the actionable bug the investigation was looking for.
A high miss count with zero same-key recompiles means the opposite: every
compile bought a distinct specialization and the cost is irreducible here.
"""
import argparse
import collections
import contextvars
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "autokernel" / "files"))

CURRENT_SITE = contextvars.ContextVar("checker_site", default=("?", "?"))

# One row per kernel launch that reached Triton.
Launch = collections.namedtuple(
    "Launch", "layer check kernel key hit compile_ms")
LAUNCHES = []


class _CacheProbe:
    """
    Wraps triton.runtime.JITFunction.run and classifies each launch.

    Hit/miss is inferred from cache growth rather than from any particular
    internal field, so it survives the 2.x -> 3.x cache refactors. If the cache
    cannot be sized at all, `sizer` returns None and every launch is recorded
    with hit=None -- reported separately rather than silently counted as a hit,
    because "unknown" and "cached" are exactly the two things this script exists
    to distinguish.
    """

    def __init__(self, jit_cls):
        self.jit_cls = jit_cls
        self._orig = jit_cls.run

    @staticmethod
    def _size(fn):
        cache = getattr(fn, "cache", None)
        if cache is None:
            return None
        try:
            if isinstance(cache, dict):
                # {device: {key: kernel}} on 2.x/3.x, or a flat {key: kernel}.
                if cache and all(isinstance(v, dict) for v in cache.values()):
                    return sum(len(v) for v in cache.values())
                return len(cache)
            return len(cache)
        except TypeError:
            return None

    def install(self):
        orig, size = self._orig, self._size

        def run(fn_self, *args, **kwargs):
            before = size(fn_self)
            t0 = time.perf_counter()
            out = orig(fn_self, *args, **kwargs)
            dt = (time.perf_counter() - t0) * 1000
            after = size(fn_self)

            hit = None if (before is None or after is None) else (after == before)
            layer, check = CURRENT_SITE.get()
            LAUNCHES.append(Launch(
                layer=layer, check=check,
                kernel=getattr(fn_self, "__name__", repr(fn_self)),
                key=_last_key(fn_self),
                hit=hit,
                compile_ms=(0.0 if hit else dt),
            ))
            return out

        self.jit_cls.run = run

    def remove(self):
        self.jit_cls.run = self._orig


def _last_key(fn):
    """Best-effort read of the specialization key just used."""
    cache = getattr(fn, "cache", None)
    try:
        if isinstance(cache, dict) and cache:
            inner = next(iter(cache.values()))
            if isinstance(inner, dict) and inner:
                return str(list(inner.keys())[-1])[:120]
            return str(list(cache.keys())[-1])[:120]
    except Exception:
        pass
    return "<unavailable>"


def _patch_run_check(KernelChecker):
    """Tag every launch with the layer/check that triggered it."""
    orig = KernelChecker._run_check

    def _run_check(self, layer, name, fn):
        token = CURRENT_SITE.set((layer, name))
        try:
            return orig(self, layer, name, fn)
        finally:
            CURRENT_SITE.reset(token)

    KernelChecker._run_check = _run_check


def report(launches):
    total = len(launches)
    if not total:
        print("no kernel launches recorded -- did the corpus actually run?")
        return {}

    hits = sum(1 for l in launches if l.hit is True)
    misses = sum(1 for l in launches if l.hit is False)
    unknown = sum(1 for l in launches if l.hit is None)
    compile_ms = sum(l.compile_ms for l in launches)

    print(f"\nlaunches {total}   hits {hits}   misses (compiles) {misses}"
          + (f"   UNKNOWN {unknown}" if unknown else ""))
    if unknown:
        print("  WARNING: cache could not be sized on this Triton build; the")
        print("  unknown rows are NOT counted as hits. Fix _CacheProbe._size")
        print("  before trusting the ratio.")
    print(f"hit rate {hits / total * 100:.1f}%   total compile time {compile_ms / 1000:.2f}s")

    # THE load-bearing number: the same (kernel, key) compiled more than once.
    per_key = collections.Counter(
        (l.kernel, l.key) for l in launches if l.hit is False)
    dupes = {k: n for k, n in per_key.items() if n > 1}
    print(f"\nsame-key recompiles: {len(dupes)}")
    if dupes:
        wasted = 0.0
        for (kernel, key), n in sorted(dupes.items(), key=lambda kv: -kv[1]):
            rows = [l for l in launches
                    if l.hit is False and (l.kernel, l.key) == (kernel, key)]
            extra = sum(r.compile_ms for r in rows[1:])
            wasted += extra
            sites = sorted({(r.layer, r.check) for r in rows})
            print(f"  {kernel} key={key[:48]} compiled {n}x "
                  f"(+{extra:.1f}ms) at {sites}")
        print(f"\n  ACTIONABLE: {wasted / 1000:.2f}s of duplicate compilation "
              f"({wasted / compile_ms * 100:.1f}% of compile time)")
    else:
        print("  none -- every compile bought a DISTINCT specialization.")
        print("  The compile bucket is irreducible at this layer.")

    print("\ncompiles by layer / check")
    by_site = collections.Counter((l.layer, l.check)
                                  for l in launches if l.hit is False)
    for (layer, check), n in by_site.most_common(15):
        ms = sum(l.compile_ms for l in launches
                 if l.hit is False and (l.layer, l.check) == (layer, check))
        print(f"  L{layer} {check:34s} {n:4d} compiles  {ms / 1000:6.2f}s")

    return {
        "launches": total, "hits": hits, "misses": misses, "unknown": unknown,
        "hit_rate": hits / total, "compile_seconds": compile_ms / 1000,
        "same_key_recompiles": len(dupes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", help="restrict to one operator")
    ap.add_argument("--json", help="write the summary here")
    args = ap.parse_args()

    try:
        import torch
        import triton
    except ImportError as e:
        sys.exit(f"{e}. This script requires a CUDA box with triton installed; "
                 "see analyze_jit_specializations.py for the static analysis "
                 "that runs anywhere.")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device -- Triton will not compile anything to measure.")

    from verification.checker import KernelChecker
    import numpy as np
    from tritonbench_registry import build_corpus

    _patch_run_check(KernelChecker)
    probe = _CacheProbe(triton.runtime.JITFunction)
    probe.install()

    try:
        corpus = build_corpus()
        if args.op:
            corpus = [e for e in corpus if e["op"] == args.op]
            if not corpus:
                sys.exit(f"no corpus entry for op={args.op!r}")

        rng = np.random.default_rng(0)
        for entry in corpus:
            spec = entry["spec"]
            inputs = entry["to_torch"](entry["input_fn"](rng))
            for is_mutant in (True, False):
                cand = entry["torch_mutant_fn"] if is_mutant else entry["torch_ref_fn"]
                raw = entry["raw_kernel_mutant"] if is_mutant else entry["raw_kernel_ref"]
                try:
                    KernelChecker(spec).run(cand, raw, entry["torch_ref_fn"], inputs)
                except Exception as e:      # a candidate that raises still tells
                    print(f"  {entry['op']} "                  # us what it compiled
                          f"({'mutant' if is_mutant else 'ref'}) raised: "
                          f"{type(e).__name__}: {e}")
    finally:
        probe.remove()

    summary = report(LAUNCHES)
    if args.json and summary:
        Path(args.json).write_text(json.dumps(
            {"summary": summary,
             "launches": [l._asdict() for l in LAUNCHES]}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
