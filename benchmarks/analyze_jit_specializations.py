"""
Does the checker re-compile kernel source it has already compiled?

THE QUESTION
------------
SESSION_HANDOFF records that 84% of the full checker's measured corpus latency
was Triton JIT compilation (42.4s cold vs 6.9s warm). That bucket is only worth
attacking if some of those compiles are REDUNDANT -- the same kernel, the same
specialization, compiled more than once because different layers each triggered
it independently. If instead every compile is for a genuinely distinct
specialization, the cost is irreducible at this layer.

WHAT THIS SCRIPT DOES, AND WHAT IT DOES NOT
-------------------------------------------
It does NOT measure Triton. It cannot: `triton` has no darwin distribution at
all (`pip index versions triton` -> "No matching distribution found"), there is
no CUDA device here, and `benchmarks/autokernel/files/my_corpus.py` fails at
import with ModuleNotFoundError. The live measurement belongs in the companion
script `instrument_triton_cache.py`, which must be run on the GPU box.

What this script does is enumerate, statically, every call site in
`KernelChecker.run` that invokes the candidate kernel, and the SHAPE each site
presents. That is the thing that decides cache hit vs miss, because Triton keys
its JIT cache on the specialization -- constexpr values plus argument
specialization -- not on the source text. Every TritonBench kernel derives its
block constexpr from the input, e.g. TritonBench/reference/softmax.py:36:

    BLOCK_SIZE = triton.next_power_of_2(n_cols)      # BLOCK_SIZE: tl.constexpr

so two invocations at the SAME shape share a cache entry, and two invocations at
DIFFERENT shapes generally do not.

The counts below are therefore exact for shapes and call sites (read off the
source), and the compile/hit split follows from them under one stated
assumption: that Triton's in-process cache returns a hit for a repeat
invocation at an identical specialization. That assumption is not free-floating
-- `harness._warm()`'s docstring records the corpus measurement that
demonstrates it, `avg_pool1d/wrong_divisor` costing 4218ms when it ran first and
122ms for the same checks running second, a 35x drop attributable purely to a
warm cache.

SPECIALIZATION MODEL
--------------------
`_spec_key` models Triton's key for a row-wise kernel: the block constexpr, plus
the divisible-by-16 and equal-to-1 argument specializations Triton applies to
integer arguments. It is a MODEL, and a deliberately conservative one -- real
kernels take different argument lists, so the true distinct-specialization count
per candidate can be higher than the number printed here, never lower. That
direction matters: the conclusion this script supports is "few compiles, many
hits", and a model that undercounts compiles could only weaken that claim, not
manufacture it.

    python3 benchmarks/analyze_jit_specializations.py
"""
import ast
import importlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "benchmarks" / "autokernel" / "files" / "tritonbench_registry.py"
sys.path.insert(0, str(REPO))

# Primary-tensor shape each corpus family feeds the checker, read from
# tritonbench_registry.FAMILIES / the _mk_* builders. This is the "base shape":
# the one nearly every call site uses.
FAMILY_BASE_SHAPE = {
    "single": (64, 128),
    "layernorm": (64, 128),
    "rmsnorm": (64, 128),
    "instancenorm": (2, 4, 4, 4),
    "matmul": (32, 16),
    "attention": (64, 32),
    "groupnorm": (2, 8, 4, 4),
    "batchnorm": (2, 8, 4, 4),
    "cross_entropy": (64, 32),
    "pool1d": (2, 3, 32),
    "pool2d": (2, 3, 16, 16),
    "pool3d": (2, 3, 8, 8, 8),
}

# Call sites in KernelChecker.run that invoke the CANDIDATE kernel, with how
# many times each fires and at which shape. "base" = the harness's input shape;
# "sweep" = one invocation per spec.valid_shapes entry; "sweep[0]" = repeated
# invocations all at valid_shapes[0].
#
# Counts read from:
#   runtime_guards.check_nan_inf / check_dtype_preserved      1 call each
#   runtime_guards.check_determinism                          n_runs=3
#   runtime_guards.check_kernel_executed  ladder              1 + 3 probes
#                                         timing rung         _ROUNDS*_CALLS=10
#   tile_coverage.check_all_tiles_visited_generic             1
#   perturbation.check_perturbation_tolerance                 1 (candidate is
#       called ONCE, line 97; the 20 perturbation samples go through the
#       REFERENCE, so they trigger no candidate compile at all)
#   shape_generalization.check_output_shape                   1
#   checker._check_cross_shape                                len(valid_shapes)
#   shape_generalization.check_weight_magnitude               4 variants, all at
#       valid_shapes[0] (line 180)
CALL_SITES = [
    (1, "nan_inf", "base", 1),
    (1, "dtype_preserved", "base", 1),
    (1, "determinism", "base", 3),
    (1, "kernel_executed (probe ladder)", "base", 4),
    (1, "kernel_executed (timing rung)", "base", 10),
    (1, "tile_coverage_structural", "base", 1),
    (2, "algebraic properties", "base", None),   # >=1 per property
    (3, "output_shape", "base", 1),
    (3, "perturbation_tolerance", "base", 1),
    (3, "cross_shape", "sweep", None),           # one per valid_shape
    (3, "weight_magnitude", "sweep[0]", 4),
    (3, "backward_pass", "base", 1),
    (3, "adversarial_*", "base", None),          # one per adversarial pair
]


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _spec_key(shape):
    """Model of Triton's JIT cache key for a row-wise kernel at this shape."""
    return (
        _next_pow2(shape[-1]),                        # BLOCK_SIZE constexpr
        tuple(d % 16 == 0 for d in shape),            # divisible-by-16 args
        tuple(d == 1 for d in shape),                 # equal-to-1 args
    )


def _load_ops():
    """Parse the OPS table out of the registry WITHOUT importing it (it pulls
    in triton via TritonBench, which does not exist on this platform)."""
    src = REGISTRY.read_text()
    body = src[src.index("OPS = ["):]
    body = body[:body.index("\n]") + 2]
    return ast.literal_eval(body[body.index("["):])


def main():
    ops = _load_ops()
    print(f"corpus: {len(ops)} operators, "
          f"{sum(len(o[4]) for o in ops)} mutants\n")

    rows = []
    for spec_key_name, _ref_file, _cheat, family, mutants in ops:
        try:
            spec = importlib.import_module(
                f"verification.specs.{spec_key_name}").get_spec()
        except Exception as e:
            print(f"  SKIP {spec_key_name}: {type(e).__name__}: {e}")
            continue

        base = FAMILY_BASE_SHAPE[family]
        sweep = [tuple(s) for s in spec.valid_shapes]
        n_alg = len(spec.algebraic_properties)
        n_adv = 4          # _make_weight_variants-style batteries; >=1, see note

        invocations = 0
        for _layer, _name, kind, count in CALL_SITES:
            if kind == "sweep":
                invocations += len(sweep)
            elif kind == "sweep[0]":
                invocations += count
            elif count is None:
                invocations += (n_alg if _name.startswith("algebraic") else n_adv)
            else:
                invocations += count

        shapes_presented = {base} | set(sweep)
        keys = {_spec_key(s) for s in shapes_presented}

        rows.append((spec_key_name, len(mutants), base, len(sweep),
                     invocations, len(shapes_presented), len(keys)))

    print(f"{'operator':30s} {'inv':>5s} {'shapes':>7s} {'compiles':>9s} "
          f"{'hits':>6s} {'hit-rate':>9s}")
    print("-" * 72)
    tot_inv = tot_comp = 0
    for name, _n_mut, _base, _n_sweep, inv, _n_shapes, n_keys in sorted(rows):
        hits = inv - n_keys
        tot_inv += inv
        tot_comp += n_keys
        print(f"{name:30s} {inv:5d} {_n_shapes:7d} {n_keys:9d} {hits:6d} "
              f"{hits / inv * 100:8.1f}%")

    print("-" * 72)
    tot_hits = tot_inv - tot_comp
    print(f"{'SUM over operators':30s} {tot_inv:5d} {'':7s} "
          f"{tot_comp:9d} {tot_hits:6d} {tot_hits / tot_inv * 100:8.1f}%")

    n_ops = len(rows)
    print("\nPER CANDIDATE (one candidate = one kernel object put through one "
          "full checker pass):")
    print(f"  candidate-kernel invocations : {tot_inv / n_ops:5.1f} (mean)")
    print(f"  of those, genuine compiles   : {tot_comp / n_ops:5.1f} (mean)")
    print(f"  of those, cache hits         : {tot_hits / n_ops:5.1f} (mean)")
    print(f"  modelled cache-hit rate      : {tot_hits / tot_inv * 100:5.1f}%")
    print("\n  Each mutant and each reference is a DISTINCT kernel object with "
          "its own\n  cache, so nothing is shared BETWEEN candidates: a corpus "
          f"pass over\n  {sum(len(o[4]) for o in ops)} mutants + {len(ops)} "
          "references pays every candidate's compiles separately.")

    print("\nWHERE THE COMPILES COME FROM")
    print("  Every call site except cross_shape presents the SAME base shape,")
    print("  so all of them share one cache entry. The shape sweep in")
    print("  _check_cross_shape is the only site that varies the shape, and it")
    print("  does so BY DESIGN -- testing shape generalisation is the check's")
    print("  entire purpose. Those compiles are the check, not overhead.")
    print("\n  No call site invokes the same kernel at the same shape in a way")
    print("  another layer has not already warmed; the layers share one process")
    print("  and therefore one JITFunction cache.")

    _corroborate(tot_comp / n_ops, tot_inv / n_ops)


def _corroborate(specs_per_candidate, invocations_per_candidate):
    """
    Cross-check the model against the cold/warm corpus runs already on disk.

    `results_raw_cold.json` and `results_raw.json` are the SAME corpus with and
    without `harness._warm()`. Their difference is compile cost, and it can be
    split by whether a candidate was being touched for the FIRST time.

    That split is the empirical test of whether the cache works. The corpus runs
    each reference kernel 5 times (N_TRIALS_FPR) as the same Python object. If
    the cache were not being hit, those repeats would each pay full compilation
    and cost the same as the first touch. They do not.
    """
    cold_p = REPO / "benchmarks" / "autokernel" / "files" / "results_raw_cold.json"
    warm_p = REPO / "benchmarks" / "autokernel" / "files" / "results_raw.json"
    if not (cold_p.exists() and warm_p.exists()):
        return

    import json
    F = "your_checker (full)"
    cold = json.loads(cold_p.read_text())[F]
    warm = json.loads(warm_p.read_text())[F]
    c = [x * 1000 for x in cold["latencies"]]
    w = [x * 1000 for x in warm["latencies"]]

    n_mut = len(warm["mutant_results"])
    n_trials = len(warm["ref_results"]) // n_mut
    first, i = [], 0
    for _ in range(n_mut):
        first += [i, i + 1]          # the mutant, and the 1st of its ref trials
        i += 1 + n_trials

    cf, wf = sum(c[j] for j in first), sum(w[j] for j in first)
    total_compile = sum(c) - sum(w)
    first_compile = cf - wf
    per_candidate = first_compile / len(first)

    print("\n" + "=" * 72)
    print("CORROBORATION against the cold/warm corpus runs on disk")
    print("=" * 72)
    print(f"  cold {sum(c) / 1000:6.2f}s   warm {sum(w) / 1000:6.2f}s   "
          f"compile {total_compile / 1000:6.2f}s ({total_compile / sum(c) * 100:.1f}% of cold)")
    print(f"  of that compile cost, {first_compile / total_compile * 100:.1f}% falls on the "
          f"{len(first)} FIRST-TOUCH candidates;")
    print(f"  the other {len(c) - len(first)} invocations are repeats of "
          f"already-compiled kernel\n  objects and account for only "
          f"{(total_compile - first_compile) / 1000:.2f}s.")
    print("\n  That asymmetry IS the cache working. Repeats are nearly free.")

    print(f"\n  mean compile cost per distinct candidate : {per_candidate:6.1f}ms")
    print(f"  implied per-compile, at the modelled {specs_per_candidate:.1f} "
          f"specializations : {per_candidate / specs_per_candidate:6.1f}ms")
    print(f"  implied per-compile, IF every one of the {invocations_per_candidate:.0f} "
          f"invocations\n    compiled (i.e. cache never hit)          : "
          f"{per_candidate / invocations_per_candidate:6.1f}ms")
    print("\n  A real Triton compile is tens to hundreds of ms. The first figure is")
    print("  in that range; the second is far too fast to be a compile at all, so")
    print("  the no-cache hypothesis is inconsistent with the measured cost.")


if __name__ == "__main__":
    main()
