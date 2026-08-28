"""
autokernel_faithful argument construction.

The bugs this file guards against are ARITY and DTYPE bugs, not numerical ones,
so a stubbed torch that records shape/dtype is a stronger test here than a real
CPU run would be. It does NOT validate numerics -- that needs a GPU, and no GPU
is available (see SESSION_HANDOFF.md section 0).

--------------------------------------------------------------------------
THE `check_*.py` FILENAME IS LOAD-BEARING. DO NOT RENAME THIS FILE.
--------------------------------------------------------------------------
This file replaces sys.modules["torch"] and sys.modules["numpy"] with stubs at
module scope. tests/pytest.ini sets `python_files = test_*.py`, so a file named
`check_*.py` is never collected by pytest -- which is the entire point.

Renaming this to the conventional `test_*.py` would let pytest collect it into
the same process as the real suite. tests/conftest.py imports the real torch at
module scope and every tests/verification/* test depends on it, so the stubs
would leak and corrupt those tests. The failure would look like unrelated tests
breaking, not like a naming problem.

Run it directly instead:
    python3 tests/instrumentation/check_autokernel_faithful_construction.py

Plain python3 -- no venv, no numpy, no torch, no pytest. See the README in this
directory for the full rationale.

--------------------------------------------------------------------------
SELF-VERIFICATION
--------------------------------------------------------------------------
Every run does two things:

  1. Runs the construction checks against the SHIPPED module and requires zero
     failures.
  2. Re-derives its own validity by mutating a copy of that module's source
     three ways -- shortened sweep, layernorm arity bug, matmul dtype leak --
     and requiring the checks to FAIL on each.

Step 2 exists because step 1 alone once passed while validating a stale copy of
the module under /tmp. A green run proved nothing. The two mutations in step 2
are exactly the bug classes item #1 found in the old AutoKernel baseline (see
benchmarks/autokernel/AUTOKERNEL_BASELINE_AUDIT.md section 3), so if this script
cannot detect them it is not testing anything worth testing.

A negative control that fails to trip is a FAILURE, and so is an anchor string
that no longer matches -- otherwise a refactor of autokernel_faithful.py would
silently disarm the self-check while leaving the run green.

Exit code 0 = everything passed. Non-zero = failures, listed on stdout.
"""
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODULE_SRC = REPO / "benchmarks" / "autokernel" / "files" / "autokernel_faithful.py"


# ── stubbed torch: records shape/dtype instead of allocating ──────────────────

class T:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape); self.dtype = dtype; self.device = "cpu"
    def __repr__(self):  return f"T{self.shape}/{self.dtype}"
    def __add__(self, o):  return self
    def __radd__(self, o): return self
    def __mul__(self, o):  return self
    def __rmul__(self, o): return self


def _norm(dims):
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        return tuple(dims[0])
    return tuple(dims)


def _install_stubs():
    torch = types.ModuleType("torch")
    torch.float32, torch.float16, torch.bfloat16 = "f32", "f16", "bf16"
    torch.randn = lambda *d, device=None, dtype=None, generator=None: T(_norm(d), dtype)
    torch.rand = lambda *d, device=None, dtype=None, generator=None: T(_norm(d), dtype)
    torch.randint = lambda lo, hi, size, device=None, generator=None: T(size, "i64")
    torch.full_like = lambda t, v: T(t.shape, t.dtype)
    torch.Generator = lambda device=None: types.SimpleNamespace(manual_seed=lambda s: None)
    torch.equal = lambda a, b: True
    torch.isfinite = lambda t: types.SimpleNamespace(all=lambda: True)
    torch.allclose = lambda *a, **k: True
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch
    sys.modules["numpy"] = types.ModuleType("numpy")
    return torch


torch = _install_stubs()


# ── module loading ────────────────────────────────────────────────────────────

def load_module(source: str, name: str):
    """Import `source` as a fresh module. Used for both the shipped file and
    each mutated variant, so the checks run against identical code paths."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, prefix=name + "_")
    tmp.write(source); tmp.close()
    spec = importlib.util.spec_from_file_location(name, tmp.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.unlink(tmp.name)
    return mod


# ── the checks ────────────────────────────────────────────────────────────────

FAMILIES = ["single", "layernorm", "instancenorm", "rmsnorm", "matmul", "attention",
            "groupnorm", "batchnorm", "cross_entropy", "pool1d", "pool2d", "pool3d"]

# Arity each family's TritonBench reference wrapper actually requires.
EXPECT_ARITY = {"single": 1, "layernorm": 3, "instancenorm": 3, "rmsnorm": 2,
                "matmul": 2, "attention": 3, "groupnorm": 4, "batchnorm": 5,
                "cross_entropy": 2, "pool1d": 4, "pool2d": 4, "pool3d": 4}

EXPECTED_SWEEP_LEN = 8   # published spec: 8-10 configs per family

OPS_29 = ["argmax", "argmin", "avg_pool1d", "avg_pool2d", "avg_pool3d", "batchnorm",
          "causal_flash_attention", "cross_entropy", "flash_attention", "frobenius_norm",
          "gelu", "groupnorm", "instancenorm", "l1norm", "l2norm", "layernorm",
          "log_softmax", "matmul", "max_pool1d", "max_pool2d", "max_pool3d",
          "max_reduction", "mean_reduction", "min_reduction", "rmsnorm",
          "scaled_dot_product_attention", "softmax", "sum_reduction", "swish"]


def run_construction_checks(akf, verbose=True):
    """Returns (n_failures, stage3_coverage, log_lines). Shared by the shipped
    module and every mutant, so a negative control exercises the same code."""
    fails, log = 0, []

    def note(msg):
        log.append(msg)
        if verbose:
            print(msg)

    for fam in FAMILIES:
        try:
            sw = akf._sweep_shapes(fam)
            edge, dropped = akf._edge_case_shapes(fam)
        except Exception as e:
            note(f"  !! {fam}: sweep/edge lookup raised {type(e).__name__}: {e}")
            fails += 1
            continue

        if len(sw) != EXPECTED_SWEEP_LEN:
            note(f"  !! {fam}: sweep has {len(sw)}, expected {EXPECTED_SWEEP_LEN}")
            fails += 1

        for shape in sw + edge:
            for dt in (torch.float32, torch.float16, torch.bfloat16):
                try:
                    args = akf._build_args(fam, shape, dt, "cpu", None)
                except Exception as e:
                    note(f"  !! {fam} {shape}: _build_args raised {type(e).__name__}: {e}")
                    fails += 1
                    continue
                if args is None:
                    note(f"  !! {fam} {shape}: no builder"); fails += 1; continue
                if len(args) != EXPECT_ARITY[fam]:
                    note(f"  !! {fam} {shape}: arity {len(args)} != {EXPECT_ARITY[fam]}")
                    fails += 1
                for a in args:
                    # int64 target vectors (cross_entropy) legitimately differ
                    if isinstance(a, T) and a.dtype not in (dt, "i64"):
                        note(f"  !! {fam} {shape}: dtype leak {a.dtype} != {dt}")
                        fails += 1
        note(f"  {fam:14s} sweep={len(sw)} edge={len(edge)} dropped={dropped} "
             f"arity={EXPECT_ARITY[fam]} OK")

    x = T((8, 16), torch.float32)
    coverage = [o for o in OPS_29 if akf._stability_primary(o, x, None) is not None]
    return fails, coverage, log


# ── negative controls ─────────────────────────────────────────────────────────
#
# (label, anchor, replacement, what the checks must notice).
# Anchors are exact source strings. If one stops matching -- e.g. someone
# refactors _build_args -- that is a hard failure, not a skip: a silently
# disarmed self-check is the exact failure mode this section exists to prevent.

NEGATIVE_CONTROLS = [
    ("shortened sweep (8 -> 3 configs)",
     'return [(1, 128), (8, 64), (32, 128), (64, 128),\n'
     '                (17, 333), (64, 1023), (128, 512), (256, 512)]',
     'return [(1, 128), (8, 64), (32, 128)]',
     "sweep length"),

    ("layernorm arity bug (drops gamma/beta)",
     'if family == "layernorm":\n        n_cols = shape[-1]\n'
     '        return (t(*shape), t(n_cols), t(n_cols))',
     'if family == "layernorm":\n        n_cols = shape[-1]\n        return (t(*shape),)',
     "arity -- the same bug class as AutoKernel baseline Bug A"),

    ("dtype leak (builder ignores requested dtype)",
     '    def t(*dims):\n        return torch.randn(*dims, device=device, dtype=dtype, generator=gen)',
     '    def t(*dims):\n        return torch.randn(*dims, device=device, dtype=torch.float32, generator=gen)',
     "dtype -- the same bug class as AutoKernel baseline Bug B"),
]


def run_negative_controls(source):
    """Each mutation MUST make run_construction_checks report >=1 failure."""
    print("\n=== self-verification: negative controls ===")
    print("    (each mutation must be DETECTED; a control that fails to trip is a FAILURE)")
    bad = 0
    for i, (label, anchor, replacement, what) in enumerate(NEGATIVE_CONTROLS):
        if anchor not in source:
            print(f"  !! CONTROL DISARMED: {label}\n"
                  f"     anchor no longer matches {MODULE_SRC.name} -- the source was "
                  f"refactored and this self-check is no longer testing anything.\n"
                  f"     Update the anchor in NEGATIVE_CONTROLS.")
            bad += 1
            continue
        mutant = load_module(source.replace(anchor, replacement), f"_akf_mutant_{i}")
        fails, _, log = run_construction_checks(mutant, verbose=False)
        if fails > 0:
            sample = next((l.strip() for l in log if l.strip().startswith("!!")), "")
            print(f"  DETECTED  {label:<44} ({fails} failures)  [{what}]")
            if sample:
                print(f"            e.g. {sample}")
        else:
            print(f"  !! MISSED  {label:<44} -- checks did not notice [{what}]")
            bad += 1
    return bad


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not MODULE_SRC.exists():
        print(f"FAIL: {MODULE_SRC} not found"); return 1
    source = MODULE_SRC.read_text()

    akf = load_module(source, "_akf_shipped")
    # Printed every run: if this is ever not the repo's own file, the run is
    # worthless. It previously pointed at a stale scratch copy under /tmp and
    # passed while validating the wrong module.
    print(f"module under test: {MODULE_SRC}")

    print("\n=== stage-2 sweep + stage-5 edge: arity & shape construction ===")
    fails, coverage, _ = run_construction_checks(akf)
    print(f"\nconstruction failures: {fails}")

    print("\n=== stage-3 probe coverage over the 29-op corpus ===")
    print(f"  covered:   {len(coverage)} {sorted(coverage)}")
    print(f"  uncovered: {len(OPS_29) - len(coverage)} "
          f"(the paper names no probe class for these)")
    print(f"\n  tolerance: {akf.tolerance_sensitivity()}")

    bad_controls = run_negative_controls(source)

    print()
    problems = []
    if fails:
        problems.append(f"{fails} argument-construction failure(s) in the shipped module")
    if not coverage:
        problems.append("stage-3 probe coverage collapsed to zero operators")
    if bad_controls:
        problems.append(f"{bad_controls} negative control(s) disarmed or not detected "
                        f"-- this script is no longer a reliable guard")
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1

    print(f"ALL PASS  ({len(FAMILIES)} families checked, "
          f"{len(NEGATIVE_CONTROLS)}/{len(NEGATIVE_CONTROLS)} negative controls detected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
