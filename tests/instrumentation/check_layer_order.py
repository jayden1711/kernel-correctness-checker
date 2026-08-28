"""
Layer ordering: structural -> algebraic -> numeric, and why it is safe.

--------------------------------------------------------------------------
THE `check_*.py` FILENAME IS LOAD-BEARING. DO NOT RENAME THIS FILE.
--------------------------------------------------------------------------
Same reason as the other files in this directory: tests/pytest.ini sets
`python_files = test_*.py`, so `check_*.py` is never collected by pytest. See
the README here.

WHAT THIS GUARDS
----------------
On 2026-08-20 KernelChecker.run was reordered so the expensive numeric layer
runs LAST, reached only when structural and algebraic have both failed to catch
the bug. Warm p50 per layer: structural 3.97ms, algebraic 1.17ms, numeric
15.71ms.

Reordering a short-circuiting pipeline is only safe because the catch sets are
NESTED: structural (4 of 40 mutants) subset of algebraic (18) subset of numeric
(40). Any mutant algebraic catches, numeric would also have caught -- so moving
numeric later can change WHICH layer reports a catch, never WHETHER there is
one.

That containment is an empirical property of the current corpus, not a
theorem. If a future check makes algebraic catch something numeric does not,
the nesting breaks and the reorder stops being verdict-preserving. This file
fails loudly if that happens, which is the entire point: the argument for the
reorder is only as good as the containment it rests on.

Plain python3 -- reads source text and results_raw.json, imports nothing heavy.
Exit code 0 = all assertions passed.
"""
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHECKER = REPO / "verification" / "checker.py"
RAW = REPO / "benchmarks" / "autokernel" / "files" / "results_raw.json"

fails = []


def ck(label, cond, ctx=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   [{ctx}]" if not cond and ctx else ""))
    if not cond:
        fails.append(label)


# ── 1. execution order in checker.py ─────────────────────────────────────────
print("\n[1] KernelChecker.run executes structural -> algebraic -> numeric")
src = CHECKER.read_text()
run_body = src[src.index("def run("):src.index("def _run_check(")]

# Order of first appearance of each layer's checks, by the layer arg actually
# passed to _run_check -- not by comment text, which can drift from the code.
first = {}
for m in re.finditer(r"_run_check\((\d),", run_body):
    first.setdefault(m.group(1), m.start())
ck("all three layers present in run()", set(first) == {"1", "2", "3"}, f"got {sorted(first)}")
if set(first) == {"1", "2", "3"}:
    ck("Layer 1 checks come first", first["1"] < first["2"] < first["3"],
       f"offsets {first}")

# The labels must match the semantics, or the numbers are decoration.
alg_pos = run_body.index("spec.algebraic_properties")
num_pos = run_body.index("check_perturbation_tolerance")
ck("algebraic block precedes numeric block", alg_pos < num_pos,
   f"algebraic@{alg_pos} numeric@{num_pos}")
# Search FORWARD from each block's marker: the _run_check call sits after the
# loop header / check name, not before it. (Searching backward finds the
# preceding layer's call and reports a false failure -- which this test did on
# its first run.)
alg_call = run_body.index("_run_check(", run_body.index("spec.algebraic_properties"))
ck("algebraic checks are labelled Layer 2",
   run_body[alg_call:alg_call + 14].startswith("_run_check(2,"),
   run_body[alg_call:alg_call + 14])
num_call = run_body.index("_run_check(", run_body.index("# Layer 3: Numeric Oracle"))
ck("numeric checks are labelled Layer 3",
   run_body[num_call:num_call + 14].startswith("_run_check(3,"),
   run_body[num_call:num_call + 14])

# Short-circuit gates: one after each of the first two layers, or the reorder
# buys nothing.
ck("three short-circuit gates remain",
   run_body.count("if any(not r.passed for r in results):") == 3,
   f"found {run_body.count('if any(not r.passed for r in results):')}")

# ── 2. the containment the reorder depends on ────────────────────────────────
print("\n[2] catch-set containment: structural subset algebraic subset numeric")
if not RAW.exists():
    ck("results_raw.json present", False, str(RAW))
else:
    raw = json.loads(RAW.read_text())

    def caught(system):
        return {f"{r['op']}/{r['mutant']}"
                for r in raw[system]["mutant_results"] if r["caught"]}

    S = caught("your_checker (structural only)")
    A = caught("your_checker (algebraic only)")
    N = caught("your_checker (numeric only)")
    print(f"        structural={len(S)}  algebraic={len(A)}  numeric={len(N)}")
    ck("structural subset of algebraic", S <= A, f"extra: {sorted(S - A)}")
    ck("algebraic subset of numeric", A <= N, f"extra: {sorted(A - N)}")
    # This is the load-bearing one: if algebraic ever catches something numeric
    # does not, moving numeric last changes a verdict.
    ck("nothing is caught ONLY by a layer that now runs before numeric",
       (S | A) <= N, f"would change verdict: {sorted((S | A) - N)}")
    ck("full checker still catches everything numeric does",
       caught("your_checker (full)") == N,
       f"full={len(caught('your_checker (full)'))} numeric={len(N)}")

# ── 3. the convention marker ─────────────────────────────────────────────────
print("\n[3] layer_convention marker is written into results")
if RAW.exists():
    raw = json.loads(RAW.read_text())
    any_sys = next(iter(raw.values()))
    has = "layer_convention" in any_sys
    # Absence is EXPECTED on files written before the marker existed -- that is
    # exactly the case the marker exists to disambiguate, so this reports
    # rather than fails.
    print(f"        this file: {any_sys.get('layer_convention', 'ABSENT (pre-2026-08-20 convention)')}")
    ck("marker is either absent (old file) or names the new convention",
       (not has) or any_sys["layer_convention"] == "structural_algebraic_numeric_v2",
       any_sys.get("layer_convention"))

# ── 4. perturbation batching allowlist ───────────────────────────────────────
print("\n[4] batch_samples allowlist gates the operators that must not batch")
# Source-level, so this runs without torch. The batching allowlist decides
# whether 20 perturbation samples are stacked into one kernel call. Getting it
# wrong on a global-reduction operator does not raise -- it silently loosens
# adaptive_tol (measured on a frobenius_norm stand-in: 0.001218 -> 0.778163,
# 639x looser), which would pass almost any wrong kernel.
base = (REPO / "verification" / "specs" / "base_spec.py").read_text()
frob = (REPO / "verification" / "specs" / "frobenius_norm.py").read_text()

# MUST be a property, not a dataclass field. As a field it is silently
# unoverridable in any spec subclass that is not itself @dataclass-decorated:
# the inherited __init__ assigns the parent default to the instance and shadows
# the subclass's class attribute. frobenius_norm hit exactly that -- class attr
# False, every instance True.
ck("batch_samples is a property on KernelSpec (not a shadowable field)",
   "@property\n    def batch_samples" in base,
   "found a bare `batch_samples: bool =` field instead")
ck("frobenius_norm overrides batch_samples as a property",
   "@property\n    def batch_samples" in frob)
ck("frobenius_norm's override returns False",
   re.search(r"def batch_samples[^\n]*\n\s*return False", frob) is not None)
ck("no spec declares batch_samples as a bare dataclass field",
   not re.search(r"^\s*batch_samples\s*:\s*bool\s*=", base + frob, re.M),
   "a field form reintroduces the shadowing trap")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
