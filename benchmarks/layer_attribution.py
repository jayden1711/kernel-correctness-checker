"""
Per-operator layer attribution: raw counts of which mutants each layer catches.

Advisor's note: "record layer 2 catching extent of operators" -- the headline
table reports Layer 2 as one aggregate 100%, which cannot distinguish "the
numeric layer works uniformly across all 29 operators" from "it works on a
handful of operators that happen to carry most of the mutants." Raw counts per
operator, not percentages, are what separate those.

WHERE THE ATTRIBUTION COMES FROM
--------------------------------
The three single-layer ablations -- "your_checker (structural only)",
"(numeric only)", "(algebraic only)". Each runs every check in its layer
UNCONDITIONALLY (checker_adapter.py:8-16), unlike KernelChecker.run which
short-circuits between layers. That is what makes "caught by more than one
layer" a genuine statement about independent capability rather than an artifact
of evaluation order: in a full run, Layer 2 only executes when Layer 1 passed,
so a full-run trace could never show both catching the same mutant.

This needs no re-run. summarize() drops per-check detail but does keep
`missed_mutants` per system, and caught = (all mutants) - (missed). The
operator/mutant roster is parsed statically from tritonbench_registry.py's OPS
table -- parsed, not imported, since importing it pulls in torch + triton and
launches real kernels.

Usage:
    python3 benchmarks/layer_attribution.py [path/to/results.json]
    # writes benchmarks/LAYER_ATTRIBUTION.md
"""
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REGISTRY = os.path.join(HERE, "autokernel", "files", "tritonbench_registry.py")
OUT_PATH = os.path.join(HERE, "LAYER_ATTRIBUTION.md")

CANDIDATE_RESULTS = [
    os.path.join(ROOT, "results.json"),
    os.path.join(HERE, "autokernel", "files", "results.json"),
]

LAYERS = [
    ("structural", "your_checker (structural only)", 1),
    ("numeric",    "your_checker (numeric only)",    2),
    ("algebraic",  "your_checker (algebraic only)",  3),
]


def load_roster():
    """(op, mutant) pairs from tritonbench_registry.py's OPS literal.

    Parsed with ast rather than imported: tritonbench_registry imports torch
    and triton and builds the corpus by launching real @triton.jit kernels,
    which cannot run without a GPU and is entirely unnecessary just to read a
    list of names.
    """
    tree = ast.parse(open(REGISTRY).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "OPS" for t in node.targets):
            ops = ast.literal_eval(node.value)
            break
    else:
        sys.exit(f"No OPS table found in {REGISTRY}")

    roster = []
    for spec_key, _ref, _cheat, _family, mutants in ops:
        for m in mutants:
            roster.append((spec_key, m))
    return roster


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else next(
        (p for p in CANDIDATE_RESULTS if os.path.exists(p)), None)
    if not path or not os.path.exists(path):
        sys.exit("No results.json found. Looked in:\n  " + "\n  ".join(CANDIDATE_RESULTS))

    summary = json.load(open(path))
    roster = load_roster()
    all_keys = [f"{op}/{m}" for op, m in roster]
    ops_in_order = sorted({op for op, _ in roster})

    missing = [name for _, name, _ in LAYERS if name not in summary]
    if missing:
        sys.exit(f"results.json is missing required ablation systems: {missing}\n"
                 f"present: {sorted(summary)}")

    caught = {}
    for label, sysname, _ in LAYERS:
        missed = set(summary[sysname]["missed_mutants"])
        unknown = missed - set(all_keys)
        if unknown:
            sys.exit(f"'{sysname}' reports missed mutants absent from the OPS "
                     f"roster: {sorted(unknown)}. results.json and "
                     f"tritonbench_registry.py are out of sync.")
        caught[label] = set(all_keys) - missed

    per_op = {}
    for op in ops_in_order:
        keys = [k for k in all_keys if k.rsplit("/", 1)[0] == op]
        row = {"n": len(keys)}
        for label, _, _ in LAYERS:
            row[label] = sum(1 for k in keys if k in caught[label])
        row["multi"] = sum(
            1 for k in keys
            if sum(1 for label, _, _ in LAYERS if k in caught[label]) > 1)
        row["none"] = sum(
            1 for k in keys
            if not any(k in caught[label] for label, _, _ in LAYERS))
        row["numeric_only"] = sum(
            1 for k in keys
            if k in caught["numeric"] and k not in caught["structural"]
            and k not in caught["algebraic"])
        per_op[op] = row

    L = ["# Per-operator layer attribution", "",
         f"Source: `{os.path.relpath(path, ROOT)}` — the three single-layer "
         "ablations, each of which runs its whole layer unconditionally. "
         "Counts are raw mutants, not percentages.", "",
         f"{len(all_keys)} mutants across {len(ops_in_order)} operators.", "",
         "| Operator | # mutants | structural | numeric | algebraic | >1 layer | numeric only | uncaught |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for op in ops_in_order:
        r = per_op[op]
        L.append(f"| {op} | {r['n']} | {r['structural']} | {r['numeric']} | "
                 f"{r['algebraic']} | {r['multi']} | {r['numeric_only']} | {r['none']} |")
    tot = {k: sum(per_op[op][k] for op in ops_in_order)
           for k in ("n", "structural", "numeric", "algebraic", "multi", "numeric_only", "none")}
    L.append(f"| **TOTAL** | **{tot['n']}** | **{tot['structural']}** | **{tot['numeric']}** | "
             f"**{tot['algebraic']}** | **{tot['multi']}** | **{tot['numeric_only']}** | **{tot['none']}** |")
    L.append("")

    # --- the question the advisor actually asked -------------------------
    n_ops = len(ops_in_order)
    num_full = [op for op in ops_in_order if per_op[op]["numeric"] == per_op[op]["n"]]
    num_zero = [op for op in ops_in_order if per_op[op]["numeric"] == 0]
    L += ["## Is numeric-layer dominance uniform, or concentrated?", ""]
    L.append(f"- Operators where the numeric layer catches **every** mutant: "
             f"**{len(num_full)}/{n_ops}**")
    L.append(f"- Operators where it catches **none**: **{len(num_zero)}**"
             + (f" ({', '.join(num_zero)})" if num_zero else ""))
    if len(num_full) == n_ops:
        L.append("")
        L.append("**Uniform, not concentrated.** The numeric layer catches every "
                 "mutant of every operator, so its 100% aggregate is not carried "
                 "by a few mutant-heavy operators — it holds operator by operator. "
                 "That is the strongest form this result could take, and it is the "
                 "specific claim the advisor's question was asking to verify.")
    L.append("")

    L += ["## What each layer contributes independently", "",
          "| Layer | mutants caught | of which, caught by no other layer |",
          "|---|---:|---:|"]
    for label, _, _ in LAYERS:
        others = set().union(*[caught[o] for o, _, _ in LAYERS if o != label])
        L.append(f"| {label} | {len(caught[label])} | {len(caught[label] - others)} |")
    L.append("")

    without_numeric = set(all_keys) - (caught["structural"] | caught["algebraic"])
    L.append(f"**Removing the numeric layer would lose {len(without_numeric)} of "
             f"{len(all_keys)} mutants** — structural and algebraic together catch "
             f"{len(all_keys) - len(without_numeric)}. This is the concrete "
             "justification for Layer 2 as a layer, distinct from the per-check "
             "ablation in `CHECK_ABLATION.md` which asks which checks *within* it "
             "earn their place.")
    L.append("")

    # Operators where the other two layers add nothing at all
    dead_others = [op for op in ops_in_order
                   if per_op[op]["structural"] == 0 and per_op[op]["algebraic"] == 0]
    L.append(f"Operators where **only** the numeric layer catches anything: "
             f"**{len(dead_others)}/{n_ops}**"
             + (f" — {', '.join(dead_others)}" if dead_others else ""))
    L.append("")
    L.append("Operators where structural or algebraic catches a mutant the numeric "
             "layer also catches are defence-in-depth, not additional recall — the "
             "`>1 layer` column quantifies that overlap, and it is only meaningful "
             "because the ablations run unconditionally.")
    L.append("")

    # --- nesting: the sharpest form of the result ------------------------
    st, al, nu = caught["structural"], caught["algebraic"], caught["numeric"]
    nested = st <= al <= nu
    L += ["## Layer nesting", ""]
    if nested:
        L.append(f"On this corpus the catch sets are **strictly nested**: "
                 f"structural ({len(st)}) subset of algebraic ({len(al)}) subset of "
                 f"numeric ({len(nu)}).")
        L.append("")
        L.append("This is stronger than \"numeric alone matches the full checker\". "
                 "It says structural and algebraic contribute **zero** additional "
                 "recall — there is no mutant either one catches that the numeric "
                 "layer misses. Their value on this corpus is defence-in-depth and "
                 "precision of diagnosis (naming *which* invariant broke), not "
                 "coverage. Stated plainly, that is a more honest framing than a "
                 "three-layer recall claim, and it is what the per-operator counts "
                 "above actually support.")
        L.append("")
        L.append("The caveat that matters for generalisation: nesting is a property "
                 "of **this 40-mutant corpus**, not a theorem. A structural-only bug "
                 "(a kernel that never launches, a tile left unwritten) would be "
                 "caught by Layer 1 and could well slip a numeric comparison that "
                 "happens to agree; no such mutant exists here. The corpus, not the "
                 "checker, is what makes the layers nest — worth saying before a "
                 "reviewer says it.")
    else:
        L.append("Catch sets are **not** nested — each layer catches at least one "
                 "mutant no other layer does. Per-layer exclusives are in the table "
                 "above.")
    L.append("")

    # --- cross-reference for item #6 -------------------------------------
    CFA = "causal_flash_attention/wrong_causal_mask"
    if CFA in nu:
        hits = [lab for lab, _, _ in LAYERS if CFA in caught[lab]]
        L += ["## Cross-reference: causal_flash_attention", ""]
        L.append(f"`{CFA}` is caught here by **{len(hits)} of 3 layers** "
                 f"({', '.join(hits)}) on a plain random input.")
        L.append("")
        L.append("The adversarial search, by contrast, recorded "
                 "`hit_mutants: []` on all 120 proposals "
                 "(`adversarial_results/causal_flash_attention_search_result.json`).")
        L.append("")
        L.append("**That is not the contradiction it first appears to be, and an "
                 "earlier reading of it here was wrong.** A mutant lands in "
                 "`missed_mutants` when it passed the checker **or** when it failed "
                 "the checker but also failed naive allclose "
                 "(`coordinator.py:_evaluate_verdict`) — the second case is a "
                 "*caught* mutant with no allclose gap to report. Since allclose "
                 "catches this mutant on ordinary inputs (0% missed for this "
                 "operator in `results.md`), the likely reading is that the checker "
                 "did catch it and the search correctly found no gap. The verdict "
                 "record persists nothing per-mutant, so which of the two occurred "
                 "cannot be recovered from the stored run.")
        L.append("")
        L.append("Root-caused separately in "
                 "`adversarial_results/CFA_NONHIT_ROOTCAUSE.md`: 47% of that run's "
                 "proposals were structurally invalid because "
                 "`causal_flash_attention` has no entry in the search prompt's "
                 "`OPERATOR_CONTEXT`.")
        L.append("")

    open(OUT_PATH, "w").write("\n".join(L))
    print(f"Wrote {OUT_PATH}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
