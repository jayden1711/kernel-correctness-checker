"""max_torch_ratio calibration probe (theory-audit flag #4).

Measures the partial-computation delegation ratio for every candidate-kernel
source population on disk (reference, cheating mutants, near-miss mutants,
experiments) exactly as check_partial_computation computes it (per wrapper
function AST, since inspect.getsource(candidate_fn) yields only the wrapper),
plus a panel of synthesized cheats spanning the delegation space. Pure AST,
no GPU, deterministic.
"""

import ast
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from verification.layer1_structural.ast_analysis import (  # noqa: E402
    _PartialComputationVisitor, check_partial_computation)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
POPULATIONS = {
    "reference": "TritonBench/reference",
    "cheating_mutants": "TritonBench/cheating",
    "near_miss": "TritonBench/near_miss",
    "experiments": "TritonBench/experiments",
}


def is_jit(fn_node):
    for dec in fn_node.decorator_list:
        s = ast.dump(dec)
        if "jit" in s or "autotune" in s:
            return True
    return False


def wrapper_ratios(path):
    """(func_name, n_torch, n_triton, ratio-or-None, verdict) per top-level
    non-@triton.jit function in the file — the checker sees exactly one such
    function's source when the candidate callable is that wrapper."""
    with open(path) as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [("<parse_error>", None, None, None, None)]
    out = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or is_jit(node):
            continue
        v = _PartialComputationVisitor()
        v.visit(node)
        nt, nl = len(v.torch_op_calls), len(v.triton_launches)
        total = nt + nl
        ratio = (nt / total) if total else None
        verdict = "skip(no ops)" if total == 0 else ("FLAG" if ratio > 0.5 else "pass")
        out.append((node.name, nt, nl, ratio, verdict))
    return out


SYNTH = {
    # canonical cheats and edge candidates, run through the real check
    "pure_delegation_listed_op": (
        "def softmax(x):\n    return torch.softmax(x, dim=-1)\n"),
    "pure_delegation_unlisted_op": (
        "def log_softmax(x):\n    import torch.nn.functional as F\n"
        "    return F.log_softmax(x, dim=-1)\n"),
    "pure_delegation_cross_entropy": (
        "def cross_entropy(logits, targets):\n"
        "    import torch.nn.functional as F\n"
        "    return F.cross_entropy(logits, targets)\n"),
    "pure_delegation_max_pool": (
        "def max_pool1d(x, k, s, p):\n    import torch.nn.functional as F\n"
        "    return F.max_pool1d(x, k, s, p)\n"),
    "delegation_plus_decoy_launch": (
        "def softmax(x):\n"
        "    decoy_kernel[(1,)](x, x.numel())\n"
        "    return torch.softmax(x, dim=-1)\n"),
    "delegation_unlisted_plus_decoy": (
        "def log_softmax(x):\n"
        "    import torch.nn.functional as F\n"
        "    decoy_kernel[(1,)](x, x.numel())\n"
        "    return F.log_softmax(x, dim=-1)\n"),
    "two_listed_ops_one_launch": (
        "def fused(x, w):\n"
        "    y = torch.matmul(x, w)\n"
        "    y = torch.relu(y)\n"
        "    epilogue_kernel[(1,)](y, y.numel())\n"
        "    return y\n"),
    "honest_fused_conv_epilogue": (
        "def fused_conv_bias_relu(x, w, b):\n"
        "    import torch.nn.functional as F\n"
        "    y = F.conv2d(x, w)\n"
        "    epilogue_kernel[(1,)](y, b, y.numel())\n"
        "    return y\n"),
    "honest_two_aux_ops": (
        "def op(x):\n"
        "    s = torch.sum(x)\n"
        "    m = torch.mean(x)\n"
        "    k_kernel[(1,)](x, s, m)\n"
        "    return x\n"),
}


def main():
    rows = []
    for pop, rel in POPULATIONS.items():
        base = os.path.join(ROOT, rel)
        for dirpath, _, files in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for fn in sorted(files):
                if not fn.endswith(".py") or fn == "__init__.py":
                    continue
                p = os.path.join(dirpath, fn)
                for (name, nt, nl, ratio, verdict) in wrapper_ratios(p):
                    rows.append({"population": pop,
                                 "file": os.path.relpath(p, ROOT),
                                 "func": name, "n_torch": nt, "n_triton": nl,
                                 "ratio": ratio, "verdict": verdict})

    synth_rows = []
    for name, src in SYNTH.items():
        ok, detail = check_partial_computation(src)
        synth_rows.append({"case": name, "passed": ok, "detail": detail})

    # summary
    from collections import Counter
    print("== on-disk populations (per wrapper function) ==")
    by = Counter()
    for r in rows:
        by[(r["population"], r["verdict"])] += 1
    for k, v in sorted(by.items()):
        print(f"  {k[0]:18s} {k[1]:14s} {v}")
    nz = [r for r in rows if r["ratio"] not in (None, 0.0)]
    print("\n  nonzero-ratio wrappers:")
    for r in sorted(nz, key=lambda r: -r["ratio"]):
        print(f"    {r['ratio']:.3f}  {r['file']}::{r['func']} "
              f"({r['n_torch']} torch / {r['n_triton']} launch) {r['verdict']}")
    print("\n== synthesized cheat/edge panel (real check_partial_computation) ==")
    for s in synth_rows:
        print(f"  {'pass' if s['passed'] else 'FLAG':4s}  {s['case']:36s} {s['detail'][:90]}")

    out = os.path.join(os.path.dirname(__file__), "..", "data", "ratio_measurements.json")
    with open(out, "w") as f:
        json.dump({"wrappers": rows, "synthetic": synth_rows}, f, indent=1)
    print(f"\nwrappers measured: {len(rows)}")


if __name__ == "__main__":
    main()
