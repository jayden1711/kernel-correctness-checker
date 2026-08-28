"""
Adjudication of the 12 outstanding softmax reference-suspect records
(attention_mask_fix_2026-08-27/FINDINGS.md §4) at the NORM_ADJUDICATION
standard: materialize each proposal's input exactly (the run's own
materializer), emulate the reference kernel's arithmetic in fp32, run the
SHIPPED check function, and then run the IDEAL MATH (float64 softmax)
through the same check. If the ideal math also fails, the record is a
check-domain false alarm; if it passes, the reference is implicated.

The derived validity domain being applied (FINDINGS.md §1): the positivity
check records column j unwritten iff EVERY row's fp value
exp(l_ij - m_i)/L_i rounds to 0 -- guaranteed once m_i - l_ij exceeds
B32 = 103.97 (fp32 subnormal boundary; 87.34 if the GPU flushes
subnormals) in every row, and B64 = 744.4 in float64. All 12 records carry
full-column-height patches at +1e3/+1e4 on zeros/randn fills, so every
unpatched column sits ~10x (fp32) / ~1.3x (fp64) beyond the boundary in
EVERY row -- the check must fail on ANY correct implementation, in both
precisions. This probe verifies that record by record instead of assuming
it.

Records with fill=randn have no recorded seed (same situation as the norm
adjudication); those are run over 10 seeds and unanimity is required.

Run:  .venv/bin/python softmax_positivity_adjudication.py
"""
import json
import math
import os
import sqlite3
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from verification.adversarial_search.materializer import materialize_proposal
from verification.adversarial_search.schemas import InputProposal
from verification.layer1_structural.tile_coverage import check_all_tiles_visited

DB = os.path.join(ROOT, "adversarial_results", "search_history.db")
B32_SUBNORMAL = 150 * math.log(2)          # 103.97
B32_FTZ = 126 * math.log(2)                # 87.34
B64 = 1075 * math.log(2)                   # 745.1


def softmax_f32(x):
    """The reference kernel's arithmetic, fp32: whole-row block, -inf pads
    (pads only affect max/sum as -inf -> exp 0; at these shapes BLOCK access
    pattern doesn't change the math), max-subtract, exp, sum, divide."""
    x = x.float()
    m = x.max(dim=-1, keepdim=True).values
    num = torch.exp(x - m)
    return num / num.sum(dim=-1, keepdim=True)


def softmax_f64(x):
    x = x.double()
    m = x.max(dim=-1, keepdim=True).values
    num = torch.exp(x - m)
    return num / num.sum(dim=-1, keepdim=True)


def load_records():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT v.proposal_id, v.verdict_json, p.proposal_json "
        "FROM verdicts v LEFT JOIN proposals p ON v.proposal_id = p.proposal_id "
        "WHERE v.operator = 'softmax'").fetchall()
    con.close()
    out = []
    for r in rows:
        vj = json.loads(r["verdict_json"])
        if vj.get("reference_passed", True):
            continue
        if "positivity" not in vj.get("failure_summary", ""):
            continue
        out.append((r["proposal_id"], json.loads(r["proposal_json"]),
                    vj.get("failure_summary", "")))
    return out


def run_one(pj, seed):
    torch.manual_seed(seed)
    prop = InputProposal.from_dict(dict(pj))
    tensors = materialize_proposal(prop, device="cpu")
    x = tensors["x"]
    checks = {}
    for name, f in (("faithful_f32", softmax_f32), ("ideal_f64", softmax_f64)):
        passed, detail = check_all_tiles_visited(lambda t, f=f: f(t), None, x)
        # NOTE: the check calls kernel_fn(x) itself; pass the emulation.
        checks[name] = (passed, detail)
    # margin: worst-case unpatched-column deficit, min over rows of m_i - l_ij,
    # maximized over unwritten columns of the f32 output
    y = softmax_f32(x)
    m = x.float().max(dim=-1).values
    deficit = (m.unsqueeze(1) - x.float())          # m_i - l_ij
    col_min_deficit = deficit.min(dim=0).values     # per column: best row
    unwritten = ~(y > 0).any(dim=0)
    margin = float(col_min_deficit[unwritten].min()) if unwritten.any() else None
    return checks, int(unwritten.sum()), margin


def main():
    records = load_records()
    print(f"{len(records)} positivity-failing softmax reference records\n")
    verdicts = []
    for pid, pj, summary in records:
        fill = pj["tensors"]["x"]["fill"]
        seeds = range(10) if fill == "randn" else [0]
        f32_fail = f64_fail = 0
        margins, ncols = [], []
        for s in seeds:
            checks, n_unwritten, margin = run_one(pj, s)
            f32_fail += 0 if checks["faithful_f32"][0] else 1
            f64_fail += 0 if checks["ideal_f64"][0] else 1
            ncols.append(n_unwritten)
            if margin is not None:
                margins.append(margin)
        n = len(list(seeds))
        patch = pj["tensors"]["x"]["patches"][0]
        verdict = ("CHECK-DOMAIN FALSE ALARM"
                   if f32_fail == n and f64_fail == n else
                   "REFERENCE-IMPLICATED" if f32_fail == n else "NO-REPRO")
        verdicts.append(verdict)
        shape = pj["tensors"]["x"]["shape"]
        print(f"{pid[:8]}  {str(shape):12s} {fill:5s} patch {patch['indices']:>12s}"
              f"={patch['value']:g}  faithful fails {f32_fail}/{n}, ideal "
              f"fails {f64_fail}/{n}; unwritten cols {min(ncols)}-{max(ncols)}; "
              f"min deficit {min(margins):.0f} (B32={B32_SUBNORMAL:.0f}, "
              f"B64={B64:.0f})  -> {verdict}")

    print(f"\nSUMMARY: {verdicts.count('CHECK-DOMAIN FALSE ALARM')}/12 "
          f"check-domain false alarms, "
          f"{verdicts.count('REFERENCE-IMPLICATED')} reference-implicated, "
          f"{verdicts.count('NO-REPRO')} no-repro")


if __name__ == "__main__":
    main()
