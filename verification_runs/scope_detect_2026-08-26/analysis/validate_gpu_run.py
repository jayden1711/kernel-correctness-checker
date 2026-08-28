"""
Score a GPU arm from scopedet.sh against the expectation set.

Run as:  python3 validate_gpu_run.py <A_no_detector.json> <B_detector.json>

Answers the three questions the validation step exists to settle:

  1. Does enabling the detector change any verdict? It must not -- the field is
     attached after `passed` is computed. A single difference is a wiring bug.
  2. Does it fire on everything GPU_NATIVE.md Section 4 marked out of scope,
     plus argmax/argmin?
  3. Does it stay silent on the other 24 operators -- and with how much margin?
     Margin is the point: "silent" and "silent by 1%" are different results and
     only the RECORD_ALL arm can tell them apart.
"""
import json, sys, gzip, collections, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from banked_fixture import ATTENTION_VARIANTS, STRUCTURALLY_EXCLUDED

EXPECT_FLAGGED = {(op, "adversarial_" + v) for op, v, *_r in ATTENTION_VARIANTS
                  if not _r[4]}
EXPECT_SILENT = {(op, "adversarial_" + v) for op, v, *_r in ATTENTION_VARIANTS
                 if _r[4] and v != "primary"}


def load(p):
    o = gzip.open if p.endswith(".gz") else open
    return json.load(o(p))


def records(d):
    for e in d["entries"]:
        rs = [("mutant", r) for r in e["mutant"]["records"]]
        rs += [("ref", r) for rf in e.get("refs", []) for r in rf["records"]]
        for kind, r in rs:
            yield e["op"], e["mutant"]["name"], kind, r


def main(a_path, b_path):
    A, B = load(a_path), load(b_path)

    # --- 1. verdicts unchanged --------------------------------------------
    va = [(op, mu, k, r["name"], r["outcome"] if "outcome" in r else r.get("passed"))
          for op, mu, k, r in records(A)]
    vb = [(op, mu, k, r["name"], r["outcome"] if "outcome" in r else r.get("passed"))
          for op, mu, k, r in records(B)]
    same = va == vb
    print(f"1. VERDICTS IDENTICAL WITH DETECTOR ON : "
          f"{'YES' if same else 'NO -- WIRING BUG'}")
    print(f"   summary A {A['summary']}")
    print(f"   summary B {B['summary']}")
    if not same:
        for x, y in zip(va, vb):
            if x != y:
                print(f"   first divergence: {x} vs {y}")
                break

    # --- 2/3. what fired ---------------------------------------------------
    # Read from `scope_flags`, the NEW field, so the promotion in
    # KernelChecker._run_check is what gets exercised. `subchecks` is counted
    # separately only to tell "the promotion is broken" apart from "the
    # detector never fired" -- those look identical from scope_flags alone.
    fired, silent = collections.defaultdict(list), collections.defaultdict(list)
    n_promoted = n_in_subchecks = 0
    for op, mu, kind, r in records(B):
        n_in_subchecks += sum(1 for sc in (r.get("subchecks") or [])
                              if isinstance(sc, dict)
                              and sc.get("kind") == "scope_divergence")
        for sc in (r.get("scope_flags") or []):
            if sc.get("kind") != "scope_divergence":
                continue
            n_promoted += 1
            key = (op, r["name"])
            (silent if sc.get("in_scope") else fired)[key].append(sc)
    print()
    print(f"   scope records in `scope_flags` (promoted) : {n_promoted}")
    print(f"   scope records in `subchecks`   (source)   : {n_in_subchecks}")
    if n_promoted != n_in_subchecks:
        print("   *** PROMOTION MISMATCH -- _run_check is dropping records ***")

    print()
    print("2. FIRED")
    print(f"   {'operator/check':<58}{'reasons':<34}{'defect%':>9}{'s/ulp':>11}")
    for (op, chk), recs in sorted(fired.items()):
        r0 = recs[0]
        rs = ",".join(x["reason"] for x in r0["reasons"] if x["severity"] != "advisory")
        d = f"{r0['defect_pct']:.1f}" if r0.get("defect_pct") is not None else "-"
        u = f"{r0['sulp_median']:.2f}" if r0.get("sulp_median") is not None else "-"
        print(f"   {op + '/' + chk:<58}{rs:<34}{d:>9}{u:>11}")

    got = {k for k in fired}
    missing = {k for k in EXPECT_FLAGGED if k not in got}
    unexpected = {k for k in EXPECT_SILENT if k in got}
    struct_ops = {op for (op, _c) in got
                  if any(x["reason"] == "structural_exclusion"
                         for rec in fired[(op, _c)] for x in rec["reasons"])}
    print()
    print(f"   expected-out-of-scope attention variants caught : "
          f"{len(EXPECT_FLAGGED) - len(missing)}/{len(EXPECT_FLAGGED)}"
          f"{'' if not missing else '   MISSING ' + str(sorted(missing))}")
    print(f"   argmax/argmin flagged structurally              : "
          f"{sorted(struct_ops & set(STRUCTURALLY_EXCLUDED))}")
    print(f"   in-scope attention variants wrongly flagged     : "
          f"{sorted(unexpected) if unexpected else 'none'}")

    # --- 3. margins on the silent set --------------------------------------
    print()
    print("3. SILENT -- margin to each threshold (this is the false-alarm headroom)")
    per_op = collections.defaultdict(lambda: {"d": [], "u": []})
    for (op, chk), recs in silent.items():
        for r in recs:
            if r.get("defect_pct") is not None:
                per_op[op]["d"].append(r["defect_pct"])
            if r.get("sulp_median") is not None:
                per_op[op]["u"].append(r["sulp_median"])
    print(f"   {'operator':<34}{'worst defect%':>15}{'x margin':>10}"
          f"{'min s/ulp':>13}{'x margin':>10}")
    for op in sorted(per_op):
        d = max(per_op[op]["d"]) if per_op[op]["d"] else None
        u = min(per_op[op]["u"]) if per_op[op]["u"] else None
        ds = f"{d:.3f}" if d is not None else "-"
        us = f"{u:,.0f}" if u is not None else "-"
        dm = f"{10.0/d:.1f}x" if d else "-"
        um = f"{u/32.0:,.0f}x" if u else "-"
        print(f"   {op:<34}{ds:>15}{dm:>10}{us:>13}{um:>10}")
    print()
    print("   A margin below ~2x on either column means the threshold is too")
    print("   close to a real in-scope operator and must be revisited before")
    print("   the detector is proposed for adoption.")


if __name__ == "__main__":
    main(*sys.argv[1:3])
