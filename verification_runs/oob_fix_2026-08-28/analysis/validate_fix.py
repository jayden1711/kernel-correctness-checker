"""
Score the OOB-fix arms against the BANKED pre-fix gram_screen arms, per the
four regression criteria of ../oob_adjudication_2026-08-28/FINDINGS.md §5.

Run:  python3 validate_fix.py <A_fix.json[.gz]> <G_fix.json[.gz]>
      (compares against ../../gram_screen_2026-08-27/arms/{A_no_detector,G_gram}.json.gz)

  C1  Verdicts byte-identical to the pre-fix arms (every check outcome, both
      arm pairs); 40/40 catch, 0/200 FP.
  C2  The two fixed classes stop being reseeding-collapsed (the falsifiable
      diagnosis check: rmsnorm's pre-fix bit-identity was OOB-content-
      driven, so post-fix its records must differ run to run through the
      varying captured gamma) AND the Gram screen now evaluates them
      (gram_n_valid = 20) and stays silent.
  C3  (contract test -- runs in pytest, reported separately.)
  C4  Exactly zero catch-attribution changes (mutant detail strings).

  PLUS the draw-then-slice guarantee, stronger than C1: outside the two
  fixed classes, every scope record's VALUE fields (gram ratios, s/ulp,
  adaptive_tol) must be BIT-identical to the banked pre-fix G arm.
"""

import collections
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(HERE, "..", "..", "gram_screen_2026-08-27", "arms")
FIXED_CLASSES = {("layernorm", "adversarial_non_power_of_two"),
                 ("rmsnorm", "adversarial_non_power_of_two")}


def load(p):
    o = gzip.open if p.endswith(".gz") else open
    with o(p, "rt") as f:
        return json.load(f)


def records(d):
    for e in d["entries"]:
        packs = [("mutant", e["mutant"]["records"])]
        packs += [(f"ref{i}", r["records"])
                  for i, r in enumerate(e.get("refs", []))]
        for tag, recs in packs:
            for r in recs:
                yield e["op"], e["mutant"]["name"], tag, r


def outcomes(d):
    return [(op, mu, tag, r["name"], r.get("outcome"))
            for op, mu, tag, r in records(d)]


def scope_values(d):
    out = {}
    for op, mu, tag, r in records(d):
        for sc in (r.get("scope_flags") or []):
            if sc.get("kind") == "scope_divergence":
                out[(op, mu, tag, r["name"])] = (
                    sc.get("gram_log10_median"),
                    tuple(sc.get("gram_log10_ratios") or ()),
                    sc.get("sulp_median"), sc.get("adaptive_tol"),
                    sc.get("in_scope"))
    return out


def main(a_fix_path, g_fix_path):
    A_pre = load(os.path.join(BANK, "A_no_detector.json.gz"))
    G_pre = load(os.path.join(BANK, "G_gram.json.gz"))
    A_fix, G_fix = load(a_fix_path), load(g_fix_path)

    # --- C1: verdict identity -------------------------------------------
    for name, pre, fix in (("A", A_pre, A_fix), ("G", G_pre, G_fix)):
        same = outcomes(pre) == outcomes(fix)
        print(f"C1 {name}-arm verdicts identical to banked pre-fix: "
              f"{'YES' if same else 'NO -- REGRESSION'}")
        if not same:
            for x, y in zip(outcomes(pre), outcomes(fix)):
                if x != y:
                    print("   first divergence:", x, "vs", y)
                    break
        print(f"   summary pre {pre['summary']}  fix {fix['summary']}")

    # --- C4: catch attribution ------------------------------------------
    pre_det = [(e["op"], e["mutant"]["name"], e["mutant"]["detail"])
               for e in G_pre["entries"]]
    fix_det = [(e["op"], e["mutant"]["name"], e["mutant"]["detail"])
               for e in G_fix["entries"]]
    print(f"\nC4 catch attribution identical: "
          f"{'YES' if pre_det == fix_det else 'NO -- REGRESSION'}")
    if pre_det != fix_det:
        for x, y in zip(pre_det, fix_det):
            if x != y:
                print("   first divergence:", x, "vs", y)

    # --- draw-then-slice: bitwise identity outside the fixed classes ----
    sv_pre, sv_fix = scope_values(G_pre), scope_values(G_fix)
    assert set(sv_pre) == set(sv_fix), "record key sets differ"
    diff_out, diff_in = [], []
    for k, v in sv_pre.items():
        cls = (k[0], k[3])
        if sv_fix[k] != v:
            (diff_in if cls in FIXED_CLASSES else diff_out).append(k)
    print(f"\nDRAW-THEN-SLICE: records changed outside the fixed classes: "
          f"{len(diff_out)}"
          f"{'  *** STREAM SHIFTED ***' if diff_out else '   (bit-identical, as designed)'}")
    for k in diff_out[:5]:
        print("   changed:", k)
    print(f"   records changed inside the two fixed classes: {len(diff_in)} "
          f"(expected: all that carry measured values)")

    # --- C2: collapse resolution + gram evaluation ----------------------
    print("\nC2 the two fixed classes, post-fix:")
    for op, chk in sorted(FIXED_CLASSES):
        recs = [sc for o, mu, tag, r in records(G_fix) if o == op
                and r["name"] == chk
                for sc in (r.get("scope_flags") or [])]
        fps = {(tuple(sc.get("gram_log10_ratios") or ()),
                sc.get("sulp_median")) for sc in recs}
        n_valid = collections.Counter(sc.get("gram_n_valid") for sc in recs)
        meds = [sc.get("gram_log10_median") for sc in recs
                if sc.get("gram_log10_median") is not None]
        fired = sum(1 for sc in recs
                    if any(x["reason"] == "gram_divergence"
                           for x in sc["reasons"]))
        worst = max((abs(m) for m in meds), default=None)
        print(f"   {op}/{chk}: {len(recs)} records, "
              f"{len(fps)} DISTINCT fingerprints "
              f"({'collapse RESOLVED' if len(fps) > 1 else '*** STILL COLLAPSED -- DIAGNOSIS WRONG ***'}), "
              f"gram_n_valid={dict(n_valid)}, gram fires {fired}, "
              f"worst |log10 r| = "
              f"{worst if worst is None else round(worst, 4)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
