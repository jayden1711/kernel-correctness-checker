"""
Score the layernorm-mask-fix arms against the BANKED post-oob-fix arms,
per verification_runs/layernorm_mask_bug_2026-08-27/FINDINGS.md §4 (a).

Run:  python3 validate_lnfix.py <A_lnfix.json[.gz]> <G_lnfix.json[.gz]>
      (baseline: ../../oob_fix_2026-08-28/arms/{A_fix,G_fix}.json.gz)

Criteria -- note this fix, unlike the OOB fix, MUST move exactly one thing:

  R1  Outcome diff is EXACTLY one record per arm:
      (layernorm, wrong_variance_estimate, mutant, cross_shape) fail->pass.
      Zero diffs means the fix did not take (or the banked catch was not
      bug-manufactured after all); more than one means unmodeled blast
      radius. Both are failures.
  R2  Catch-attribution diff is EXACTLY one detail string:
      '[L3]cross_shape; [L3]adversarial_wrong_variance_trigger'
      -> '[L3]adversarial_wrong_variance_trigger'. Still caught. 40/40, 0/200.
  R3  cross_shape subchecks for that mutant post-fix: all 5 shapes pass,
      including (1000,333).
  R4  oob_fix prediction (oob_fix_2026-08-28/FINDINGS.md bonus note): the
      layernorm adversarial_non_power_of_two variant (width 127) shows NO
      verdict movement and no gram fires; its measured ratio may shift by
      ~6e-5 (the pad term mean^2/127 leaving the reference) but stays deep
      in scope.
  R5  Value-level bit identity everywhere else: scope-record VALUE fields
      differ ONLY on (layernorm, adversarial_non_power_of_two) records --
      modulo the documented frobenius_norm/wrong_norm atomic-add flake.
"""

import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(HERE, "..", "..", "oob_fix_2026-08-28", "arms")

EXPECTED_FLIP = ("layernorm", "wrong_variance_estimate", "mutant",
                 "cross_shape")
PRE_DETAIL = "[L3]cross_shape; [L3]adversarial_wrong_variance_trigger"
POST_DETAIL = "[L3]adversarial_wrong_variance_trigger"

ok = True


def flag(cond, msg):
    global ok
    print(f"  {'PASS' if cond else '*** FAIL ***'}  {msg}")
    ok = ok and cond


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


def main(a_path, g_path):
    A_pre = load(os.path.join(BANK, "A_fix.json.gz"))
    G_pre = load(os.path.join(BANK, "G_fix.json.gz"))
    A_new, G_new = load(a_path), load(g_path)

    # --- R1: exactly one outcome flip, per arm --------------------------
    print("R1 outcome diff (must be exactly the cross_shape flip):")
    for name, pre, new in (("A", A_pre, A_new), ("G", G_pre, G_new)):
        o_pre, o_new = outcomes(pre), outcomes(new)
        assert [x[:4] for x in o_pre] == [x[:4] for x in o_new], \
            f"{name}: record key sequence changed"
        diffs = [(x, y) for x, y in zip(o_pre, o_new) if x != y]
        flag(len(diffs) == 1
             and diffs[0][0][:4] == EXPECTED_FLIP
             and diffs[0][0][4] == "fail" and diffs[0][1][4] == "pass",
             f"{name}-arm: {len(diffs)} outcome diff(s): "
             f"{[(d[0][:4], d[0][4], '->', d[1][4]) for d in diffs][:5]}")
        s_pre, s_new = pre["summary"], new["summary"]
        flag(s_new["n_caught"] == 40 and s_new["n_fp"] == 0,
             f"{name}-arm summary: catch {s_new['n_caught']}/40, "
             f"fp {s_new['n_fp']}/200 (pre: {s_pre['n_caught']}/40, "
             f"{s_pre['n_fp']}/200)")

    # --- R2: exactly one attribution change -----------------------------
    print("\nR2 catch attribution:")
    for name, pre, new in (("A", A_pre, A_new), ("G", G_pre, G_new)):
        pre_det = [(e["op"], e["mutant"]["name"], e["mutant"]["detail"])
                   for e in pre["entries"]]
        new_det = [(e["op"], e["mutant"]["name"], e["mutant"]["detail"])
                   for e in new["entries"]]
        diffs = [(x, y) for x, y in zip(pre_det, new_det) if x != y]
        flag(len(diffs) == 1
             and diffs[0][0] == ("layernorm", "wrong_variance_estimate",
                                 PRE_DETAIL)
             and diffs[0][1] == ("layernorm", "wrong_variance_estimate",
                                 POST_DETAIL),
             f"{name}-arm: {len(diffs)} attribution change(s): {diffs[:3]}")

    # --- R3: cross_shape subchecks post-fix -----------------------------
    print("\nR3 cross_shape subchecks, layernorm/wrong_variance_estimate:")
    for op, mu, tag, r in records(G_new):
        if (op, mu, tag, r["name"]) == EXPECTED_FLIP[:3] + ("cross_shape",):
            subs = [(s["name"], s["outcome"]) for s in r["subchecks"]]
            flag(all(o == "pass" for _, o in subs) and
                 any("333" in n for n, _ in subs),
                 f"subchecks: {subs}")

    # --- R4: non_pow2 variant no-movement -------------------------------
    print("\nR4 layernorm adversarial_non_power_of_two (width 127):")
    def np2(d):
        out = {}
        for op, mu, tag, r in records(d):
            if op == "layernorm" and r["name"] == "adversarial_non_power_of_two":
                scs = r.get("scope_flags") or []
                fired = sum(1 for sc in scs
                            for x in sc.get("reasons", [])
                            if x.get("reason") == "gram_divergence")
                med = [sc.get("gram_log10_median") for sc in scs
                       if sc.get("gram_log10_median") is not None]
                out[(mu, tag)] = (r["outcome"], fired,
                                  max((abs(m) for m in med), default=None))
        return out
    pre_np, new_np = np2(G_pre), np2(G_new)
    verd_same = all(pre_np[k][0] == new_np[k][0] for k in pre_np)
    fires = sum(v[1] for v in new_np.values())
    worst_pre = max((v[2] for v in pre_np.values() if v[2] is not None),
                    default=None)
    worst_new = max((v[2] for v in new_np.values() if v[2] is not None),
                    default=None)
    flag(verd_same and fires == 0,
         f"verdicts unchanged ({verd_same}), gram fires {fires}; "
         f"worst |log10 r| pre {worst_pre} -> post {worst_new}")

    # --- R5: bit identity outside the two expected surfaces -------------
    print("\nR5 value-level bit identity (scope records):")
    sv_pre, sv_new = scope_values(G_pre), scope_values(G_new)
    assert set(sv_pre) == set(sv_new), "scope record key sets differ"
    changed = [k for k in sv_pre if sv_pre[k] != sv_new[k]]
    allowed = [k for k in changed
               if (k[0] == "layernorm" and k[3] == "adversarial_non_power_of_two")
               or (k[0] == "frobenius_norm" and k[1] == "wrong_norm")]
    unexplained = [k for k in changed if k not in allowed]
    ln_changed = [k for k in changed if k[0] == "layernorm"]
    frob_changed = [k for k in changed if k[0] == "frobenius_norm"]
    flag(not unexplained,
         f"changed: {len(changed)} (layernorm non_pow2 {len(ln_changed)}, "
         f"frobenius flake {len(frob_changed)}); "
         f"UNEXPLAINED: {unexplained[:5]}")

    print("\n" + ("ALL CRITERIA MET" if ok else "*** REGRESSION FAILED ***"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
