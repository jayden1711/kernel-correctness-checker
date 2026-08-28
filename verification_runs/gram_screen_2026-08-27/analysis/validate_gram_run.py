"""
Score a GPU arm pair from gramdet.sh.

Run as:  python3 validate_gram_run.py <A_no_detector.json[.gz]> <G_gram.json[.gz]>

Questions, in the order the 2026-08-26 round asked them (same standard):

  1. Verdict safety: detector on must change nothing. Byte-level comparison of
     every check outcome, plus the scope_flags promotion count.
  2. The KNOWN out-of-scope saturation cases -- the three attention classes
     GPU_NATIVE.md Section 4 established and the defect screen could not
     separate -- must carry a Gram fire on every invocation.
  3. The KNOWN in-scope set must be silent, WITH MARGIN: every primary
     invocation, and the two attention adversarial variants GPU_NATIVE.md
     validated as in scope (approx_denominator, wrong_causal_mask). Margin is
     |log10 ratio| distance to the log10(2) threshold.
  4. The floor screen must reproduce its validated 2026-08-26 fire set
     (regression -- it was not touched).
  5. The formerly-UNSCORED adversarial classes (the 2026-08-26 round's "10
     out-of-expectation fires", which had no ground truth) are reported in
     their own bucket. The Gram statistic itself upgrades them from unscored
     to adjudicated: |log10 r| far from 0 is direct evidence the measured
     response escapes the Jacobian at that input (the exact derivative IS the
     ground truth for linearity), not a threshold judgement.
  6. Separation: worst in-scope |log10 r| vs least out-of-scope |log10 r| --
     the number whose 2026-08-26 value (0.68x, i.e. overlap) killed the
     defect screen. > 1 means separated; report the factor.
  7. Probe-size convergence, offline: classification at k = 3..20 from the
     banked per-delta ratios (prefix property; no extra arms needed).
"""
import collections
import gzip
import json
import math
import statistics
import sys

THRESH = math.log10(2.0)

# The three saturation classes with banked ground truth (GPU_NATIVE.md §4).
EXPECT_GRAM_FLAGGED = {
    ("causal_flash_attention", "adversarial_large_magnitude_qk"),
    ("flash_attention", "adversarial_multi_tile_rescaling"),
    ("scaled_dot_product_attention", "adversarial_large_magnitude_qk"),
}
# Attention adversarial variants validated IN scope by the same document.
EXPECT_SILENT_ATTENTION = {
    ("flash_attention", "adversarial_approx_denominator"),
    ("flash_attention", "adversarial_wrong_causal_mask"),
}
# Floor-mechanism classes the (unchanged) s/ulp screen owns; regression set
# from the 2026-08-26 run's fire table.
EXPECT_FLOOR_FLAGGED = {
    ("flash_attention", "adversarial_last_tile_dropped"),
    ("flash_attention", "adversarial_skip_rescaling"),
    ("flash_attention", "adversarial_equal_attention_weights"),
}
STRUCTURAL = {"argmax", "argmin"}

# The 2026-08-26 round's out-of-expectation fires -- no ground-truth label
# existed for these. Reported separately, adjudicated by the Gram evidence.
UNSCORED_2026_08_26 = {
    ("softmax", "adversarial_near_zero_variance"),
    ("softmax", "adversarial_max_in_last_tile"),
    ("softmax", "adversarial_extreme_range"),
    ("log_softmax", "adversarial_near_zero_variance"),
    ("groupnorm", "adversarial_near_zero_variance"),
    ("instancenorm", "adversarial_near_zero_variance"),
    ("swish", "adversarial_near_global_min"),
    ("gelu", "adversarial_near_global_min"),
    ("cross_entropy", "adversarial_large_magnitude_logits"),
    ("layernorm", "adversarial_wrong_variance_trigger"),
}


def load(p):
    o = gzip.open if p.endswith(".gz") else open
    with o(p, "rt") as f:
        return json.load(f)


def records(d):
    for e in d["entries"]:
        rs = [("mutant", r) for r in e["mutant"]["records"]]
        rs += [("ref", r) for rf in e.get("refs", []) for r in rf["records"]]
        for kind, r in rs:
            yield e["op"], e["mutant"]["name"], kind, r


def main(a_path, g_path):
    A, G = load(a_path), load(g_path)

    # --- 1. verdict safety -------------------------------------------------
    va = [(op, mu, k, r["name"], r.get("outcome", r.get("passed")))
          for op, mu, k, r in records(A)]
    vg = [(op, mu, k, r["name"], r.get("outcome", r.get("passed")))
          for op, mu, k, r in records(G)]
    same = va == vg
    print(f"1. VERDICTS IDENTICAL WITH DETECTOR ON : "
          f"{'YES' if same else 'NO -- WIRING BUG'}")
    print(f"   summary A {A['summary']}")
    print(f"   summary G {G['summary']}")
    if not same:
        for x, y in zip(va, vg):
            if x != y:
                print(f"   first divergence: {x} vs {y}")
                break

    scopes = collections.defaultdict(list)
    n_promoted = n_sub = 0
    for op, mu, kind, r in records(G):
        n_sub += sum(1 for sc in (r.get("subchecks") or [])
                     if isinstance(sc, dict)
                     and sc.get("kind") == "scope_divergence")
        for sc in (r.get("scope_flags") or []):
            if sc.get("kind") == "scope_divergence":
                n_promoted += 1
                scopes[(op, r["name"])].append(sc)
    print(f"\n   scope records promoted/{'source':<6}: {n_promoted}/{n_sub}"
          + ("" if n_promoted == n_sub else "  *** PROMOTION MISMATCH ***"))

    def reasons_of(sc):
        return {x["reason"] for x in sc["reasons"] if x["severity"] != "advisory"}

    def fmt_g(sc):
        g = sc.get("gram_log10_median")
        return f"{g:+.3f}" if g is not None else "   -  "

    # --- 2. the three known saturation classes -----------------------------
    print("\n2. KNOWN OUT-OF-SCOPE (saturation) -- must carry gram_divergence")
    ok = True
    for key in sorted(EXPECT_GRAM_FLAGGED):
        recs = scopes.get(key, [])
        hit = sum(1 for sc in recs if "gram_divergence" in reasons_of(sc))
        meds = [sc.get("gram_log10_median") for sc in recs]
        meds = [m for m in meds if m is not None]
        lo = min((abs(m) for m in meds), default=None)
        ok &= recs != [] and hit == len(recs)
        print(f"   {key[0]}/{key[1]:<44} {hit}/{len(recs)} fired"
              f"   min|log10 r| = {lo if lo is None else round(lo, 3)}")
    print(f"   => {'ALL FLAGGED, EVERY INVOCATION' if ok else '*** MISSES ***'}")

    # --- 3. known in-scope silence, with margin ----------------------------
    print("\n3. KNOWN IN-SCOPE -- must be gram-silent; margin = "
          "log10(2) - |log10 r|")
    worst = (None, -1.0)
    fails = []
    rows = []
    for (op, chk), recs in sorted(scopes.items()):
        primary = chk == "perturbation_tolerance" and op not in STRUCTURAL
        att_ok = (op, chk) in EXPECT_SILENT_ATTENTION
        if not (primary or att_ok):
            continue
        meds = [sc.get("gram_log10_median") for sc in recs]
        have = [abs(m) for m in meds if m is not None]
        fired = [sc for sc in recs if "gram_divergence" in reasons_of(sc)]
        if fired:
            fails.append((op, chk))
        w = max(have) if have else None
        if w is not None and w > worst[1]:
            worst = ((op, chk), w)
        rows.append((op, chk, len(recs), len(have), w))
    for op, chk, n, nh, w in rows:
        ws = f"{w:.4f}" if w is not None else "no gram signal"
        mg = f"{THRESH - w:+.3f}" if w is not None else "-"
        print(f"   {op + '/' + chk:<58} n={n:<3} gram_on={nh:<3} "
              f"worst|log10 r|={ws:<14} margin {mg}")
    print(f"   in-scope gram fires: {fails if fails else 'NONE'}")
    print(f"   worst in-scope: {worst[0]} |log10 r| = {worst[1]:.4f}")

    # --- 4. floor regression ----------------------------------------------
    print("\n4. FLOOR SCREEN regression -- 2026-08-26 fire set must reproduce")
    for key in sorted(EXPECT_FLOOR_FLAGGED):
        recs = scopes.get(key, [])
        hit = sum(1 for sc in recs if "quantisation_floor" in reasons_of(sc))
        print(f"   {key[0]}/{key[1]:<44} {hit}/{len(recs)} floor-fired")

    # --- 5. the formerly-unscored bucket -----------------------------------
    print("\n5. FORMERLY-UNSCORED classes (2026-08-26 had no ground truth) --")
    print("   adjudicated by the exact-derivative evidence, not by a label")
    for key in sorted(UNSCORED_2026_08_26):
        recs = scopes.get(key, [])
        if not recs:
            print(f"   {key[0]}/{key[1]:<44} (no records)")
            continue
        gs = [sc.get("gram_log10_median") for sc in recs]
        gs_have = [g for g in gs if g is not None]
        n_gram = sum(1 for sc in recs if "gram_divergence" in reasons_of(sc))
        n_floor = sum(1 for sc in recs if "quantisation_floor" in reasons_of(sc))
        med = statistics.median(gs_have) if gs_have else None
        ms = f"{med:+.3f}" if med is not None else "  -  "
        print(f"   {key[0]}/{key[1]:<44} gram fires {n_gram}/{len(recs)}, "
              f"floor {n_floor}/{len(recs)}, median log10 r = {ms}")

    # --- 6. separation ------------------------------------------------------
    print("\n6. SEPARATION (the number that killed the defect screen)")
    out_vals = []
    for key in EXPECT_GRAM_FLAGGED:
        for sc in scopes.get(key, []):
            m = sc.get("gram_log10_median")
            if m is not None:
                out_vals.append(abs(m))
    if out_vals and worst[1] >= 0:
        least_out = min(out_vals)
        sep = least_out / worst[1] if worst[1] > 0 else float("inf")
        print(f"   worst in-scope |log10 r|      : {worst[1]:.4f}  {worst[0]}")
        print(f"   least out-of-scope |log10 r|  : {least_out:.4f}")
        print(f"   separation factor             : {sep:.1f}x "
              f"({'SEPARATED' if least_out > worst[1] else '*** OVERLAP -- FALSIFIED ***'})")
        print(f"   threshold log10(2) margins    : in-scope "
              f"{THRESH / worst[1] if worst[1] > 0 else float('inf'):.1f}x below, "
              f"out-of-scope {least_out / THRESH:.1f}x above")

    # --- 7. probe-size convergence (offline, prefix property) ---------------
    print("\n7. PROBE-SIZE CONVERGENCE -- classification at k vs k=20")
    full = {}
    for key, recs in scopes.items():
        for i, sc in enumerate(recs):
            rr = sc.get("gram_log10_ratios")
            if rr:
                full[(key, i)] = rr
    for k in (3, 5, 8, 10, 15, 20):
        diff = 0
        for (_key, _i), rr in full.items():
            pref = rr[:k]
            med_k = sorted(pref)[len(pref) // 2] if len(pref) >= 5 else None
            med_f = sorted(rr)[len(rr) // 2]
            fk = med_k is not None and abs(med_k) >= THRESH
            ff = abs(med_f) >= THRESH
            diff += (fk != ff)
        print(f"   k={k:<3} classification differs from k=20 on {diff}/{len(full)}"
              f" records")


if __name__ == "__main__":
    main(*sys.argv[1:3])
