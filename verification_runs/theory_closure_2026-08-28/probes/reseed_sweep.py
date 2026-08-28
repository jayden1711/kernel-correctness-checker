"""
RESEEDING-COLLAPSE SWEEP -- how many banked "n = k" records are independent?

Mechanism (checker.py KCC_ABLATION_SEED): torch's global generator is
reseeded from crc32(check_name) before every check. Consequence: any
quantity a check derives from torch RNG is IDENTICAL on every invocation of
that check name -- across the 5 reference runs, the mutant run, and even
across different mutant entries of the same operator. Two populations:

  COLLAPSED   adversarial generators that draw FRESH torch randn
              (flash_attention's _make_qkv family, layernorm's randn_like
              variants, ...): the variant input is the same tensor every
              time. A class reported as "22/22" is one draw replayed 22x.
  VARYING     captured-transform generators (x*20, x+150, zero-half, ...):
              the transform is deterministic but the CAPTURED base input
              comes from the harness's numpy rng, which advances across
              runs -- these are genuinely distinct draws.

The perturbation DELTAS are reseeded the same way, so within a class even
the deltas repeat; a collapsed class repeats (input, deltas) exactly and its
records are bit-identical replicas.

This probe does not argue from the code -- it counts. Fingerprint per
record: the banked 40-sample sensitivity vector (CURVE/VALID arms), the
(defect_pct, sulp_median) pair (scope arms), or the 20-entry gram ratio
vector (gram arm). Bit-identical fingerprints = replicas.

Output: per round, per (op, check): n_records vs n_distinct, plus the
roll-up used for the FINDINGS corrections.
"""

import collections
import gzip
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "..")
DATA = os.path.join(HERE, "..", "data")

BANKS = [
    ("n_samples_curve_2026-08-25/arms/CURVE_n40.json.gz", "sens"),
    ("n_samples_curve_2026-08-25/arms/VALID_n20.json.gz", "sens"),
    ("scope_detect_2026-08-26/arms/D_defect_n40.json.gz", "scope"),
    ("scope_detect_2026-08-26/arms/B_detector.json.gz", "scope"),
    ("gram_screen_2026-08-27/arms/G_gram.json.gz", "gram"),
]


def load(rel):
    p = os.path.join(RUNS, rel)
    o = gzip.open if p.endswith(".gz") else open
    with o(p, "rt") as f:
        return json.load(f)


def records(d):
    for e in d["entries"]:
        packs = [e["mutant"]["records"]]
        packs += [r["records"] for r in e.get("refs", [])]
        for recs in packs:
            for r in recs:
                yield e["op"], r


def fingerprint(kind, r):
    if kind == "sens":
        for sc in (r.get("subchecks") or []):
            if isinstance(sc, dict) and sc.get("kind") == "perturbation_sensitivities":
                return tuple(sc["sensitivities"])
    elif kind == "scope":
        for sc in (r.get("scope_flags") or []):
            if sc.get("kind") == "scope_divergence" and (
                    sc.get("defect_pct") is not None
                    or sc.get("sulp_median") is not None):
                return (sc.get("defect_pct"), sc.get("sulp_median"))
    elif kind == "gram":
        for sc in (r.get("scope_flags") or []):
            if sc.get("kind") == "scope_divergence":
                rr = sc.get("gram_log10_ratios")
                if rr:
                    return tuple(rr)
                if sc.get("sulp_median") is not None:
                    return ("sulp", sc.get("sulp_median"))
    return None


def main():
    os.makedirs(DATA, exist_ok=True)
    out = {}
    for rel, kind in BANKS:
        d = load(rel)
        cls = collections.defaultdict(list)
        for op, r in records(d):
            fp = fingerprint(kind, r)
            if fp is not None:
                cls[(op, r["name"])].append(fp)
        rows = []
        for (op, chk), fps in sorted(cls.items()):
            rows.append(dict(op=op, check=chk, n=len(fps),
                             distinct=len(set(fps))))
        out[rel] = rows
        n_cls = len(rows)
        collapsed = [r for r in rows if r["n"] >= 4 and r["distinct"] == 1]
        partial = [r for r in rows if 1 < r["distinct"] < r["n"]]
        full = [r for r in rows if r["distinct"] == r["n"]]
        print(f"\n=== {rel}  ({n_cls} classes) ===")
        print(f"  fully distinct: {len(full)}   partial: {len(partial)}   "
              f"COLLAPSED (>=4 records, 1 draw): {len(collapsed)}")
        for r in collapsed:
            print(f"    COLLAPSED  {r['op']}/{r['check']:44s} "
                  f"{r['n']} records, 1 distinct")
        for r in partial:
            print(f"    partial    {r['op']}/{r['check']:44s} "
                  f"{r['n']} records, {r['distinct']} distinct")
    json.dump(out, open(os.path.join(DATA, "reseed_sweep.json"), "w"))
    print("\nwrote data/reseed_sweep.json")


if __name__ == "__main__":
    main()
