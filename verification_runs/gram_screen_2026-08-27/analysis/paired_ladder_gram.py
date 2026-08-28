"""
Record-by-record pairing of the 2026-08-26 ladder defects (arm D, 40 deltas)
with this round's Gram ratios (arm G) on the SAME replayed inputs -- the
evidence behind FINDINGS.md §3's label revision.

Same seeds, same corpus order, and the attention mask fix is bitwise-inert
at the exercised shapes (N = 64/128, multiples of 32), so records pair 1:1.

For each large_magnitude_qk / multi_tile_rescaling record, also evaluates
the single-parameter exponential-onset model: if the local response along a
ray is s(t) = c(e^{ta} - 1), then the gram ratio is rho = (e^a - 1)/a and
the ladder defect is |(e^a-1) - 10(e^{a/10}-1)| / (e^a-1). Solving a from
the measured rho predicts the measured defect's order of magnitude and
ranking (consistency check, not a law -- the two medians are taken over
different deltas and per-record scatter is real).

Run:  python3 paired_ladder_gram.py   (from this directory)
"""
import collections
import gzip
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D_PATH = os.path.join(HERE, "..", "..", "scope_detect_2026-08-26", "arms",
                      "D_defect_n40.json.gz")
G_PATH = os.path.join(HERE, "..", "arms", "G_gram.json.gz")

KEYS = [("causal_flash_attention", "adversarial_large_magnitude_qk"),
        ("scaled_dot_product_attention", "adversarial_large_magnitude_qk"),
        ("flash_attention", "adversarial_multi_tile_rescaling")]


def recs(d):
    for e in d["entries"]:
        packs = [("mutant", e["mutant"]["records"])]
        packs += [("ref%d" % i, r["records"])
                  for i, r in enumerate(e.get("refs", []))]
        for tag, rr in packs:
            for r in rr:
                yield e["op"], tag, r


def coll(d, field):
    out = collections.defaultdict(list)
    for op, tag, r in recs(d):
        for sc in (r.get("scope_flags") or []):
            if sc.get("kind") == "scope_divergence":
                out[(op, r["name"])].append((tag, sc.get(field)))
    return out


def bis(f, lo, hi):
    flo = f(lo)
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) * flo <= 0:
            hi = mid
        else:
            lo, flo = mid, f(mid)
    return (lo + hi) / 2


def main():
    D = json.load(gzip.open(D_PATH, "rt"))
    G = json.load(gzip.open(G_PATH, "rt"))
    dd, gg = coll(D, "defect_pct"), coll(G, "gram_log10_median")
    print(f'{"record":34s} {"defect26%":>9} {"ratio27":>8} {"a":>7} {"model%":>7}')
    for key in KEYS:
        for (tagd, dv), (tagg, gv) in zip(dd[key], gg[key]):
            assert tagd == tagg
            name = key[0].split("_")[0] + "/" + key[1].split("_")[-1] + "/" + tagd
            if gv is None or dv is None:
                print(f'{name:34s} {"-" if dv is None else round(dv, 2):>9} '
                      f'{"None (floor-gated, s/ulp<32)" if gv is None else round(10 ** gv, 4):>8}')
                continue
            rho = 10 ** gv
            f = lambda a: (math.exp(a) - 1) / a - rho
            a = 0.0 if abs(rho - 1) < 1e-9 else (
                bis(f, 1e-9, 30) if rho > 1 else bis(f, -30, -1e-9))
            dm = 0.0 if a == 0 else abs(
                (math.exp(a) - 1) - 10 * (math.exp(a / 10) - 1)) / abs(
                math.exp(a) - 1) * 100
            print(f"{name:34s} {dv:9.2f} {rho:8.4f} {a:7.3f} {dm:7.2f}")


if __name__ == "__main__":
    main()
