"""
Runtime cost of the Gram screen: A vs G per-check durations, KCC_CHECK_TIMING
serialised. NOTE these walls are upper bounds and not comparable to published
latency (same caveat as every timed arm); the comparison is A-vs-G only.

Run:  python3 cost_compare.py <A.json[.gz]> <G.json[.gz]>
"""
import collections
import gzip
import json
import statistics
import sys


def load(p):
    o = gzip.open if p.endswith(".gz") else open
    with o(p, "rt") as f:
        return json.load(f)


def per_check(d):
    out = collections.defaultdict(list)
    tot = collections.defaultdict(float)
    for e in d["entries"]:
        packs = [e["mutant"]["records"]] + [r["records"] for r in e.get("refs", [])]
        for recs in packs:
            for r in recs:
                if r.get("duration_ms") is not None:
                    routed = (r["name"] == "perturbation_tolerance"
                              or r["name"].startswith("adversarial_"))
                    out[routed].append(r["duration_ms"])
                    tot["all"] += r["duration_ms"]
                    if routed:
                        tot["routed"] += r["duration_ms"]
    return out, tot


def main(a_path, g_path):
    A, ta = per_check(load(a_path))
    G, tg = per_check(load(g_path))
    for routed in (True, False):
        na, ng = A[routed], G[routed]
        print(f"{'perturbation-routed' if routed else 'other checks':<22} "
              f"A median {statistics.median(na):8.2f} ms  "
              f"G median {statistics.median(ng):8.2f} ms  "
              f"(n={len(na)}/{len(ng)})")
    print(f"total check wall: A {ta['all'] / 1000:.1f}s  G {tg['all'] / 1000:.1f}s  "
          f"(+{(tg['all'] - ta['all']) / ta['all'] * 100:.1f}%)")
    print(f"routed-only wall: A {ta['routed'] / 1000:.1f}s  G {tg['routed'] / 1000:.1f}s  "
          f"(+{(tg['routed'] - ta['routed']) / ta['routed'] * 100:.1f}%)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
