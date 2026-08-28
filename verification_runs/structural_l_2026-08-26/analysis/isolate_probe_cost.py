"""
Isolate the cost of the perturbation SENSITIVITY LOOP (the "L-probing" step)
from banked n_samples_curve arms.

Method. Every perturbation-routed check's measured duration decomposes as

    duration_ms(n)  =  a  +  b * n

where b*n is the sensitivity loop (n reference launches + n randn_like draws)
and `a` is everything else in the call that does not scale with n: the base
reference launch, the candidate launch, the quantile, the device transfer,
the shape/finite checks, and _run_check's own sync overhead.

The six banked arms (n = 3, 5, 10, 15, 20, 40) were all run with
KCC_CHECK_TIMING=1 and KCC_ABLATION_SEED=1 on the same T4 session, so the
per-check durations across arms differ ONLY by n. An OLS fit of duration on n
therefore recovers `b` -- the probing cost -- without having to assume it.

This is the same corpus (40 mutants + 200 reference trials) and the same
instrumentation every prior measurement in this project used.
"""
import gzip, json, os, glob

ARMS = "verification_runs/n_samples_curve_2026-08-25/arms"
NS = [3, 5, 10, 15, 20, 40]


def load(n):
    return json.load(gzip.open(f"{ARMS}/VALID_n{n}.json.gz"))


def records(d):
    """Yield (trial_kind, op, check_name, duration_ms) for every check run."""
    for e in d["entries"]:
        op = e["op"]
        for r in e["mutant"]["records"]:
            if r.get("duration_ms") is not None:
                yield ("mutant", op, r["name"], r["duration_ms"])
        for ref in e.get("refs", []):
            for r in ref["records"]:
                if r.get("duration_ms") is not None:
                    yield ("ref", op, r["name"], r["duration_ms"])


def is_perturbation_routed(op, name, exact_ops):
    """perturbation_tolerance always routes through the sensitivity loop.
    adversarial_* routes through it UNLESS the spec declares output_dtype
    (argmax/argmin), which checker.py sends to _check_exact_match instead."""
    if name == "perturbation_tolerance":
        return True
    if name.startswith("adversarial_") and name != "adversarial_setup":
        return op not in exact_ops
    return False


def main():
    # Identify the exact-match ops from the recorded comparator, not by name.
    d20 = load(20)
    exact_ops = set()
    pert_ops = set()
    for e in d20["entries"]:
        for r in e["mutant"]["records"] + [x for ref in e.get("refs", []) for x in ref["records"]]:
            for sc in (r.get("subchecks") or []):
                c = sc.get("comparator")
                if c == "exact_match":
                    exact_ops.add(e["op"])
                elif c == "perturbation_tolerance":
                    pert_ops.add(e["op"])
    print(f"exact-match ops (argmax/argmin path): {sorted(exact_ops)}")
    print(f"perturbation-routed ops: {len(pert_ops)}")
    print()

    # totals per arm
    totals = {}
    for n in NS:
        d = load(n)
        all_ms = 0.0
        pert_ms = 0.0
        pert_calls = 0
        by_check = {}
        for kind, op, name, ms in records(d):
            all_ms += ms
            by_check[name] = by_check.get(name, 0.0) + ms
            if is_perturbation_routed(op, name, exact_ops):
                pert_ms += ms
                pert_calls += 1
        totals[n] = dict(all_ms=all_ms, pert_ms=pert_ms, pert_calls=pert_calls,
                         by_check=by_check)

    print(f"{'n':>3} {'all checks ms':>14} {'pert path ms':>13} {'pert calls':>11} {'pert share':>11}")
    for n in NS:
        t = totals[n]
        print(f"{n:>3} {t['all_ms']:>14.1f} {t['pert_ms']:>13.1f} "
              f"{t['pert_calls']:>11d} {t['pert_ms']/t['all_ms']*100:>10.1f}%")
    print()

    # OLS of pert-path total on n  ->  slope = per-sample probing cost
    xs = [float(n) for n in NS]
    ys = [totals[n]["pert_ms"] for n in NS]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot
    print(f"OLS  pert_path_ms(n) = {a:.1f} + {b:.3f} * n     R^2 = {r2:.4f}")
    ncalls = totals[20]["pert_calls"]
    print(f"     per-call fixed  = {a/ncalls:.4f} ms   ({ncalls} calls)")
    print(f"     per-sample cost = {b/ncalls:.4f} ms/sample/call")
    print()

    at20 = totals[20]
    probe20 = b * 20
    print("=== THE NUMBER ASKED FOR, at the shipped default n_samples=20 ===")
    print(f"  sensitivity-loop (probing) cost      : {probe20:.1f} ms")
    print(f"  perturbation-path total              : {at20['pert_ms']:.1f} ms"
          f"   -> probing is {probe20/at20['pert_ms']*100:.1f}% of it")
    print(f"  all instrumented check time          : {at20['all_ms']:.1f} ms"
          f"   -> probing is {probe20/at20['all_ms']*100:.1f}% of it")
    print()

    # per-check share at n=20, top 8
    print("per-check share of measured check time (n=20 arm):")
    bc = sorted(at20["by_check"].items(), key=lambda kv: -kv[1])
    adv = sum(v for k, v in at20["by_check"].items() if k.startswith("adversarial_"))
    for k, v in bc[:8]:
        if not k.startswith("adversarial_"):
            print(f"  {k:<32} {v/at20['all_ms']*100:>6.1f}%")
    print(f"  {'all adversarial_* combined':<32} {adv/at20['all_ms']*100:>6.1f}%")


if __name__ == "__main__":
    main()
