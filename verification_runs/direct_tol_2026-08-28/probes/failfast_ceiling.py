"""
Ceiling arithmetic for the two SMALL levers the binding/validity theory
suggests, computed from the banked current arms (no new GPU run):

  L1 FAIL-FAST ORDERING: with the binding-check table one could order each
     op's Layer-3 battery by predicted delta* and stop at the first fail.
     Verdict-identical by construction (any fail is a fail; references run
     the full battery either way). The ceiling is the serialised time of
     checks that ran AFTER the first failing check on each caught-trial,
     as a share of total serialised check time.

  L2 PRECISION-COERCION FP16-ARM DROP: l3_margins showed the 0.9 factor
     live on 4/52 records with an 11%-to-36% dead zone; dropping the fp16
     arm can only ADD catches (the factor gates fails off). Ceiling = the
     fp16-arm share of precision_coercion time (~half) as a share of total.

Data: ../../layernorm_mask_fix_2026-08-28/arms/A_lnfix.json.gz
(current-arms baseline, KCC_CHECK_TIMING=1 -- shares meaningful,
absolutes serialised upper bounds).

Run:  .venv/bin/python failfast_ceiling.py
"""
import gzip
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = os.path.join(HERE, "..", "..", "layernorm_mask_fix_2026-08-28",
                   "arms", "A_lnfix.json.gz")


def main():
    d = json.load(gzip.open(ARM))
    total_ms = 0.0
    after_fail_ms = 0.0          # L3 checks after the first L3 fail, caught trials
    pc_ms = 0.0
    n_trials = 0
    n_caught_l3 = 0
    for e in d["entries"]:
        trials = [("mutant", e["mutant"])] + [
            (f"ref{i}", r) for i, r in enumerate(e["refs"])]
        for tag, rec in trials:
            n_trials += 1
            recs = rec["records"]
            for x in recs:
                if x.get("duration_ms"):
                    total_ms += x["duration_ms"]
                if x["name"] == "precision_coercion" and x.get("duration_ms"):
                    pc_ms += x["duration_ms"]
            # layer-3 fail-fast: records are in execution order
            l3 = [x for x in recs if x.get("layer") == 3]
            fail_idx = next((i for i, x in enumerate(l3)
                             if x["outcome"] == "fail"), None)
            if fail_idx is not None:
                n_caught_l3 += 1
                after_fail_ms += sum(x.get("duration_ms") or 0
                                     for x in l3[fail_idx + 1:])
    print(f"trials: {n_trials}; trials with an L3 fail: {n_caught_l3}")
    print(f"total serialised check time: {total_ms/1000:.2f} s")
    print(f"L1 fail-fast ceiling (post-first-fail L3 time on failing "
          f"trials): {after_fail_ms/1000:.2f} s = "
          f"{after_fail_ms/total_ms:.2%} of check time")
    print(f"L2 precision_coercion total: {pc_ms/1000:.2f} s "
          f"({pc_ms/total_ms:.2%}); fp16-arm share ~half -> ceiling "
          f"~{pc_ms/2/total_ms:.2%} of check time")
    # translate to corpus share using the check_timing round's calibration:
    # checker check-time is ~16% of corpus runtime (9.89 s of 60.8 s)
    f = 9.89 / 60.8
    print(f"as corpus share (x{f:.3f}): fail-fast "
          f"~{after_fail_ms/total_ms*f:.2%}, fp16-arm "
          f"~{pc_ms/2/total_ms*f:.2%}")


if __name__ == "__main__":
    main()
