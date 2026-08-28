"""
Derive the scope-detector thresholds FROM the banked data, and report the
margin each one has. No fitting, no free parameters: each threshold is the
geometric midpoint of the gap between the extreme in-scope observation and the
extreme out-of-scope one, which is the choice that maximises the smaller of the
two log-margins.

Also shows WHY the s/ulp statistic has to be the median. `cross_entropy`'s
minimum s/ulp over 40 samples is 2.0 -- numerically identical to the fp-floor
attention variants the screen is supposed to catch. On the min statistic the
two classes OVERLAP and no threshold exists. On the median they separate by
120x. Getting this wrong would have produced a detector that fires on a fully
in-scope operator on every run.
"""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from banked_fixture import (ATTENTION_VARIANTS, PRIMARY_DEFECTS,
                            WORST_IN_SCOPE_DEFECT_PCT,
                            CROSS_ENTROPY_SULP_MIN, CROSS_ENTROPY_SULP_MEDIAN,
                            OTHER_OPS_SULP_MIN_FLOOR)


def gap(lo_class, hi_class, name, unit, direction):
    """lo_class/hi_class are the two extremes bracketing the decision gap."""
    t = math.sqrt(lo_class * hi_class)
    print(f"  {name}")
    print(f"    extreme in-scope     : {lo_class if direction=='high_is_bad' else hi_class:>12,.2f} {unit}")
    print(f"    extreme out-of-scope : {hi_class if direction=='high_is_bad' else lo_class:>12,.2f} {unit}")
    print(f"    separation           : {hi_class/lo_class:>12,.1f}x")
    print(f"    THRESHOLD (geo mid)  : {t:>12,.2f} {unit}")
    print(f"    margin either side   : {math.sqrt(hi_class/lo_class):>12,.1f}x")
    return t


print("=" * 70)
print("SIGNAL 1 -- linearisation defect   (high = nonlinear = out of scope)")
print("=" * 70)
worst_primary_op = max(PRIMARY_DEFECTS, key=PRIMARY_DEFECTS.get)
print(f"  in-scope, per-operator medians : {min(PRIMARY_DEFECTS.values()):.3f}% .. "
      f"{max(PRIMARY_DEFECTS.values()):.3f}%  (worst: {worst_primary_op})")
print(f"  in-scope, worst SINGLE invocation over all 228 : {WORST_IN_SCOPE_DEFECT_PCT}%")
in_scope_att = [d for *_, d, _, ok, _ in
                [(a, b, c, s, d, cv, ok, m) for a, b, c, s, d, cv, ok, m in ATTENTION_VARIANTS]
                if ok] if False else [r[4] for r in ATTENTION_VARIANTS if r[6]]
oos_defect = [r[4] for r in ATTENTION_VARIANTS if not r[6]]
print(f"  in-scope attention variants    : {min(in_scope_att):.1f}% .. {max(in_scope_att):.1f}%")
print(f"  OUT-OF-SCOPE variants          : {min(oos_defect):.1f}% .. {max(oos_defect):.1f}%")
print()
DEFECT_T = gap(WORST_IN_SCOPE_DEFECT_PCT, min(oos_defect), "defect", "%", "high_is_bad")

print()
print("=" * 70)
print("SIGNAL 2 -- s/ulp   (low = below fp granularity = out of scope)")
print("=" * 70)
oos_sulp = [r[3] for r in ATTENTION_VARIANTS if not r[6] and r[7] == "fp_floor"]
in_sulp_att = [r[3] for r in ATTENTION_VARIANTS if r[6]]
print(f"  out-of-scope (fp floor)        : {min(oos_sulp):.2f} .. {max(oos_sulp):.2f}")
print(f"  in-scope attention variants    : {min(in_sulp_att):,.0f} .. {max(in_sulp_att):,.0f}")
print(f"  in-scope non-attention MINIMUM : >= {OTHER_OPS_SULP_MIN_FLOOR:,.0f}"
      f"   (all 26 ops except cross_entropy)")
print()
print("  Why the statistic must be the MEDIAN, not the MIN:")
print(f"    cross_entropy min s/ulp    = {CROSS_ENTROPY_SULP_MIN:.2f}   (IN SCOPE)")
print(f"    fp-floor variants  s/ulp   = {min(oos_sulp):.2f} .. {max(oos_sulp):.2f}   (OUT OF SCOPE)")
print(f"    -> on the MIN statistic the classes OVERLAP; no threshold separates them.")
print(f"    cross_entropy median s/ulp = {CROSS_ENTROPY_SULP_MEDIAN:.0f}")
print(f"    -> on the MEDIAN they separate by "
      f"{CROSS_ENTROPY_SULP_MEDIAN/max(oos_sulp):.0f}x.")
print()
SULP_T = gap(max(oos_sulp), CROSS_ENTROPY_SULP_MEDIAN, "s/ulp (median)", "ulp", "low_is_bad")

print()
print("=" * 70)
print("REJECTED SIGNALS -- recorded so they are not re-proposed")
print("=" * 70)
cv_in = [r[5] for r in ATTENTION_VARIANTS if r[6]]
cv_out = [r[5] for r in ATTENTION_VARIANTS if not r[6]]
print(f"  CV median  in-scope {min(cv_in):.3f}..{max(cv_in):.3f}   "
      f"out-of-scope {min(cv_out):.3f}..{max(cv_out):.3f}")
print(f"    -> OVERLAPS. skip_rescaling (out of scope) has CV 0.080, BELOW every")
print(f"       in-scope variant. GPU_NATIVE.md Section 4 also measured CV ranging")
print(f"       0.080-3.333 across seeds on one variant. Unusable, as already found.")
pk_in = [r[2] for r in ATTENTION_VARIANTS if r[6]]
pk_out = [r[2] for r in ATTENTION_VARIANTS if not r[6]]
print(f"  peak attn weight  in-scope {min(pk_in):.3f}..{max(pk_in):.3f}   "
      f"out-of-scope {min(pk_out):.3f}..{max(pk_out):.3f}")
print(f"    -> OVERLAPS at 1.000. Falsified in GPU_NATIVE.md Section 4.")

print()
print("=" * 70)
print(f"ADOPTED:  defect >= {DEFECT_T:.2f}%   OR   median(s/ulp) < {SULP_T:.2f}")
print(f"ROUNDED:  defect >= 10.0%            OR   median(s/ulp) < 32.0")
print("=" * 70)
