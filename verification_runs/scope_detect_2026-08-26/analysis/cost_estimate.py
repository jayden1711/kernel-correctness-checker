"""
What the detector costs, from the banked per-sample launch price.

The detector adds exactly KCC_SCOPE_DEFECT_SAMPLES reference launches per
perturbation-routed call -- s(1) is already in hand from the sensitivity loop,
only s(0.1) is new -- and draws no extra RNG (it reuses the loop's deltas).
s/ulp is free: it is computed from sensitivities already collected.

Price per launch comes from ../../structural_l_2026-08-26/: the sensitivity
loop's measured slope of 0.1218 ms/sample/call over 856 calls on a T4 under
KCC_CHECK_TIMING=1. Instrumented, so this is an upper bound.
"""
import gzip, json

MS_PER_LAUNCH = 0.1218        # measured slope, structural_l_2026-08-26
WALL20 = 8474.3               # instrumented checker wall at n=20, banked
CORPUS_S, CHECKER_S = 60.8, 9.89
EXACT = {"argmax", "argmin"}

d = json.load(gzip.open(
    "verification_runs/n_samples_curve_2026-08-25/arms/VALID_n20.json.gz"))

calls = 0
for e in d["entries"]:
    if e["op"] in EXACT:
        continue          # discrete output: flagged structurally, no probe
    rs = [r for r in e["mutant"]["records"]] + \
         [r for rf in e["refs"] for r in rf["records"]]
    for r in rs:
        n = r["name"]
        if n == "perturbation_tolerance" or (
                n.startswith("adversarial_") and n != "adversarial_setup"):
            calls += 1

print(f"perturbation-routed calls that would run the defect probe : {calls}")
print(f"(argmax/argmin excluded -- discrete output is flagged with no probe)")
print()
print(f"{'defect samples':>15}{'extra launches':>16}{'added ms':>11}"
      f"{'checker wall':>14}{'corpus':>9}")
for k in (1, 3, 5, 20):
    ms = calls * k * MS_PER_LAUNCH
    pct = ms / WALL20
    print(f"{k:>15}{calls*k:>16,}{ms:>11.0f}{pct*100:>13.1f}%"
          f"{CHECKER_S*pct/CORPUS_S*100:>8.1f}%")
print()
print("Shipped default is 3. The banked defect medians are over 40 samples, so")
print("3 is a cost choice that has NOT been validated for stability -- arm C of")
print("scopedet.sh exists to settle it against 20.")
