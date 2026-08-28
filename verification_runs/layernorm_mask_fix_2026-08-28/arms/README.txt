Arms for the layernorm padded-lane variance fix round (2026-08-28).

A_lnfix.json.gz  KCC_CHECK_TIMING=1                                (T4, seed 1)
G_lnfix.json.gz  KCC_CHECK_TIMING=1 KCC_SCOPE_DETECT=1 KCC_SCOPE_RECORD_ALL=1
t.txt            wall-clock + RC per arm, plus GPU-probe RC

Baseline for every comparison: the BANKED post-oob-fix arms
../../oob_fix_2026-08-28/arms/{A_fix,G_fix}.json.gz (same tree except the
one-line layernorm kernel fix, same probe_redundancy.py, same seeds).

Scored by ../analysis/validate_lnfix.py -> ../analysis/out_validate_lnfix.txt
(ALL CRITERIA MET). GPU-side criteria (b)+(c) probe log:
../data/ln_gpu_probe.log (LN-GPU-PROBE-OK).
