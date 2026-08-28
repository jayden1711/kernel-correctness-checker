Arms measured 2026-08-26, Colab T4, session kccscope (stopped).
torch 2.11.0+cu128, triton 3.6.0. Cold Triton cache; arm A paid the compiles (72.5s), B/C/D warm (17-20s).

A_no_detector    KCC_CHECK_TIMING=1
B_detector       + KCC_SCOPE_DETECT=1 KCC_SCOPE_RECORD_ALL=1            (3 deltas, the then-default)
C_defect_n20     + KCC_SCOPE_DEFECT_SAMPLES=20
D_defect_n40     + KCC_SCOPE_DEFECT_SAMPLES=40

All four: catch 40/40, fp 0/200. KCC_ABLATION_SEED=1 on every arm.
