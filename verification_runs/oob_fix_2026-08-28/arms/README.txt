Arms measured 2026-08-28, Colab T4, session kccfix (stopped).
torch 2.11.0+cu128, triton 3.6.0. Cold Triton cache; A_fix paid the compiles.

A_fix    KCC_CHECK_TIMING=1
G_fix    + KCC_SCOPE_DETECT=1 KCC_SCOPE_RECORD_ALL=1

Both: catch 40/40, fp 0/200, KCC_ABLATION_SEED=1. Scored against the BANKED
pre-fix ../../gram_screen_2026-08-27/arms/ (same corpus/seeds; the OOB fix is
draw-then-slice so all deterministic records outside the two fixed classes
are bit-identical to pre-fix). wrapper_assert.log: the reference-wrapper
shape asserts exercised on GPU (3/3 mismatches raise ValueError, matched
companions run).
