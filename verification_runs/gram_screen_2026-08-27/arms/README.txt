Arms measured 2026-08-27, Colab T4, session kccgram (stopped).
torch 2.11.0+cu128, triton 3.6.0. Cold Triton cache; arm A paid the compiles.

A_no_detector    KCC_CHECK_TIMING=1
G_gram           + KCC_SCOPE_DETECT=1 KCC_SCOPE_RECORD_ALL=1   (Gram screen, 20 deltas)

Both arms: catch 40/40, fp 0/200. KCC_ABLATION_SEED=1 on both.
Records pair one-to-one with ../scope_detect_2026-08-26/arms/ (same corpus,
same seeds; attention mask fix bitwise-inert at the exercised shapes).
t.txt has wall clocks. Stored gzipped; every analysis script accepts .gz directly.
