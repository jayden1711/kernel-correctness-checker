# Corrected AutoKernel re-run — 2026-08-25

The three corrections in `AUTOKERNEL_BASELINE_AUDIT.md` §7 change what the gate
does, so the 80.0% / 0.5% figures could not be reused. Re-measured on a Colab
T4, unmodified corpus, shipped-warm Triton cache (~55s per run).

## Result: the catch rate is unchanged, and the FP rate survives too

| reading | catch | FP | p50 | p90 | mean |
|---|---:|---:|---:|---:|---:|
| **A.** pre-correction (superseded) | 80.0% | 0.5% | 12.02 ms | 336.05 ms | 59.88 ms |
| **B.** corrected, literal | **80.0%** | **13.0%** | 15.58 ms | 345.40 ms | 65.41 ms |
| **C.** corrected + reference-infeasible skip | **80.0%** | **0.5%** | 16.18 ms | 353.38 ms | 64.92 ms |

**C is the reading to cite.** B and C differ by one rule, explained below.

Artifacts: `rerun_B_literal/`, `rerun_C_skip/`.

## Catch rate: 80.0% in all three, with the identical 8 misses

```
avg_pool{1,2,3}d/wrong_divisor    max_pool{1,2,3}d/wrong_padding
max_reduction/wrong_padding       min_reduction/wrong_padding
```

**This is the axis argument confirming itself.** Stage 3 was widened from 3
probe classes on 11 operators to 5 value transforms on all 29 — a large
increase in *value-distribution* coverage — and it moved the catch rate by
exactly **zero**. The surviving bugs are conditional on **hyperparameters**
(`padding`), and no value transform can reach a hyperparameter. Predicted
before the run; confirmed by it.

The looser fp32 `rtol` (1e-5 → 1e-4) also cost nothing, which says the mutants
this corpus catches are caught by margins far wider than 10x tolerance.

## Why B and C differ — and why C is right

Under B the FP rate is 13.0%: **26 false positives, of which 25 are one
mechanism.**

| count | mechanism |
|---:|---|
| 25 | `numerical_stability/near_max: mismatch` |
| 1 | `determinism: three runs not bitwise identical` |

`near_max` scales the primary by `1e30`. For the attention family QK<sup>T</sup>
then reaches ~1e60, which **overflows fp32** (max ≈ 3.4e38) and returns `inf`.
The reference does not raise — it silently returns non-finite output.
`_allclose` rejects non-finite candidate output, so on a
**reference-vs-reference** trial the reference's own overflow was scored as a
candidate failure: a deterministic false positive, 100% of `flash_attention`
and `scaled_dot_product_attention` trials.

This is exactly the artifact class the original audit was written to eliminate —
a re-implementation detail manufacturing false positives on a correct kernel,
via a mechanism that says nothing about the candidate.

C applies **the file's own existing rule** to the silent case. `autokernel_faithful.py`
already skips a config when the reference *raises* ("that is a limitation of
this corpus's references, not evidence about the candidate"). A reference that
cannot produce a **finite** answer is the same situation. The old stage 3 never
hit it because it probed only 11 of 29 operators, none of which overflowed.

Under C the FP rate returns to **0.5% — a single `frobenius_norm` determinism
flip**, the known `tl.atomic_add` non-associativity flake documented in
SESSION_HANDOFF §3, which varies 0/5–2/5 run to run with no code change.

**Not established:** whether AutoKernel's real `bench.py` skips or fails a
non-finite reference output. Its stage-1 criterion is documented as "matches
reference within tolerance, no NaN/Inf", but stage-3 handling could not be read
from the source I fetched. **If it fails rather than skips, B is the faithful
number and AutoKernel has a real 13% FP rate on this corpus driven entirely by
attention overflow.** Both readings are banked so the choice is visible rather
than silently made.

## Latency moved slightly, in the expected direction

p50 12.02 → 16.18 ms, mean 59.88 → 64.92 ms. Stage 3 now runs 5 transforms on
29 operators where it ran 3 on 11, so the gate does **more** work than before.
This reinforces rather than weakens `FINDINGS.md`'s conclusion: the faithful
gate is not the cheap option.

The pass-path latency table in `FINDINGS.md` is from the pre-correction run and
is **not** restated here; the corrected gate is ~8% more expensive, which does
not change any comparison in it.

## One other movement, and it is noise

`autokernel_gate` (the old approximation, untouched by these corrections) moved
FP 17.5% → 18.0% — one `frobenius_norm` trial, the same atomic-add flake. Every
other system is bit-identical across B and C.
