# `n_samples` — the corpus cannot answer this question, and the honest saving is ~2–3%

**Measured 2026-08-25 on a Colab T4** (shipped-warm Triton cache). Driver
`nsamp.sh`, probe `probe_redundancy.py`, arms in `arms/`.
`n_samples` default is **UNCHANGED at 20**; nothing was altered this pass.

---

## Verdict

1. **Catch and FP are flat at 40/40 and 0/200 for every n from 1 to 40.** Not
   "20 is fine" — the corpus has **no signal at all** on this parameter.
2. **The reason is saturation: 806 of 854 perturbation invocations have
   `max_err` exactly 0.0** (candidate bitwise identical to reference). They
   cannot flip at any tolerance. The entire false-positive signal in a
   200-reference-trial run rests on **one** invocation.
3. **The saving is ~2–3% of corpus runtime, not the order-of-magnitude win the
   previous round projected.** That projection is corrected below.

---

## Method — the curve is derived exactly, not sampled

The verdict is `fail ⟺ max_err > scale · quantile(sensitivities[:n], 0.95)`.
`max_err` does not depend on `n`, and `perturbation.py` draws deltas **one at a
time** in a fixed order from a per-check seed (`KCC_ABLATION_SEED=1`). So the
n-sample sensitivity vector is a strict **prefix** of the 40-sample one.

Recording the full vector once at n=40 therefore determines the verdict at every
smaller n **exactly**, with no cross-arm RNG noise and no extra GPU time. Six
discrete arms (n = 3, 5, 10, 15, 20, 40) were then actually run as validation —
every one matched the derived curve.

Instrumentation added, both opt-in and default OFF:
`KCC_N_SAMPLES=<int>`, `KCC_RECORD_SENSITIVITIES=1`.

**Direction was predicted before the run and held.** `adaptive_tol` is `3.0 ×
q95(sensitivities)`, an order statistic — fewer samples ⇒ smaller q95 ⇒ *tighter*
band ⇒ more failures. So the risk of lowering `n_samples` is **false positives,
never lost catches**. Confirmed: catch safety *improves* monotonically as n falls.

---

## The curve

| n | catch | FP | | n | catch | FP |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 40/40 | 0/200 | | 12 | 40/40 | 0/200 |
| 3 | 40/40 | 0/200 | | 15 | 40/40 | 0/200 |
| 5 | 40/40 | 0/200 | | 20 | 40/40 | 0/200 |
| 10 | 40/40 | 0/200 | | 40 | 40/40 | 0/200 |

Flat at every integer n from 1 to 40. **No mutant is ever lost; no false
positive is ever gained.**

## Why the curve is uninformative

| | count |
|---|---:|
| perturbation invocations recorded | 854 |
| with `max_err` **exactly 0.0** | **806 (94.4%)** |
| …of which are reference trials | 784 |
| reference invocations with nonzero `max_err` | **1** |
| mutant invocations that actually fail | 36 |

A reference trial runs the candidate *against itself*, so `max_err` is
identically zero and `0 > tol` is false for any tolerance. **94% of the
measurement is structurally incapable of responding to `n_samples`.**

Safety factors, split by trial kind (a passing *mutant* invocation flipping would
*gain* a catch; only a *reference* one is a true FP):

| n | FP safety (reference) | catch safety (mutant) |
|---:|---:|---:|
| 1 | 1111.6 | 1.839 |
| 5 | 1162.4 | 1.325 |
| 10 | 1174.5 | 1.365 |
| 20 | 1180.8 | 1.453 |
| 40 | 1211.9 | 1.462 |

The FP column is **one invocation** — `frobenius_norm/adversarial_dominant_outlier`,
`max_err` 1.19e-07 against `tol` 1.41e-04, a ratio of 0.0008. It carries 1174x of
headroom and it is the *entire* false-positive evidence base. **That is the
honest limit of what this corpus can say.**

## The floor, and a concern that did not survive checking

`adaptive_tol` is floored at 1e-6. At the floor the check stops being adaptive
and becomes exact-match. The worst single tolerance collapse going 20 → 3 is
**7.8e4x**, which looks alarming — but the frequency is flat:

| n | invocations at the floor |
|---:|---:|
| 1 | 105 (12.3%) |
| 3 | 99 (11.6%) |
| 10 | 99 (11.6%) |
| 20 | 98 (11.5%) |

**Only one additional invocation reaches the floor at n=3 versus n=20.** The ~98
that sit there do so at every n, because their reference is genuinely
perturbation-insensitive (discrete/index outputs — argmax, argmin). Median
tolerance shrinkage is **2.4% at n=5** and **6.1% at n=3**. The collapse is a
single outlier, not a systematic degradation.

---

## Time saved

Actually-run arms, `KCC_CHECK_TIMING=1` (CUDA serialised — shares meaningful,
absolutes are upper bounds):

| n | perturbation path | all checks | checker wall | vs n=20 |
|---:|---:|---:|---:|---:|
| 3 | 1.12 s | 4.61 s | 6818 ms | **−19.5%** |
| 5 | 1.28 s | 4.67 s | 6830 ms | **−19.4%** |
| 10 | 1.83 s | 5.34 s | 7422 ms | −12.4% |
| 15 | 2.19 s | 5.44 s | 7485 ms | −11.7% |
| 20 | 2.95 s | 6.34 s | 8474 ms | — |
| 40 | 4.94 s | 8.39 s | 10457 ms | +23.4% |

**n=5 and n=3 are indistinguishable (−19.4% vs −19.5%)** — below ~5 samples the
fixed per-call cost (base reference call, candidate call, quantile, device
transfer) dominates and further reduction buys nothing.

Translated to the 60.8 s warm corpus run (perturbation runs in
`your_checker (full)` 5.28 s + `(numeric only)` 4.61 s = 9.89 s):

| n | corpus saving |
|---:|---:|
| 10 | 1.23 s = **2.0%** |
| 5 | 1.92 s = **3.2%** |

---

## Correcting the previous round's projection

`verification_runs/check_timing_2026-08-25/FINDINGS.md` closed by naming this as
potentially "an order of magnitude above anything available from removing a
check", reasoning from *46% of check time*. **That was wrong, and the error was
the denominator:** the checker is only ~16% of the corpus run, so 46% of it is
~7%, and a 40% cut to that is ~3%. Measured:

| lever | corpus saving |
|---|---:|
| shipping a warm Triton cache | **75%** |
| `n_samples` 20 → 5 | 3.2% |
| `n_samples` 20 → 10 | 2.0% |
| removing `check_weight_magnitude` entirely | 1.9% |

`n_samples` is the same ~2–3% class as everything else at check level, not a
different order. The cache remains the only large lever found.

---

## Recommendation

**No default change is recommended on this evidence, and the reason is the
evidence, not the size of the win.**

A corpus where 94% of the relevant invocations have zero margin cannot
distinguish a safe `n_samples` from an unsafe one. The curve being flat from
n=1 to n=40 is not a licence to set n=1; it is a measurement that returned no
information. Setting the default from it would be inferring safety from a
control that never fired — the failure mode this project has hit repeatedly
(SESSION_HANDOFF §5 instance 12).

If `n_samples` is to be reduced anyway, **n=10 is the defensible choice** —
−12.4% checker time, median tolerance shrinkage 1.1%, no additional floor
collapses versus n=20, and catch safety strictly better than at n=20. n=5 saves
more (−19.4%) and still shows nothing, but its median shrinkage is 2.4% and it
sits closer to the regime where the single observed collapse occurred.

**What would actually settle it:** a corpus of *near-miss* candidates — kernels
wrong by a margin comparable to `adaptive_tol` rather than by 400x. The current
mutants fail by a median ratio of **461x**; the closest fails by 1.45x. Until
inputs exist that land near the boundary, any `n_samples` between about 5 and 40
is indistinguishable on evidence, and 20 is as defensible as anything else.

> **EFFECTIVE-SAMPLE ACCOUNTING, 2026-08-28 — see
> `../theory_closure_2026-08-28/FINDINGS.md` §3.** The 854 banked 40-sample
> sensitivity vectors in `arms/CURVE_n40.json.gz` contain **513 bit-distinct**
> vectors: under `KCC_ABLATION_SEED` per-check reseeding, 23 adversarial
> (op, check) classes are one (input, deltas) draw replayed 6–22× (list in
> the closure round). Verdict-level results (catch/FP at every n, the flat
> curve) are unaffected — they are exact per-record replays either way — but
> any DISTRIBUTIONAL statement over "the 854 invocations" (including the
> 2026-08-27 theory audit's H2 analysis) is really over 513 independent
> draws with the collapsed classes over-weighted by their replica counts.
