# The Fréchet α = 2.02 is neither derivable nor measured: the two-point estimator that produced it has a sampling spread covering [0.9, 18] under every plausible tail, the arms' tails are statistically indistinguishable, and 70 exceedances cannot even separate Pareto from lognormal — the defensible statement is α ∈ [2.3, 3.7]

**Analyzed 2026-08-28** from the banked pre-fix contention samples
(`../forkserver_2026-08-21/race_rate.jsonl`, n = 2765 non-null ratios:
1400 spawn / 1365 forkserver). Probe `probes/alpha_structure.py`, log
`data/run1.log`. This answers the round's lead — *is the α = 2.02
contention-tail exponent derivable from the launch/scheduling structure,
or is 2.02 coincidental?* — with the third option the question didn't
offer: **2.02 is not a stable property of the data at all**, and the
contention_tail round's deepest-tail claim is corrected below.

## 1. What 2.02 actually was

The 2026-08-28 contention_tail round derived α = 2.02 from
ln(2765/560)/ln(51.24/23.26) — max growth **across two different
samples** (23.26 is the max of the separately-instrumented 560-execution
stream; 51.24 of this one). Computed *within* this sample (record-order
prefix of 560), the same estimator gives α̂ = **1.26**: the first 560
records' max here is 14.4, not 23.26. Two draws of the estimator, 2.02 and
1.26 — the first hint that it measures noise.

## 2. Q2 — the estimator has no resolving power (the decisive negative)

Semi-parametric null: resample the observed body below p99, attach a
Pareto tail with true index α₀ above it, and compute the two-point
estimator's sampling distribution (4000 replicates per α₀):

| true α₀ | α̂ p5/p50/p95 | P(α̂ ≤ 1.26 observed) |
|---|---|---|
| 2.0 | 0.70 / 1.79 / 11.9 | 0.30 |
| 2.5 | 0.89 / 2.30 / 17.7 | 0.17 |
| 3.0 | 1.06 / 2.66 / 17.0 | 0.10 |
| 3.5 | 1.24 / 3.11 / 22.7 | 0.05 |

The p5–p95 band spans **an order of magnitude** at every α₀, and the
observed value is unremarkable under all of them. A ratio of two sample
maxima at n ratio ~5 carries ~one effective tail draw of information; "the
deepest tail behaves like α ≈ 2 — infinite variance" was resting on that
one draw. **Retracted** (correction appended to the contention_tail
FINDINGS).

## 3. Q1 — the arm-mixture hypothesis fails too

The two banked outliers (51.24, 23.26) are both spawn-arm, suggesting the
pooled deep tail might be the spawn arm's. Per-arm Hill with bootstrap
95% CIs says no:

| k | spawn | forkserver |
|---|---|---|
| 50 | 3.40 [2.42, 4.76] | 3.64 [3.08, 4.79] |
| 100 | 2.69 [2.26, 3.45] | 3.04 [2.28, 3.88] |
| 200 | 3.13 [2.72, 3.73] | 2.55 [2.29, 2.96] |

CIs overlap at every k; the ordering even flips between k = 100 and 200.
The spawn-arm maxima are a 2-of-2 coincidence on ~1400 draws per arm, not
a tail difference. (Consistent with the forkserver round's own bulk
comparison, which found the arm distributions matching.)

## 4. Q3 — the data cannot pick a family, so no derivation is testable

Exceedance likelihoods above p95 (n ≈ 70 per arm): Pareto vs
truncated-lognormal log-likelihood differences of **+1.8 (spawn) and −0.9
(forkserver)** — coin-flip territory in both arms. Any
scheduling-structure derivation would have to predict not just an index
but a family, and 70 exceedances per arm cannot falsify either family,
let alone an index within one. This bounds what *any* α derivation could
claim from this dataset; a derivation attempt would be unfalsifiable with
the data at hand and was therefore not manufactured (candidate mechanisms
— length-biased burst sampling, M/G/1 busy periods — each predict
regularly-varying tails with indices tied to burst-length distributions
nobody has instrumented on these VMs).

## 5. What survives, stated positively

- **Regular variation itself is not in doubt** for the operational
  conclusion: pooled Hill is stable at 2.9–3.7 across k = 25–200 with
  bootstrap CIs inside [2.3, 7.4], and every α < ∞ gives unbounded
  expected max growth ~ (N)^{1/α}. The pre-fix construction remains
  threshold-unsafe at any constant — the contention_tail round's *core*
  claim stands.
- The quantitative production-year extrapolation weakens honestly: at
  α ∈ [2.3, 3.7] the expected max over 10⁵ executions is ~p99·(10³)^{1/α}
  ≈ **21–100**, not the single 170 quoted at α = 2. Still far above any
  tightened threshold; the operational table (post-fix FP surface,
  threshold dead zones) is unaffected.
- **Answer to the lead question: 2.02 is coincidental** — an artifact of a
  cross-sample two-point construction whose sampling spread covers
  [0.9, 18] under every tail the data admits. Nothing about the
  launch/scheduling structure selects it.

## Limits

- The Q2 null resamples the observed body; a body-tail dependence
  (contention episodes correlating consecutive records) would widen, not
  narrow, the estimator's spread — the negative is conservative.
- Hill itself assumes exact Pareto beyond the k-th order statistic; its
  2.9–3.7 range is a regularly-varying summary, not a certified index
  (Q3 is the reason no tighter statement is offered).
- n = 2765 pre-fix records from one VM class; all statements are about
  that environment.

## Reproduce

```bash
.venv/bin/python probes/alpha_structure.py   # ~2 min, deterministic
```
