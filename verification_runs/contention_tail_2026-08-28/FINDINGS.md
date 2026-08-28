# The contention-ratio heavy tail is Fréchet-domain (α ≈ 2–3.5, max ~ n^{1/α}) — pre-fix no threshold was safe at any value; post-fix the FP surface starts at ~1.2–1.3× and everything ≥ 2× is empirically dead

**Modeled 2026-08-28** from the banked 1d-investigation samples
(`../forkserver_2026-08-21/race_rate.jsonl`, n = 2765 pre-fix
reference-vs-itself ratios under 4-way T4 contention;
`race_rate_POSTFIX.jsonl`, n = 140 post-fix). Probe
`probes/tail_model.py`, log `data/tail_model.log`. Nothing in production
changes in this round — the shipped guard (interleaved best-of-5 min,
threshold 10×, delegation_fix_2026-08-21) is untouched; this answers the
open modeling question the theory audit ranked "strongest undocumented
statistical result in the repo".

## 1. The pre-fix phenomenon, now modeled

Mechanism: the old construction timed a 10-call candidate block, then a
10-call reference block; a scheduling stall of excess S in one block gives
ratio ≈ 1 + S/T₀. Under contention S is **regularly varying**, so the
ratio inherits a power-law (Fréchet-domain) tail and the sample maximum
grows like n^{1/α} without bound:

- **Hill estimator** on the top order statistics: α = 2.9–3.7 (k = 25–200,
  the usual Hill instability across k, all firmly in the heavy-tail range).
- **The banked two-point max growth is the cleanest measurement**:
  max 23.26 @ n=560 → 51.24 @ n=2765 gives α = 2.02, and a Fréchet
  extrapolation anchored at p99 with α = 2 reproduces both maxima
  (predicted 24.3 and 53.9). The deepest tail behaves like α ≈ 2 —
  infinite variance.

This *derives* the banked one-liner "no constant derived from a finite
sample is provably safe": with a regularly-varying tail, the expected
maximum in N executions is ~p99·(0.01·N)^{1/α}; at α = 2 a production year
of 10⁵ executions expects a max of order 170. Every fixed threshold is
eventually crossed at a computable rate — that is a property of the
construction, not of any particular sample.

The FP surface itself was brutal and *flat*: P(ratio ≥ t) =
30% / 29% / 25.5% / 7.7% / 1.2% at t = 1.3 / 2 / 3 / 5 / 10 — a quarter of
reference self-comparisons "sped up" 3× by lottery. Pre-fix, tightening
was not merely risky; there was no usable threshold anywhere.

## 2. The post-fix surface — where FPs would appear if the threshold moved

The shipped estimator (interleave + min-of-5) removes the mechanism: a
stall lands on both arms of an interleaved round, and inflating the min
requires every round slow on one side only. Measured (n = 140): p50 1.00,
p90 1.06, p99 1.16, **max 1.22**; top ratios
1.215, 1.159, 1.151, 1.149, 1.117…

**The FP surface now begins at t ≈ 1.2–1.3**: P(ratio ≥ 1.15) = 2.1%,
zero events ≥ 1.3. For any threshold t ≥ 1.3 the observed rate is 0/140
(95% binomial upper bound ≈ 2.1%); the current 10× threshold carries 8.2×
of margin above the largest ratio ever observed post-fix.

Answer to the item's specific question, in usable form:

| tightened threshold | post-fix evidence |
|---|---|
| ≥ 2× | dead zone — 1.64× above the observed max; no fire in any sample |
| 1.5× | still above every observation, margin 1.23× — plausible but uncertified |
| 1.3× | boundary — upper edge of the observed support |
| ≤ 1.2× | inside the measured surface (~2% fire rate at 1.15) |

Caveats that keep this honest: n = 140 post-fix cannot certify sub-2%
rates at production scale, and the min-of-5 independence bound
(q(t)⁵ using pre-fix per-round exceedance ≈ 0.3%) *underpredicts* the
measured 2.1% at t = 1.15 — rounds are positively correlated through
shared contention episodes, so tail extrapolation by independence is
unsafe and is not used for the table above; the table is purely
empirical support. Certifying a tightened threshold (e.g. 2×) at
production N needs the same powered ~900/arm re-run the forkserver
default-flip decision is already queued behind — one run answers both.

Structural note that bounds the incentive: a genuinely delegating kernel
calls the reference and times at ratio ≈ 1.0 — **below any threshold** —
so tightening buys detection only of precomputed-output ghosts, which sit
orders of magnitude away (a uniformly-100× ghost is caught at either 10×
or 2×). The value of tightening is small by construction; the cost of
over-tightening (reference FPs at ~1.2) is the thing this surface now
locates precisely.

## 3. Consistency notes

- This dataset's pre-fix p99 is 10.24; the separately-instrumented
  560-execution stream quoted p99 = 11.45 — same phenomenon, different
  sample, both inside the Hill-α range's expected sampling spread.
- The two most extreme pre-fix outliers (51.24, 23.26) are both spawn-arm,
  as the forkserver round recorded; pooling arms is justified because the
  bulk distributions match (that round's own analysis).

## Limits

- Post-fix n = 140 (plus the 70/arm directional check, consistent);
  everything about t < 2 is support-boundary reasoning, not a certified
  rate.
- Hill α is k-sensitive (2.9–3.7); the α ≈ 2 statement rests on the
  two-point max growth and the extrapolation fit, and the operational
  conclusion (unbounded max growth) holds for any α in the range.
- CPU wall-clock timing on a shared VM; the model treats stalls as iid
  across executions, which the max-growth fit supports but does not prove.

## Reproduce

```bash
.venv/bin/python probes/tail_model.py
```

---

> **CORRECTION, 2026-08-28 (later the same day) — see
> `../contention_alpha_2026-08-28/FINDINGS.md`.** The α ≈ 2 deepest-tail
> claim does not survive scrutiny and is RETRACTED:
>
> 1. The two-point max-growth estimate mixed two different samples
>    (23.26 is the 560-execution stream's max; 51.24 this stream's).
>    Computed within this stream (prefix max 14.4 at n = 560) the same
>    estimator gives α̂ = 1.26.
> 2. Under a semi-parametric null with ANY true α₀ ∈ [2.0, 3.5], the
>    two-point estimator's p5–p95 sampling band spans an order of
>    magnitude and covers both observed values. It has no resolving
>    power at this n; "α ≈ 2, infinite variance" rested on effectively
>    one tail draw.
> 3. Per-arm Hill CIs overlap at every k (the both-maxima-are-spawn
>    observation is a 2-draw coincidence), and 70 exceedances per arm
>    cannot even separate Pareto from lognormal.
>
> What stands: regular variation with pooled Hill α ∈ [2.3, 3.7], the
> qualitative unbounded-max-growth conclusion, and every operational
> number in §2 (the post-fix FP surface is empirical and untouched). The
> production-year expected-max figure softens from "~170" to a range
> ~21–100. §1's "the deepest tail behaves like α ≈ 2 — infinite
> variance" and the α = 2.02 anchoring should not be cited.
