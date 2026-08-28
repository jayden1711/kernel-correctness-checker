# Theory audit — one new result (the scan family's exact tolerance law), one lemma proved behind an in-code constant, one derived negative, one partial unification, one clean negative

**Investigated 2026-08-27 on the Apple-silicon dev machine. No GPU was used and
none was needed: every comparison is against already-banked GPU measurements**
(`phase1_derivations_2026-08-27/native_run/`, `adaptive_tol_theory_2026-08-25/
native_run/`, `n_samples_curve_2026-08-25/arms/CURVE_n40.json.gz`). Probes in
`probes/`, run logs and derived data in `data/`, the full candidate inventory in
`INVENTORY.md`. **Nothing in the checker was changed.**

---

## Verdict up front

| hypothesis | outcome |
|---|---|
| **H1** — exact law for the scan family's tolerance from the Jacobian **Gram matrix** (the one family M3 misses) | **CONFIRMED — this is the eighth result.** Zero fitted constants; 4/4 operators within 2.2% of banked GPU tolerance (worst family z = −1.1); reproduces the +24.7% M3 residual as the independence-vs-Brownian gap (predicted 1.231, measured ~1.247); survives three out-of-sample shape predictions and a four-decimal input-invariance test. §1 |
| **H5** — the un-derived in-code claim `CV ≤ 0.7555` (`scope_detect.py`) | **DERIVED AND VALIDATED.** 0.7555 = √(π/2−1), the half-normal CV. Sharp ceiling supported by a 4 000-structure adversarial search (every apparent violation collapses to the ceiling under precision, all extremal structures rank-1); a weaker ceiling √(π/2) is provable outright. The banked "violations" are exactly what the lemma predicts. §2 |
| **H2** — the open question from the n_samples round: margin → detection-probability formula | **Mixed, mostly negative.** An exact order-statistic bracket exists and is near-perfect where the parent is continuous; every *estimable* version of it is falsified in the transition zone (split-half median error 0.11); the saturation regime is now **derived** — worst P(miss) over every caught mutant ≤ 1.9e-3 — which retro-derives the flat catch curve. The formula's failure set is exactly the taxonomy's fp32-quantization category, re-derived independently. §3 |
| **H3** — do the three exception categories share deeper structure? | **Partial.** Categories 1 and 2 unify cleanly as the absolute and relative arms of one resolvability criterion (15/15 flagged, 2/172 false flags — both of which turn out to be genuine unlisted floor-sitters). The m=1 category does **not** separate on those axes (8/18 flagged): it remains a genuinely different phenomenon, but it is *not* cleanly outside the resolution regime either. §4 |
| **H4** — provable completeness for the adversarial search | **NO, and none is available from this construction.** The only provable property is the soundness of the hit invariant. By-product, proved and verified: the diversity penalty is **inert at shipped defaults** — 20 000/20 000 pools identical to plain beam search whenever proposals are valid — plus two docstring/code mismatches in the scorers. §5 |
| Inventory | 136 candidates catalogued (`INVENTORY.md`); everything else cross-references to the seven documented results, standard facts, or magic constants; the open items already known (skip_boundary_tiles, defect-screen overlap, wrong_dtype fp16 condition) are listed there, not re-litigated here. |

---

## 1. H1 — the eighth result: the scan family's tolerance obeys an exact Gram-matrix law, and it is the Brownian reflection law

### 1.1 The claim, and why it is derivable rather than fitted

For an **exactly linear** operator, the sensitivity sample is exactly
`s = σ‖Jg‖_∞ = σ·max_i |⟨J_i, g⟩|` with `g ~ N(0, I)` — no linearisation
error at all. The vector `(⟨J_i, g⟩)_i` is Gaussian with covariance the **Gram
matrix `JJᵀ`**, so the *entire distribution* of `s`, and hence the distribution
of `adaptive_tol = 3·q95_n(s)`, is a functional of `JJᵀ` alone. M3 keeps only
the diagonal (the row-norm profile) and assumes the rest away; the scans are
the family where the rest is everything.

For a prefix scan, row `i` of `J` is `(1,…,1,0,…,0)` with `i+1` ones, so
`(JJᵀ)_{ik} = min(i,k)+1` — the covariance of a **random walk**. Therefore

```
    y  =  tol/(3σL)  =  q95_n( max_{r≤R, k≤C} |W_k^{(r)}| ) / √C
```

with `W` a standard Gaussian walk, `R×C` the tensor shape — and in the
continuous limit `max_k|W_k|/√C → max_{t≤1}|B_t|`, whose CDF is the classical
reflection-principle theta series. Everything on the right side is computable
in advance from the shape alone: **no kernel, no probe, no fitted constant.**

### 1.2 Validation against the banked GPU measurements

Predictions from `probes/scan_brownian.py` / `scan_precise.py` (NREP = 1500
replications of the exact 40-sample estimator, torch-quantile convention
replicated exactly); measurements are the 24 banked Phase-1 invocations,
`y = tol/(3σ·L_closed)`:

| operator | measured (n=6) | predicted | meas/pred | z of the 6-inv mean |
|---|--:|--:|--:|--:|
| `cumsum` | 3.4706 ± 0.267 | 3.4620 ± 0.166 | **1.0025** | +0.13 |
| `cumsum_reverse` | 3.4922 ± 0.206 | 3.4621 ± 0.167 | **1.0087** | +0.44 |
| `cumsum_exclusive` | 3.3830 ± 0.073 | 3.4565 ± 0.165 | 0.9787 | −1.09 |
| `masked_cumsum` | 3.2205 ± 0.155 | 3.2907 ± 0.160 | 0.9749 | −1.07 |

All 24 per-invocation z-scores lie within ±2.78 — for 24 draws the expected
worst is ≈2.4, so nothing is out of family. `masked_cumsum` is the structurally
distinct case (a Bernoulli-masked, time-changed walk with per-row totals) and
the sim handles it by drawing masks from the spec's own distribution and
normalising by each draw's own closed-form `L`, exactly as the checker
normalises by the invocation's own mask.

### 1.3 The M3 residual is explained, quantitatively

Simulating M3's own orthogonal-rows assumption in the same harness reproduces
the banked `y_M3` to **0.02%** (4.2442 sim vs 4.2435 banked) — so the
simulator is faithful — and the ratio of the two laws is

```
    orthogonal / Brownian  =  4.2442 / 3.4476  =  1.231
```

against the GPU-measured scan-family M3 residual of **+24.7%** (per-op banked
ratios 1.229/1.220/1.255/1.257). The documented mystery residual of the M3
story is therefore not merely "correlation, sign as expected" — it is the
**computable gap between the independent-max law and the Brownian-max law**,
and it lands within ~2% of the measured value for every scan operator.

### 1.4 Falsification attempts, all survived

- **Input invariance (the law says y depends on J only).** Across the banked
  adversarial variants — `primary`, `alternating_signs`, `large_then_tiny`
  (σ spans 1e-3 to **5060**), `all_ones` — the measured y is identical to
  **four decimal places** (e.g. cumsum: 3.2569/3.2569/3.2568/3.2565). Shared
  per-(op,invocation) RNG makes this an exact test of "same J ⇒ same y": five
  wildly different inputs, one tolerance.
- **Out-of-sample shape.** The `non_power_of_two` variants run C=333, a shape
  the derivation never saw: measured vs predicted z = **+0.47 / −0.46 / +0.30**
  for cumsum / reverse / exclusive.
- **The continuous closed form and its discrete correction.** Reflection series
  + exact order-statistic integration gives y_∞ = 3.4818; the finite-C law
  approaches it as `y_∞ − c/√C` with `c ≈ 0.50` measured across C = 64…2048
  (gap·√C = 0.51, 0.50, 0.47, 0.52, 0.40, 0.65, sem 0.05–0.27). The proximity
  of c to Siegmund's boundary-correction constant 0.5826 is noted as
  suggestive; it is **not claimed** — the setting (two-sided max, row-max of
  64 copies, a q95 rather than a mean) differs from the classical theorem.

### 1.5 Scope and consequence

The general statement — *y is a functional of the Gram matrix; M3's error is
exactly the off-diagonal it discards* — is proved by the linearisation theorem
already banked (§2.1 of the 2026-08-25 round, <0.1% defect) plus elementary
Gaussian algebra; the scans are the family where it has a **closed form**. The
practical consequence: the one family M3 mis-predicts can now be predicted
exactly, from shape alone, with no simulation of correlations — extending
zero-fitted-constant tolerance prediction to the full 62-operator corpus'
worst family. Flash attention's +17% (rows sharing a softmax denominator) is
the obvious next Gram-matrix target; it has no banked Q/K/V and is out of
scope here. *(DONE 2026-08-27:
`../adaptive_tol_theory_2026-08-25/attention_gram/ATTENTION_GRAM.md` — the
law holds on all 36 banked + 108 fresh GPU measurements; the "+17%" turned
out to be mostly single-draw noise around a +3–4% true correction, and the
out-of-sample run uncovered a real padded-column bug in the flash/sdpa
reference kernels at N % 32 ≠ 0.)*

---

## 2. H5 — the CV ceiling: an in-code constant is actually a sharp lemma

`scope_detect.py` asserts, without derivation: *"CV ≤ 0.7555 is a correct
property of the linear regime but is not a usable screen."* This audit derives
and validates it.

**Lemma (numerically supported; weak form provable).** For any centred
Gaussian vector `(X_1,…,X_m)` — i.e. for `s = max_i |⟨J_i, d⟩|` in the linear
regime — the coefficient of variation of `max_i |X_i|` satisfies
`CV ≤ √(π/2 − 1) = 0.75551`, **with equality iff the vector is rank-1** (then
the max is a single half-normal). The weaker ceiling `CV ≤ √(π/2) = 1.2533`
follows outright from Gaussian Poincaré (`Var ≤ σ*²`) plus
`E[max_i|X_i|] ≥ σ*√(2/π)`; the sharp constant is the conjecture the code was
already relying on.

**Adversarial search** (`probes/taxonomy_cv.py` + `cv_refine.py`): 4 000
random covariance structures across five families built to break it (near
rank-1 + noise, heavy-tailed row norms, anti-correlated pairs, dominant row +
correlated small rows). Stage-1 estimates reached 0.7842 — pure selection
noise: re-evaluated at 4×10⁶ draws, **every top candidate collapses to
0.7547–0.7560**, and all of them are near-rank-1, i.e. *at the conjectured
equality case*. Controls: rank-1 = 0.7553–0.7557 (at the ceiling), two equal
orthogonal rows 0.534, eight iid rows 0.295.

**The banked "violations" confirm rather than refute it.** Of 290 banked
linear-regime invocations (ladder defect < 5%), five exceed 0.7555:

- Four are `nll_loss` (0.771–0.892). `nll_loss` is *exactly* rank-1 (a linear
  gather to a scalar), so its parent sits **at** the ceiling — and the
  40-sample CV of a rank-1 parent exceeds 0.7555 **42% of the time**
  (q95 = 0.890, measured by simulation). The four values are that sampling
  distribution, not a counterexample.
- One is `flash_attention/wrong_causal_mask` at CV = **5.496** with ladder
  defect 0.0012. That combination is impossible for a Gaussian max — which
  makes it a **certificate of non-linearity that the ladder cannot see**: the
  t-ladder tests linearity *along one ray* (any positively-homogeneous,
  piecewise-linear response passes it exactly), while CV integrates *across
  directions*. The two screens the scope round treated as redundant-in-spirit
  are provably complementary, and this invocation is the witness.

This also gives the m=1 diagnostic-blindness category a second derivation:
at m=1 the parent is *forced* to the rank-1 equality case, so the sensitivity
sample carries only its scale and no shape information — the q95/RMS
degeneracy of the documented result 7, reached from the CV side.

---

## 3. H2 — margin → detection probability: exact bracket, un-estimable transition, derived saturation

The n_samples round left open whether a formula relates a mutant's distance
from correct behaviour to its detection probability. Resolution, in three
parts (`probes/margin_detect.py`, validated against all 854 banked 40-sample
GPU sensitivity vectors):

**(i) The exact bracket exists and is trivial to state.** With
`u = F(max_err/3)` (parent CDF of the sensitivity) and the shipped
q95-blend lying between the top two order statistics,

```
    u^n   ≤   P(detect)   ≤   u^n + n·u^(n−1)·(1−u)        (P = 0 if max_err ≤ 1e-6)
```

Where the parent is continuous, resampling the real vectors reproduces the
bracket midpoint to **median 0.0006** at n = 20 (903 (invocation, threshold)
points; n = 5/10 medians 0.026/0.008).

**(ii) No estimable version survives the transition zone.** The split-half
test (Gumbel parameters from samples 0–19, detection resampled from disjoint
samples 20–39) gives **median error 0.112, p90 0.474**. This is structural,
not fixable: `δP ≈ n·u^{n−1}·δu`, so predicting P to ±10% near the boundary
needs the parent CDF to ±0.5% at its 93rd percentile — precision 20–40 samples
cannot deliver. The previously-validated Gumbel model predicts the *tolerance
ratio curve* (a location statistic) well and the *detection probability* (a
tail-exponent statistic) badly, and both facts are consistent. **The open
question's answer for the regime where it matters is: no such usable formula,
for a quantified reason.**

**(iii) What is derivable is the saturation — the previously-observed flat
curve is now a theorem about this corpus.** Applying the bracket to every live
invocation: the worst miss-probability bound over all caught mutant
invocations is **P(miss) ≤ 1.94e-3** (the gelu near-miss at margin 1.46×; every
other caught invocation is below 1e-6), and the single live reference
invocation (`frobenius_norm/adversarial_dominant_outlier`, margin 8.3e-4) has
P(FP) ≈ 0 to double precision. The 40/40-catch, 0/200-FP curve being flat in
`n` from 1 to 40 is therefore not just an observation — it is implied, with
explicit bounds, by the margins the corpus happens to contain.

**A coherence bonus:** the bracket's failure set — the points where resampled
truth and formula disagree by up to 1.0 — is *exactly* the set of invocations
with a quantized (tied, few-unique-values) sensitivity parent: 94 of 787
usable invocations, the fp32-floor sitters. The exception taxonomy's
category 2 re-emerges, unprompted, as the violation set of a continuity
assumption made for an unrelated purpose. (Continuous-parent points off the
knife edge: median 0.0006; quantized points: p90 = max = 1.0.)

---

## 4. H3 — the taxonomy is two arms of one criterion, plus one genuinely different axis

Claim tested: categories 1 and 2 are the **absolute** and **relative** arms of
a single resolvability criterion,

```
    exception   ⟺   min( (s_med/ulp(‖f‖_∞)) / 32 ,   tol / 1e-6 )  ≤  1
```

with 32 the scope round's validated median-s/ulp threshold. Measured on the
205 banked invocations where the median statistic is recoverable
(`probes/taxonomy_median.py`; a first pass in `taxonomy_cv.py` used the banked
**min**-based `s_over_ulp` and its m=1 flags are an artifact of that — kept,
labelled, as the record of the wrong turn):

| group | n | flagged | note |
|---|--:|--:|---|
| category 1 (absolute floor) | 5 | **5** | and their s_med/ulp is 2.5–6.0 — they sit in **both** arms |
| category 2 (fp32 floor) | 10 | **10** | |
| everything else | 172 | 2 | both `flash_attention/multi_tile_rescaling` (s_med/ulp = 2.0 and 16.5) — a genuine fp32-floor sitter the taxonomy's variant list never named; a correct flag, not a false one |
| **m=1 losses** | 18 | **8** | s_med/ulp 12–25: *near* the relative threshold, not far inside |

So: **the two floor categories unify** — one criterion, two arms, 15/15 with
no misses and no false flags once the two `multi_tile_rescaling` invocations
are recognised as true members. But the hoped-for clean picture — "m=1 far
inside the allowed region on both axes, hence a third orthogonal phenomenon" —
is **not what the data shows**: 8 of 18 m=1 invocations trip the relative arm.
The m=1 category is mechanistically distinct (the rank-1/shape-ratio
degeneracy of §2 and documented result 7 — a property of output *dimension*,
not of scale), yet its instances also hover near the resolution boundary,
because a scalar loss output of order 1 leaves its σL-sized response only
~10–60 ulp of headroom. The honest summary: **two categories are one
phenomenon; the third is a different phenomenon that partially co-occurs with
the first.** Not the clean unification hypothesised, and reported as such.

---

## 5. H4 — no completeness property for the adversarial search; the diversity penalty is provably inert

The search (`verification/adversarial_search/`) is an LLM-proposal loop over
an unbounded input space with heuristic additive scoring; there is no
partition of the space, no acceptance guarantee, and the diversity mechanism
keys on the **self-reported** `predicted_failure_mode` string. No completeness
property — even a weak coverage one — is derivable from this construction, and
none should be claimed. The one provable property is **soundness of the hit
invariant** (`coordinator.py:21-26`): every confirmed hit has machine-verified
reference validity, checker failure, and allclose gap. That is real, already
documented, and the correct thing to cite.

**By-product theorem, verified by enumeration** (`probes/diverse_inert.py`):
with the shipped defaults (λ = 3.0, beam_width = workers = 4), any
reference-passing proposal scores ≥ 10 while the maximum diversity penalty
within a beam is 3·3 = 9, and the selection loop never re-ranks and back-fills
skipped candidates in rank order. Hence `DiverseBeamStrategy` ≡ plain
`BeamSearchStrategy` on every pool of valid proposals — **20 000/20 000 random
pools identical**; divergence occurs only on pools containing broken proposals
(313/20 000), i.e. the mechanism can only ever discriminate among proposals
that already failed. The "diverse" strategy, as shipped, does not diversify
anything that matters.

Two docstring/code mismatches found while proving it, verified directly:
`beam.py` documents −2 per errored mutant, `greedy.py` documents +2 per
no-gap catch and −3 per errored mutant — none of the three terms exists in
the corresponding `score()` implementations. Recorded, not fixed: scoring
changes alter search behaviour and belong in their own change.

---

## 6. What was looked at and did not become a result

- **The full inventory** (`INVENTORY.md`): 136 candidates. Beyond the five
  hypotheses above, the remainder are (a) restatements of the seven documented
  results, (b) textbook facts used as property checks, or (c) magic constants
  and stated-but-unproven arguments now catalogued with file:line for future
  rounds. The known open items (skip_boundary_tiles masking condition, the
  falsified defect-screen separation, wrong_dtype's fp16-exactness condition,
  l1norm/l2norm eps-vs-variance) were confirmed still open and **not**
  re-investigated — each needs GPU work or new mutants this pass could not do.
- **B.3 chaining** (M3 × the n-curve model) remains not validated end-to-end;
  it needs nothing banked here and stays flagged in the generalization round.
- **A stale in-code comment** contradicting a banked measurement
  (`structural_l.py:350-352`, the refuted m=1 prediction) and an unretracted
  causal claim (`SOTA_CHECKS_REGISTRY.md` fixed-tolerance-FP mechanism vs the
  autokernel audit) are flagged in `INVENTORY.md` as documentation defects.

---

## 7. Reproduce

```bash
cd verification_runs/theory_audit_2026-08-27
PY=../../.venv/bin/python
$PY probes/scan_brownian.py    # H1: exact law vs banked GPU y, M3 baseline, C=333, closed form
$PY probes/scan_precise.py     # H1: NREP=1500 precision pass + discrete-correction ladder
$PY probes/margin_detect.py    # H2: bracket sweep, split-half, saturation bounds
$PY probes/taxonomy_cv.py      # H3 (first pass, min-statistic artifact) + H5 banked cv + search
$PY probes/taxonomy_median.py  # H3 corrected: median-statistic two-arm criterion
$PY probes/cv_refine.py        # H5: high-precision re-evaluation of search candidates
$PY probes/diverse_inert.py    # H4: diversity-penalty inertness enumeration
```

Run logs are banked in `data/*.log`; derived predictions in
`data/scan_brownian_preds.json`, `data/scan_precise.json`.

---

## 8. Limits

- **H1's measurement side is 24 + 14 banked invocations at two shapes
  (C = 512, 333), one GPU, one seed regime.** The prediction curve in C is
  nearly flat (3.42–3.47), so the shape test has limited discriminating power;
  the strong tests are the four distinct Gram structures, the input-invariance
  identity, and the M3-gap match. `cumsum_exclusive` and `masked_cumsum` both
  sit ~2% low (z ≈ −1.1 each) — within noise individually, but the shared sign
  is worth one look if the family is ever re-measured with more invocations.
- **H1 claims exactness only for exactly-linear operators.** For nonlinear
  operators the Gram-matrix statement holds to the linearisation defect, and
  the attention family (the next-worst M3 residual) is untested — no banked
  Q/K/V.
- **H5's sharp constant is a conjecture with numerical support**, not a proof;
  only the √(π/2) ceiling is proved here. The search covered m ≤ 11 and five
  structure families; equality cases beyond rank-1 were not sought
  analytically.
- **H2's split-half test** uses n = 10 subsets of 20 samples (the only
  non-circular split the banked vectors allow); its error magnitudes are
  specific to that n, though the amplification argument is general.
- **H3's median statistic is recoverable for only 205 of 368 candidate
  invocations** (the rest bank a min-based field or have a zero minimum);
  `pass2.jsonl`'s adversarial invocations could not be included at all. The
  two-arm result should be re-checked if median s/ulp is ever banked corpus-wide.
- The m=1/CV connection in §2 and the category-3 overlap in §4 describe the
  same six operators from two angles; neither constitutes a validated new
  *category*, and the documented taxonomy's wording stands.

---

> **EFFECTIVE-SAMPLE NOTE, 2026-08-28 — see
> `../theory_closure_2026-08-28/FINDINGS.md` §3.** H2's "all 854 banked
> 40-sample GPU sensitivity vectors" (and derived counts such as the
> 94-of-787 quantized-parent set) are drawn from a bank in which 23
> adversarial classes are bit-identical replicas of one draw under the
> ablation reseed; the bank holds 513 bit-distinct vectors. H2's
> conclusions are aggregate and directionally unaffected, but its
> denominators over-weight the collapsed classes. Also resolved since this
> audit: §1.5's open B.3 chaining is CLOSED (same closure round, §1) — the
> M3/Gram structural parent predicts the full n-curve with per-invocation
> z ~ N(0,1), and the scan family's curve needs the Brownian parent exactly
> as H1 would predict.
