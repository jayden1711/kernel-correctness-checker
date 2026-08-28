# Theory closure round: B.3 chained and closed, the kink margin derived, the reseeding collapse counted, and three doc defects fixed

**Executed 2026-08-28 on the Apple-silicon dev machine. No GPU was used and
none was needed**: every comparison is against already-banked GPU
measurements (`../adaptive_tol_theory_2026-08-25/native_run/gpu_native.jsonl`,
`../phase1_derivations_2026-08-27/native_run/phase1_native.jsonl`, the
CURVE/scope/gram corpus arms). Probes in `probes/`, outputs in `data/`.
**One checker file was edited (comment-only: `structural_l.py` docstrings);
no check logic changed anywhere.**

## Verdict up front

| item | outcome |
|---|---|
| **1. B.3 chaining** (M3 parent × p_eff(n) → probe-free n-curve) | **CLOSED — the chain is validated end-to-end, and the shipped n = 20 default is now inside the validated regime.** 228/228 native invocations replayed bit-consistent; the structural parent's direct curve matches the measured aggregate to **0.1% at n = 20** (0.9841 vs 0.9852) and per-invocation deviations are **z ~ N(0,1)** against the parent's own single-draw noise (mean z +0.026, sd 0.938, worst \|z\| 3.43/228). The known crack is confirmed where H1 says it must be: the scan family needs the exact **Brownian** parent (M3's independent parent: +7.5% at n = 2, CV 0.075 vs measured ~0.14; Brownian: ≤1.9% everywhere, CV 0.139 vs 0.141). §1 |
| **2. Kink bound** for the Gram screen | **DERIVED AND VALIDATED.** The l1norm deviation is a first-order, scale- and magnitude-invariant functional of the kink fraction p: g(1/2) simulates to 1.41 [1.30, 1.54] against the five banked corpus medians 1.34–1.44 (5/5 inside), g grows ~ p/(1−p), and the factor-2 flag line is crossed at **p\* ≈ 0.64** (earliest plausible single-invocation fire p ≈ 0.59). The binding margin of the Gram screen is now a derived quantity with a stated exclusion rule, not a lucky measurement. §2 |
| **3. Reseeding-collapse sweep** | **COUNTED, and bigger than the one flagged instance.** 23 of ~83 adversarial (op, check) classes per corpus round are bit-identical replicas of ONE (input, deltas) draw; the 854-record bank holds **513 bit-distinct** measurements (adversarial 632 → 296). No verdict or outcome changes; evidence weights do. Two by-products: the gram round's FINDINGS misfiled an *unevaluated* class as measured-smooth (corrected), and the sweep exposed a **real reference-suspect: out-of-bounds companion reads in layernorm/rmsnorm `non_power_of_two` at this corpus's shapes** (flagged, not fixed). Affected FINDINGS annotated in place. §3 |
| **4. Doc defects** | **All three fixed in place** with dated corrections: the refuted m=1 comment in `structural_l.py`; the SOTA registry's fixed-tolerance-causes-FP causal claim (retracted for AutoKernel, descoped for gpuemu); and the faithful-gate FP units — which turned out to be **reconcilable**: the two artifacts agree (0.5%/1.0% = 1 and 2 of 200), the handoff prose matched neither and was a transcription error. §4 |

---

## 1. B.3 chaining — the n-curve is a functional of the structural parent, verified at every n

### The chain, as run

`probes/b3_chain.py`, `probes/b3_zscores.py`. Per invocation of the 228-row
native bank (27 operators × 6 draws, minus argmax/argmin):

1. **Replay** the exact input (numpy `default_rng(0)`, registry order —
   the attention-gram round's proven path). Hard gate: banked `sigma` must
   equal `1e-3·std(x)` — **228/228 pass**.
2. **Structural parent, exactly**: profile `w = row_norms(op, x, rest)/L`
   (the shipped closed forms), parent CDF `F(t) = ∏_i (2Φ(t/w_i) − 1)`
   evaluated on a grid — no simulation of the max, no fitted constants —
   then inverse-transform sampling of the 40-draw estimator.
3. **Three predictors** of `tol_n/tol_40`: DIRECT (E[q95_n]/E[q95_40] under
   the parent), CHAIN (the validated Gumbel one-parameter model fed the
   parent's structural CV — the literal composition B.3 named), GUMBEL-M
   (the model fed the measured CV — the previously-validated baseline).
4. **Measured side**: the banked 40-sample vector's own prefix curve
   (exact, per the prefix-monotonicity property).

### Result

| n | measured (mean) | DIRECT | CHAIN | GUMBEL-M |
|---:|---:|---:|---:|---:|
| 2 | 0.8331 | 0.8512 | 0.8292 | 0.8278 |
| 5 | 0.9160 | 0.9236 | 0.8933 | 0.8925 |
| 10 | 0.9628 | 0.9651 | 0.9378 | 0.9373 |
| **20** | **0.9852** | **0.9841** | 0.9741 | 0.9738 |
| 30 | 0.9955 | 0.9963 | 0.9905 | 0.9904 |

The DIRECT route is the right one: at the shipped default it lands within
0.11% of the measured aggregate, while both Gumbel-model routes carry the
model's documented −2.3% bias (the model, not the parent, is the
approximation — feeding it a perfect CV cannot fix its tail shape). The
per-invocation test is the sharp one: scoring each invocation's measured
prefix ratio against the parent's own predicted (mean, sd) for that joint
statistic gives, over 228 invocations,

    n = 20:  mean z = +0.026 (sem 0.066),  sd(z) = 0.938,  worst |z| = 3.43
    n =  5:  mean z = −0.070 (sem 0.066),  sd(z) = 0.977,  worst |z| = 4.17

i.e. the measured n-dependence, *including its scatter*, is fully explained
by single-draw sampling noise around the structural prediction. The
`y_profile` docstring's flag ("one step outside what was checked") is
resolved and updated in place; `generalization/FINDINGS.md` §B.3 carries a
DONE note.

### The scan family — the crack, exactly where H1 requires it

Shape-only parents at (64, 512), measured side = the 24 banked phase-1 scan
invocations:

| n | measured | M3-independent | Brownian (exact Gram) |
|---:|---:|---:|---:|
| 2 | 0.8569 | 0.9209 (+7.5%) | 0.8665 (+1.1%) |
| 5 | 0.9501 | 0.9595 (+1.0%) | 0.9324 (−1.9%) |
| 20 | 0.9917 | 0.9911 (−0.1%) | 0.9866 (−0.5%) |

CV: measured 0.141 (range 0.103–0.187), M3 0.075, **Brownian 0.139**. The
independence assumption halves the scan CV and distorts the deep-prefix end
of the curve; the exact Gram parent repairs both. So the honest statement of
the closed gap is: **the n-curve is a functional of the parent distribution,
and the chain is validated with the *correct* (Gram) parent** — M3's
independent parent suffices for the 27-op corpus (correlations ≤ 4%) and
fails for scans by exactly the mechanism the eighth result derived.

Limits: the native corpus's shapes are small (m ≤ 8192); each measured curve
is one 40-draw realization (addressed by the z-test, not by more draws); the
scan arm is family-level (shape-only parent, 24 invocations at one shape).

## 2. The kink bound — g(p), and p\* ≈ 0.64

### Derivation (first order, exact structure)

l1norm rowwise: `f(x)_j = x_j / D`, `D = Σ_k |x_k| + ε`. Let a fraction `p`
of a row's entries be exactly 0 (set Z), the rest nonzero (set S), and
perturb by `d` with iid N(0, σ²) entries, σ small vs min_S \|x_k\|. Then

    D(x+d) − D  =  A + K,     A = Σ_S sign(x_k) d_k   ~ N(0, (1−p)C σ²)
                              K = Σ_Z |d_k|           ≥ 0,  E K = pCσ√(2/π)

    f(x+d) − f(x) =  d/D − x·(A + K)/D²  + O(σ²)          (measured)
    J d           =  d/D − x·A/D²                          (autograd, sign(0)=0)

The screen's per-delta discrepancy is carried entirely by the **rectified
sum K** — a first-order-in-σ, strictly positive term the Jacobian cannot
represent. Consequences, each verified in `probes/kink_bound.py`:

- **First order is the whole story**: worst relative error of the
  decomposition vs the exact float64 response is 4.7% (the O(σ²) residue at
  this corpus's σ), over 10 invocations × 8 deltas.
- **Scale-invariance**: both sides are Θ(σ), so the ratio is independent of
  `delta_scale` (measured 1.4197/1.4185/1.4069 across 1e-4/1e-3/1e-2 — the
  1e-2 drift is the second-order onset) and **exactly** independent of the
  nonzero magnitude τ (1.4185 at τ = 0.1, 10, 1000). The deviation is
  geometry, not noise — which is why the corpus measures a *fixed* 1.44×.
- **The corpus number is reproduced from the math function alone**: at the
  exact corpus configuration (p = 1/2, C = 128, τ = 10), 200 simulated
  invocations of the shipped screen statistic give median ratio **1.406**,
  95% band [1.30, 1.54] — the five banked fp32-Triton medians (1.34–1.44)
  all sit inside. The kernel contributes nothing material; the 1.44× is the
  operator's non-C¹ geometry.

### g(p) and the exclusion rule

Median screen statistic vs kink fraction (C = 128, Gaussian nonzeros; the
sweep is the semi-closed form — no probe of any kernel):

| p | 1/8 | 1/4 | 3/8 | 1/2 | 5/8 | 3/4 | 7/8 | 15/16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| g(p) | 1.006 | 1.043 | 1.16 | 1.41 | 1.91 | 3.02 | 6.43 | 11.98 |

g − 1 grows as `c·p/(1−p)` (c ≈ 0.4→0.8, the log-factor drift of the
asymptotic), unbounded as p → 1. The median crosses the factor-2 flag line
at **p\* ≈ 0.64**, and the upper 95% band crosses at **p ≈ 0.59** — the
earliest configuration at which a single in-scope invocation could
plausibly fire.

**Rule, per the gram round's §2 position (threshold not to be moved):** an
l1-type operator evaluated at inputs with kink measure `p ≳ 0.59` is
*structurally* outside the C¹ scope — route it to structural handling (as
argmax/argmin are), do not retune `GRAM_MAX_ABS_LOG10`. The corpus's worst
case (p = 1/2) sits below with the derived margin, and the margin is now a
function of p, not a hope.

Scope honestly stated: the derivation is for kinks entering through Σ\|·\|
(l1-normalization family). Max/pool selection kinks are measure-zero at
generic inputs and produce no such term; other non-C¹ families would need
their own K-analogue (same recipe: split the increment into linear +
rectified parts).

## 3. The reseeding collapse — 854 records, 513 independent measurements

### Mechanism, now with the correct boundary

`KCC_ABLATION_SEED` reseeds torch from `crc32(check_name)` before every
check. Empirical sweep (`probes/reseed_sweep.py`) over the CURVE, scope, and
gram banks, fingerprinting each record by its banked sensitivity vector /
scope signals / gram ratio vector:

| bank | records | bit-distinct |
|---|---:|---:|
| CURVE_n40 (n_samples round) | 854 | **513** (adversarial 632 → 296) |
| scope arms B / D | 842 | 505 / 501 |
| gram arm G | 842 | 499 |

**Collapsed (one draw, replayed 6–22×), 23 classes, identical across all
three rounds**: all six flash_attention adversarial variants, all six
matmul variants, all five softmax variants, rmsnorm/non_power_of_two,
max/min all_negative/positive_nonpow2 — these draw fresh torch tensors
under the fixed per-check seed — plus the three `full_like(x,3)+x·1e-6`
near_zero_variance variants, which differ in construction but are
bit-identical at fp32 measurement resolution (resolution-collapse), and
argmax's all-zero vectors (trivial). gelu/swish `near_global_min`
(`x·0.01` transform) are *partially* collapsed (2–5 distinct of 6),
straddling the resolution boundary.

**Not collapsed**: every primary invocation (base inputs advance through
the harness numpy rng) and every captured-transform variant — including a
subtle sub-case the sweep exposed: layernorm's variants replace `x` with a
fresh (collapsed) draw, yet vary run-to-run **through their captured
gamma/beta companions**, which ride along from the numpy-drawn base inputs.
Companion capture is the difference between flash/matmul (regenerate
everything → collapsed) and layernorm (companions vary → distinct).

**What changes and what does not.** No verdict, fire, margin, or
classification moves — each record is a faithful measurement; the replicas
are simply not additional evidence. Claims citing per-class denominators
("22/22", "10/10", "15/15") for collapsed classes have per-class evidence
n = 1; distributional claims over "the 854 invocations" (theory-audit H2
included) are over 513 independent draws with collapsed classes
over-weighted. The three affected FINDINGS (n_samples, scope_detect,
gram_screen) and theory_audit carry dated annotations. The native-corpus
rounds (adaptive_tol, phase1/2, attention_gram, method blind test) are
UNAFFECTED — their harnesses seed per-invocation and never used the
ablation reseed; B.3's replay above independently confirms their inputs
vary per invocation.

### Two by-products

**(a) A misfiled class in the gram round's FINDINGS, corrected.**
`layernorm/adversarial_non_power_of_two` was listed in §2's measured-smooth
bucket; in fact the Gram screen evaluated **neither** layernorm nor rmsnorm
`non_power_of_two` — `gram_n_valid = 0, n_skipped = 20` on every record
(the math definition's companion slicing raised on every delta; the screen
declined fail-open, silently). Their silence is absence of measurement.
Corrected by annotation; a fail-open counter worth surfacing in any future
adoption pass.

**(b) REFERENCE-SUSPECT FLAG (flagged, NOT fixed): out-of-bounds companion
reads.** In the autokernel corpus, layernorm/rmsnorm inputs are (64, 128)
with length-128 gamma/beta; the `non_power_of_two` variants replace `x`
with a width-333 tensor while the captured companions ride along, and the
reference kernels load them with `mask = col_offsets < n_cols` (n_cols =
333) over 128-float allocations — **columns 128–332 are out-of-bounds
reads** (banked input_stats confirm shape [64, 333]). Consistent with the
data: rmsnorm's sensitivity vectors are bit-identical across runs despite
*varying* in-bounds gamma — the response is dominated by stable OOB lanes.
At the spec corpus's 512-wide shapes the same variants are in-bounds; the
exposure is specific to this corpus's shapes. 31 records per round
(15 rmsnorm + 16 layernorm), all silent/in-scope rows; no catch, FP, or
reported number rests on them. Adjudication and any fix (the variant should
resize companions or cap its width at the fed width) belongs to its own
round with blast radius, per house rules — this note is the flag.

## 4. Doc-defect fixes, all applied in place with dated notes

1. **`structural_l.py` (m=1 comment, was ~line 351).** Said m=1 is "M3's
   known-worst regime (+121% over-prediction)" and predicted the five
   losses would drag the fit down — refuted by the phase-1 GPU round (the
   +121% belongs to M1′; under M3 cross_entropy is −1.8% and the losses are
   unbiased as a group). Corrected to state the measurement and cite
   GPU_NATIVE.md.
2. **`SOTA_CHECKS_REGISTRY.md` (fixed-tolerance causal claim).** Claimed
   fixed tolerance is "the direct mechanism behind" autokernel_gate's 18%
   and gpuemu's 82% FP rates. The AutoKernel half is contradicted by the
   baseline audit (the 18% was harness exceptions; the faithful re-run
   measures 80% / 0.5%); the gpuemu half was never audited. Corrected:
   adaptive-vs-fixed remains the design differentiator, but no measured FP
   rate in that table is evidence for it.
3. **Faithful-gate FP units.** RESOLVED rather than merely flagged:
   `results.json` (0.005 / 0.010 of n = 200, frobenius per-op 1/5 and 2/5)
   and `results.md` (0% / 1%, integer rounding of the same values) agree;
   `SESSION_HANDOFF.md` §1's "1% / 2%" matched neither artifact and was a
   transcription error. Handoff corrected in place; RESULTS_SUMMARY's
   "unreconciled" flag replaced with the reconciliation. The one-draw
   frobenius-flake caveat stands unchanged.

## 5. Limits

- **B.3**: validated on the native corpus (small shapes, ordinary inputs,
  27 + scan ops). Adversarial-input n-curves were not chained (no
  adversarial rows carry banked vectors in the native bank); the corpus
  bank's adversarial vectors exist but their inputs are not CPU-replayable.
  The z-test buys exactness at the cost of resting on the parent's own sd —
  a parent mis-shape would inflate sd(z) away from 1, which is what was
  checked (0.94–0.98).
- **Kink bound**: derived for the Σ\|·\| kink family; the p\* value is for
  C = 128, iid Gaussian nonzeros (the probe is the general tool — rerun it
  for other configurations). The 4.7% first-order residual is σ-dependent.
- **Sweep**: "bit-distinct" is the operational definition of independence
  here; distinct records from the same generator are still same-family
  draws, and the three corpus rounds replay each other's seeds by design
  (cross-round records are the SAME draws — that is what makes the rounds
  comparable, and it means the three arms are one sample, not three).
- The OOB flag is evidence-complete at the code+data level but its blast
  radius (does any recorded number depend on the OOB lanes' values?) is
  asserted from record inspection, not from a re-run with resized
  companions; the deferred adjudication round should do that re-run.

## 6. Reproduce

```bash
cd verification_runs/theory_closure_2026-08-28
PY=../../.venv/bin/python
$PY probes/b3_chain.py       # replay + parents + 3 predictors vs banked prefix curves
$PY probes/b3_zscores.py     # per-invocation z against the parent's own sd
$PY probes/kink_bound.py     # corpus-config validation, g(p) sweep, p*, invariances
python3 probes/reseed_sweep.py   # fingerprint census over the three corpus banks
```
Run logs and derived data in `data/`.

---

> **§3(b) FLAG ADJUDICATED, 2026-08-28 — `../oob_adjudication_2026-08-28/FINDINGS.md`.**
> Verdict: **spec-construction artifact, not a kernel bug.** The OOB read is
> real and was proven at the byte level (recovered leak columns equal the
> contents of adjacent named allocations, mapped by pointer arithmetic;
> leaked columns change when a neighbor's contents change) — but on
> valid-length companions both kernels reproduce their own arithmetic to
> ≤ 1e-6, and the defect is `_non_power_of_two`'s hardcoded width 333
> wrapped with captured companions in the two specs. Blast radius
> independently re-established across ALL banks (558 affected records, zero
> catch/FP/margin dependencies; 1 of 17 search proposals in the window,
> verdict independent). The three non-pow2 findings are NOT one bug: the two
> kernel instances share the sentinel-neutrality cause (family closed at
> 3/64); this one is harness-side. Fix specified there (width-adaptive
> variant + wrapper shape asserts + a spec contract test), deferred with
> regression criteria. Also: compute-sanitizer fails a positive control on
> Triton JIT kernels in this environment — never cite it as evidence of
> in-bounds behaviour on this stack.

> **§3(b) FIX SHIPPED, 2026-08-28 — `../oob_fix_2026-08-28/`.** Width-adaptive
> variants + wrapper asserts + the spec contract test are in; all four
> adjudication criteria pass, and the §3 prediction that rmsnorm's collapse
> was OOB-content-driven is confirmed (15/15 distinct post-fix). The two
> classes now measure at Gram ratio ≤ 1.0001 with `gram_n_valid = 20`.
