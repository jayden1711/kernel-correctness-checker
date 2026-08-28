# The Jacobian-scope method: an operator-agnostic procedure for deriving, predicting, and guarding adaptive numerical tolerances

**Formalized 2026-08-27.** This document states the *procedure* that the last
month of per-operator results instantiates, in the form a new operator would
consume it. The 51+ operators measured across
`adaptive_tol_theory_2026-08-25/`, `phase1_derivations_2026-08-27/`,
`phase2_convolution_2026-08-27/`, `theory_audit_2026-08-27/` (scans) and
`attention_gram/` (attention) are its **validation set, not its content**:
nothing below is specific to any of them. The blind test in §6 applies the
procedure to an operator none of those rounds ever touched
(`logcumsumexp`) with predictions registered before the kernel was run.

The method's one-sentence form:

> For a tensor operator `f` at input `x`, the entire behaviour of a
> perturbation-based tolerance is a functional of the Jacobian `J(x)` --
> its max row norm sets the scale, its Gram matrix `JJ^T` sets the
> distribution, and a per-delta comparison against `J d` certifies whether a
> real kernel is inside that description -- so derive `J` structurally,
> predict from `JJ^T`, and validate by directional derivative, in that order.

---

## 0. Setting and objects

The checker's Layer-2 oracle measures, for a reference implementation `f32`
(float32, GPU) at input `x`:

    s_k = || f32(x + d_k) - f32(x) ||_inf,    d_k = sigma * g_k,
    g_k ~ N(0, I) iid,  sigma = delta_scale * std(x),  k = 1..n

    adaptive_tol = max(3 * q95_n(s), 1e-6)

The theory (documented results 1-8) says: **in scope** -- `f` C^1 at `x`,
response above the fp32 floor, below saturation --

    s_k  =  sigma * max_i |<J_i, g_k>|  + O(curvature),

so the vector `(<J_i, g>)_i` is Gaussian with covariance `JJ^T`, and the
distribution of `s`, of `q95_n(s)`, and of `adaptive_tol` are functionals of
the Gram matrix alone. Everything in the procedure exploits one face of that
single fact.

Dimensionless form used throughout: `y = tol / (3 * sigma * L)` with
`L = max_i ||J_i||_2`. In scope, `y` depends only on the *correlation
structure* of `JJ^T` (scale-free), which is what makes cross-operator and
cross-input comparison meaningful.

## 1. The procedure

Given a NEW operator `f` (its mathematical definition, not its kernel):

**(a) Derive `L` structurally from the Jacobian.**
Differentiate the math definition. For operators built from linear maps and
smooth nonlinearities with known derivatives -- every operator this project
has met -- the row norms `||J_i||_2` have a mechanical closed or
semi-closed form (worked examples: §5 table). Three outcomes are possible,
and each is information:
  - `L` closed-form in shape alone (exactly linear ops: scans, matmul with
    fixed B, reductions);
  - `L` closed-form in the input (softmax families, normalizers, attention:
    evaluate a formula at `x`);
  - `J = 0` almost everywhere (index-valued outputs) -> the operator is
    **structurally excluded**: perturbation tolerances are undefined for it,
    stop here (taxonomy class 4).

**(b) Derive or estimate the Gram correlation structure.**
The question is only: *how correlated are the rows of `J`?* Because `y` is
scale-free, only `JJ^T`'s correlation profile matters.
  - Rows (near-)orthogonal or `m` large with local coupling -> the
    independent-max model (M3) is already accurate: **independent-row
    class**. (Attention landed here after measurement: +3-4% true
    correction, not the +17% a single noisy draw suggested.)
  - Rows nested/overlapping by construction (prefix structures) ->
    **correlated-row class**; M3 over-predicts `y` by a computable factor
    (scans: exactly the Brownian-vs-independent gap, 1.231, matching the
    measured +24.7%). For an exactly-linear member the law can be pushed to
    a zero-constant closed form (reflection principle); otherwise simulate
    `max |J z|` with the exact `J(x)` -- zero fitted constants either way.
  - `m = 1` (scalar output) -> the Gram matrix is 1x1: the sensitivity
    sample carries scale but NO shape information (documented result 7, and
    the rank-1 equality case of the CV <= 0.7555 lemma): **m=1 class**,
    diagnostic blindness expected, prediction still exact.
Class boundaries are structural, but *membership can be input-dependent*:
the blind test's operator moves from independent-row behaviour at ordinary
inputs (m3/gram 1.00) to scan-like correlation under saturating inputs
(m3/gram 1.22-1.30) -- the classification is a property of `JJ^T(x)`, and
the procedure computes it per input rather than assuming it per operator.

**(c) Validate against the real kernel by directional derivative.**
The paired test: with the SAME deltas the tolerance was built from,

    r_k = s_meas_k / || J(x) d_k ||_inf,   J d_k by float64 forward-mode
                                            autodiff of the math definition

In scope, every `r_k` is 1 up to curvature at the delta scale (banked
<= 0.1% ordinary, <= ~15% at the most adversarial in-scope inputs measured)
plus fp32 measurement noise (bounded above the floor). This is a *paired*
comparison, so the +-8% single-draw noise of distributional tests cancels;
it needs no closed form, no simulation, and no threshold fitting -- the
exact derivative is the ground truth for what a linear response would be.
A kernel/input pair failing (c) while (a)-(b) predict in-scope behaviour
means either the input left the theory's scope (saturation, fp floor) or
**the kernel does not compute `f`** -- the method has now found two real
shipped-reference bugs this way (attention padded-column, found by a -10%
Gram deviation at an out-of-sample shape; layernorm unmasked pad lanes,
confirmed by the same machinery).

**(d) Classify into the taxonomy and act accordingly.**

| class | signature (measured, not assumed) | consequence |
|---|---|---|
| independent-row | m3/gram ~ 1, r_k ~ 1 | M3 row-norm prediction valid as-is |
| correlated-row | m3/gram > 1 by the derived factor, r_k ~ 1 | predict from `JJ^T` (closed form or exact-J simulation) |
| m=1 | rank-1 Gram; CV at the 0.7555 ceiling | tolerance predictable, diagnostics blind: q95/RMS degeneracy |
| structurally excluded | `J = 0` a.e. | no perturbation tolerance; exact-match checks instead |

*Added 2026-08-28 — a third exclusion flavour, distinct from `J = 0`.*
Structural exclusion has (at least) three mechanically different causes, and
the taxonomy should name them because the remedies differ. **(i) `J = 0`
a.e.** (argmax/argmin): the tolerance is undefined because there is nothing
to linearise — exact-match checks instead. **(ii) non-C¹ kink mass** (the
l1norm boundary case): `J` exists a.e. but the measured/Jacobian discrepancy
is first-order in the rectified mass; excluded when the kink measure passes
p ≳ 0.59 (`theory_closure_2026-08-28` §2) — never a threshold retune.
**(iii) unbounded conditioning** (`cumprod`, CORPUS_EXPANSION_PLAN L1 #90):
`J_ij = ∏_{k≤i,k≠j} x_k` is everywhere defined and smooth, but its row norms
are input-dependent with unbounded condition number — a single near-zero
entry swings `‖J_i‖` by orders of magnitude, so no useful `L` exists and a
40-delta sensitivity estimate has no stable population parameter to
converge to. The operator is excluded not because the derivative is
degenerate but because the *tolerance functional of the derivative* is
ill-posed. Remedy differs accordingly: (i) exact-match, (ii) exclusion
above the kink-mass bound, (iii) either per-input exact-J evaluation
(no closed `L`, cost accepted) or property-based checks only.

Runtime counterpart: the scope detector (`scope_detect.py`) is the
productionized residue of (c) -- structural exclusion for class 4, the
s/ulp floor screen for the quantisation boundary, and the Gram screen
(median log10 r_k against a pre-registered factor-2 band) for everything
the Jacobian cannot explain. Annotate-only by construction.

## 2. What each step consumes and produces

| step | needs | produces | cost |
|---|---|---|---|
| (a) | math definition | `L` formula or exclusion | pencil-and-paper / one autodiff |
| (b) | `J`'s sparsity/overlap pattern | class + `y` prediction (+ sd) | closed form, or NREP x JVP simulation |
| (c) | shipped kernel + banked deltas | per-delta `r_k`, verdict on scope | n float64 JVPs, no extra kernel launches |
| (d) | (b) + (c) outputs | taxonomy class, screen wiring | arithmetic |

## 3. Why the pieces are believed (the validation set)

- **Linearisation theorem** (2026-08-25 round): defect < 0.1% on ordinary
  inputs across the corpus -- step (c)'s null.
- **Gram law, exactly-linear closed form** (theory_audit H1): scans, 4/4
  operators within 2.2%, zero constants, M3 residual reproduced as the
  independence-vs-Brownian gap (predicted 1.231, measured ~1.247).
- **Gram law, input-dependent Jacobian** (attention_gram): 36 banked + 108
  fresh measurements, mean z +0.13, out-of-sample shapes clean once the
  kernel-faithful Jacobian is used -- and the one deviation was a real
  reference bug, which is (c) working as designed.
- **CV ceiling / m=1 degeneracy** (theory_audit H5 + result 7): the sharp
  half-normal bound, equality iff rank-1 -- step (d)'s m=1 signature.
- **Floor boundary** (scope_detect round): median s/ulp >= 32, margins
  9-381x, corpus-validated.
- **Gram screen** (gram_screen_2026-08-27): the corpus run scoring step
  (c) as a runtime detector -- see that round's FINDINGS.md for the
  separation the retired defect threshold could not deliver.

## 4. Generalization claim, stated honestly

**Claimed:** the procedure applies mechanically to any operator whose math
definition is built from linear maps and C^1 nonlinearities with known
derivatives, because (a) is then a chain-rule computation, (b) reads
correlation off `J`'s overlap structure (computable numerically from JVPs
even with no insight at all), and (c) is pure autodiff against the kernel.
No step consults a fitted constant or an operator-specific measurement.

**Not claimed:**
- a closed form for every family (the scan reflection law needed genuine
  derivation; a new correlated family gets exact *simulation* for free but
  a formula only with work);
- coverage of non-C^1 points: piecewise structure (max/pool ties,
  saturated selects) is exactly where scope ends -- the method *detects*
  this (step c) rather than modelling it;
- immunity to spec ambiguity: (c) compares against the math definition, so
  an eps placement transcribed wrongly indicts the transcription, not the
  kernel (guarded by output-level cross-checks in tests);
- anything about m=1 diagnostics: prediction works there, discrimination
  provably cannot (result 7).

**The real residual risk** for a genuinely novel family is step (b)'s
*interpretation* -- knowing which structural feature drives the correlation.
The blind test below probes exactly that: an operator whose class is not
fixed but input-dependent, decided by the derivation itself.

## 5. Worked structural table (the corpus as instances)

| family | J row structure | L | class |
|---|---|---|---|
| reductions (sum/mean) | one dense row | sqrt(C), sqrt(C)/C | m=1 per row / independent |
| max/min reduction | one-hot row | 1 | independent (piecewise) |
| scans (cumsum family) | nested 0/1 prefixes | sqrt(C) | correlated (Brownian, closed form) |
| softmax / log_softmax | diag(p) - p p^T per row | rowwise formula | independent-row (local coupling) |
| normalizers (LN/RMS/GN/IN) | centered projection / sqrt(var+eps) | formula in row stats | independent-row |
| matmul (A@B, perturb A) | B's columns, block per row | max_j ||B_j|| pattern | independent-row |
| attention (perturb Q) | per-row D_v x D block, shared softmax denominator | exact-J | independent-row (+3-4%) |
| elementwise (gelu/swish) | diagonal | max|f'| | independent |
| argmax/argmin | 0 a.e. | -- | structurally excluded |
| **logcumsumexp** (blind) | **prefix softmax rows** | **exactly 1** (row 0 = e_0) | **input-dependent: independent at randn, correlated 1.22-1.30 under saturation** |

## 6. The blind test (pre-registered)

Operator: `logcumsumexp` (rowwise). Present in NO spec, NO reference kernel,
NO banked measurement, NO prior derivation in this repository (grep-verified
before writing stage 1). Kernel under test: ATen's shipped CUDA
implementation via `torch.logcumsumexp` -- code this project did not write.

Protocol (`probes/blind_predict.py`, `blind_measure.py`, `blind_compare.py`):
stage 1 derives (a)+(b) on CPU float64 and BANKS 18 configurations
(3 shapes x {randn, x50 saturating, sorted} x 2 seeds), the 40 deltas each,
their exact directional derivatives, and the full-distribution predictions
`y_pred +- sd` plus the M3 baseline -- all before any GPU contact. Stage 2
measures the shipped kernel with those exact deltas. Stage 3 scores:

  - distributional law: z = (y_meas - y_pred)/sd_pred, pass = all |z| <= 3
    and family mean consistent with noise;
  - step-(c) screen: median |log10 r_k| < log10 2 at every config
    (pre-registered expectation: the kernel is in scope even at the
    saturating inputs, because a saturated prefix-LSE degrades into a
    smooth running-max -- unlike attention's saturated softmax-times-V);
  - classification: m3/meas > 1 wherever the derived m3/gram > 1.05.

Two structural discoveries fell out of step (a) alone, before measurement --
`L = 1` exactly and input-independently (row 0 of every prefix softmax is
`e_0`, and no row norm can exceed 1), and the class membership is
input-dependent (table above). Both were derived sight-unseen; whether the
shipped kernel agrees is what stage 3 reports. **Results: see FINDINGS.md
of this round -- the outcome section is written from the banked stage-3
output and this section is not edited after the fact.**

## 7. How to run this procedure on the next operator

1. Write the float64 math definition, register it in
   `verification/layer2_numeric_oracle/math_refs.py` (one function; slice
   companions to the fed width if the spec allows width-changing variants).
2. Differentiate for `L`; if the output is index-valued, OR the row norms
   have unbounded input-dependent conditioning (the cumprod flavour, §1(d)
   iii), add it to the structural exclusion list — with its flavour — and
   stop.
3. Compute `JJ^T`'s correlation profile at representative inputs (JVP
   sampling needs ~10 lines; closed form optional).
4. Bank predictions, then measure the kernel with the checker's own
   protocol; score z and median log10 r.
5. File the operator in the taxonomy row its NUMBERS put it in -- not the
   row analogy suggests. (Attention "obviously" shared the scans'
   correlation mechanism; measurement said 3%, not 25%.)

---

**Deliberately deferred (2026-08-28)** — noted so they are not read as
silently dropped:

- **A second blind test (candidate: RoPE)** is deferred unless/until this
  method is written up externally. One passed pre-registered blind test
  (logcumsumexp, §6) is the current evidence for generalisation; a second
  would strengthen a publication claim but changes no in-repo decision.
- **The scan family's ~−2% signed residual** (the direct structural parent
  predicts the prefix curve to ≤1.9% with a consistent sign;
  `theory_closure_2026-08-28` §1) is deferred unless the scan family is
  ever re-measured — it sits inside single-draw noise for every banked
  decision and is only diagnosable with fresh multi-draw data.
