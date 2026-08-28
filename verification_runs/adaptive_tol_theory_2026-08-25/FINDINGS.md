# `adaptive_tol = 3.0 × q95(sensitivities)` — one theorem, two structural results, and one clean negative

**Investigated 2026-08-25.** Probes in `probes/`; all numbers reproducible from
them plus the banked `verification_runs/n_samples_curve_2026-08-25/arms/CURVE_n40.json.gz`.
**Nothing in the checker was changed.**

---

## Verdict up front

| question asked | answer |
|---|---|
| Is `3.0` derived? | **No.** Arbitrary round number, present in the first non-empty version of the file, never justified anywhere in the repo. |
| Is there a provable **false-positive-rate** bound? | **No, and there cannot be one from this construction.** The exchangeability the order-statistic route needs is false, not approximately-true. Stated plainly in §4. |
| Is there *any* rigorous bound? | **Yes — a two-sided one, §3.** It bounds the tolerance itself and yields a one-sided **detection** guarantee. Assumptions verified, 0/19 violations. |
| Is there a predictive formula? | **Yes for the `n_samples` dependence (§5), median error ≤4.3%.** **No for catch/FP rate (§6)** — the corpus response surface is constant, so nothing is fittable. |
| Is this the right place to look for the second paper result? | **Partly.** §3 is a real theorem but it characterises the *mechanism*, not the constant. §2 is arguably the more paper-worthy finding. See §7. |

---

## 1. The mechanism, exactly

`verification/layer2_numeric_oracle/perturbation.py:41-123`.

```
sigma   = delta_scale * x.float().std()            delta_scale = 1e-3
d_k     = randn_like(x) * sigma                    k = 1..n, n = 20, drawn ONE at a time
s_k     = || f(x + d_k) - f(x) ||_inf              f = the REFERENCE, not the candidate
tol     = max( 3.0 * quantile(s, 0.95), 1e-6 )     quantile = torch.quantile, linear interp
verdict = fail  <=>  || f~(x) - f(x) ||_inf  >  tol
```

Same `n_samples=20` mechanism as the latency work. `q95` is computed over the
`n` scalars `s_1..s_n` — one per perturbation sample, each already an
`inf`-norm over the whole output tensor. The perturbation is applied to the
**primary** tensor only; companion tensors (`gamma`, `beta`, `B`, `K`, `V`) are
held fixed (`checker.py:281`).

**Provenance of the constants.** `scale=3.0`, `quantile=0.95`,
`delta_scale=1e-3`, `n_samples=20` all appear together, fully formed, in the
first non-empty version of the file (`d7966dc`, 2026-05-25). Its docstring
*describes* `3.0` — "scale=3 means the candidate may be up to 3x noisier than
the reference's natural sensitivity" — but derives nothing. `git log -S` finds
no later change to any of them, and a repo-wide grep finds no ablation, no
tuning record, and no other mention of the value except one descriptive line in
`n_samples_curve_2026-08-25/FINDINGS.md`. **It is an arbitrary round number.**

---

## 2. The mechanism is not measuring what its name says — and `3.0` is not identifiable

### 2.1 Linearisation holds, to <0.1%

`s_k = ||f(x + sigma g_k) - f(x)||_inf` was compared against the exact
directional derivative `||J sigma g_k||_inf` (`torch.func.jvp`) on 19 op/shape
configurations drawn from `verification/specs/*`:

| median relative error `|s - s_lin| / s` | across all 19 configs |
|---|---|
| **0.00% – 0.08%** | max 0.08% (softmax `(256,1024)`) |

So to five significant figures the sampled quantity is **`s_k = ||J d_k||_inf`,
a max of `m` correlated centred Gaussians** — where `J` is the reference's
Jacobian at `x` and `m` is the output dimension.

**This means the check is a randomised estimator of the reference's local
Lipschitz constant `L = max_i ||J_i||_2`** (the `2 -> inf` operator norm),
scaled by the step size. It is *not* a noise-floor estimator, and the tolerance
it produces is a statement about **conditioning**, not about numerical error.
That reframing is what makes §3 possible and §4 impossible.

The theory also predicts its own failure cases: where `f` is piecewise constant
`J = 0` a.e., so `s = 0` and the `1e-6` floor takes over. Measured: **98/854
invocations sit on the floor, and `argmax` sits there on 100% of its
invocations** — exactly the discrete-output operators.

### 2.2 `3.0` and `delta_scale` are one constant, not two

`s(cd) = c·s(d)` exactly, under linearisation. So `tol` must be exactly
proportional to `delta_scale`. Measured log-log slope of `tol` vs `delta_scale`
over four decades (`1e-5 … 1e-1`), all 19 configs:

| slope | min 0.9982, max 1.0013, **median 1.0001** |
|---|---|

**Therefore `scale = 3.0` and `delta_scale = 1e-3` are not separately
identifiable: only their product `3e-3` can affect any verdict, ever.** The
mechanism presents one arbitrary constant as two independent tuning knobs.
Changing `3.0` to `6.0` is indistinguishable from changing `delta_scale` to
`2e-3`.

This is a rigorous, verified, one-line-provable structural defect, and it is
the cheapest real result in this document.

---

## 3. THEOREM — a two-sided bound on the tolerance, and a detection guarantee

**Setup.** `sigma = delta_scale · std(x)`, `L = max_i ||J_i||_2`, `m` = output
dimension, `n` = `n_samples`, `eta` in (0,1).

**Assumptions.**
- **(A1)** `d_k = sigma · g_k` with `g_k ~ N(0, I)`. *True by construction.*
- **(A2)** The `g_k` are i.i.d. *True by construction* — `perturbation.py:100`
  draws one `randn_like` per sample, and its comment records that batching the
  RNG was deliberately avoided.
- **(A3)** `f` is `C^1` at `x` and the second-order term is negligible at
  `sigma`. *Verified to <0.1% (§2.1). Fails exactly for `argmax`/`argmin` and
  the piecewise-constant family, where the theory correctly predicts the floor
  instead.*

**Claim.** With probability at least `1 - eta - (n+1)/2^n` over the `n` deltas,

```
    2.023 · sigma · L   <=   adaptive_tol   <=   3 · sigma · L · ( sqrt(2 ln 2m) + sqrt(2 ln(n/eta)) )
```

**Proof sketch.**
*Lower.* `q95_n >= X_(n-1:n)`, and `X_(n-1:n) >= median(s)` unless at most one
of `n` samples exceeds the parent median, which has probability `(n+1)/2^n`
(= 2.0e-5 at n=20). For the maximising row `i*`, `s >= |<J_i*, d>|`, whose
median is `sigma·L·Phi^-1(0.75) = 0.6745·sigma·L`. Multiply by 3.
*Upper.* `s = ||J d||_inf` is `sigma·L`-Lipschitz in `g`, so Borell–TIS gives
`P(s >= E s + t) <= exp(-t^2 / 2 sigma^2 L^2)`; `E s <= sigma L sqrt(2 ln 2m)`
is the standard max-of-`2m`-Gaussians bound. `q95_n <= max_k s_k`; union over
`n` and set `t = sigma L sqrt(2 ln(n/eta))`. ∎

**Corollary (detection guarantee).** Any candidate with
`||f~(x) - f(x)||_inf > 3 sigma L (sqrt(2 ln 2m) + sqrt(2 ln(n/eta)))`
is rejected with probability at least `1 - eta`. This is a *completeness*
statement and needs no assumption about the candidate.

**Verification** (`probes/sandwich.py`, `L` estimated per-row by Monte Carlo,
validated against exact `jacrev` row norms to 2–10%):

| | result |
|---|---|
| lower-bound violations | **0 / 19** |
| upper-bound violations | **0 / 19** |
| upper bound looseness `UB/tol` | 2.06 – 4.11× |
| `tol/(3 sigma L)` vs the leading term `sqrt(2 ln 2m)` | ratio **0.81–0.97 for 15 of 19 configs** |

The four configs where the leading term is ~2× loose (`softmax` 0.48/0.52,
`sdpa` 0.50, `layernorm (512,512)` 0.81) are exactly those whose output
coordinates are strongly correlated, so the effective `m` is well below the
nominal `m` — the bound's known conservative direction.

**So `adaptive_tol = Theta(sigma · L · sqrt(log m))`, and the constant in front
is `3 · delta_scale`.** That is the honest formula.

---

## 4. THE NEGATIVE — there is no false-positive bound, and the obvious route is not repairable

The tempting argument is conformal prediction. It is exact and
distribution-free: for a continuous parent and **exchangeable** scores, with
`q95_n` blending `X_(19:20)` and `X_(20:20)`,

```
    P( max_err > q95_20 )  <=  2/21  =  9.52%
```

before the `3.0` multiplier is even applied. **This bound does not apply here,
and the failure is structural, not a technicality.**

Exchangeability requires `max_err` to be a draw from the same distribution as
the `s_k`. It is not:

- `s_k = ||f(x + d_k) - f(x)||_inf` — one implementation, two *different* inputs.
- `max_err = ||f~(x) - f(x)||_inf` — two *different* implementations, one input.

These are different quantities with different physics. For a genuinely correct
but non-identical kernel, `max_err` is float reassociation error, of order
`eps_mach`; `s_k` is `sigma·L`, of order `delta_scale`. The gap is measurable:

> The **only** reference invocation in the whole 785-trial corpus with nonzero
> `max_err` — `frobenius_norm/adversarial_dominant_outlier` — has
> `max_err = 1.1921e-07`, which is **exactly 1.00 ulp of float32**, against
> `tol = 1.4076e-04`: a margin of **1181×**.

That margin is itself derivable. Writing `kappa_rel = L·std(x)/||f||_inf`,

```
    margin  =  tol / (eps_mach · ||f||_inf)  ~  3 · delta_scale · kappa_rel · sqrt(2 ln 2m) / eps_mach
```

which for this case predicts **2.3e3** against a measured **1.18e3** — right to
a factor of 2, the same factor the correlated-output cases carry in §3.

**Conclusion.** The true false-positive rate of this check on correct kernels is
not ~9.5%; it is smaller by three orders of magnitude, for a reason that has
nothing to do with `q95` or `3.0` and everything to do with `delta_scale/eps_mach ~ 1e4`.
A conformal FP bound stated for this mechanism would be **both unprovable and
wildly wrong in the conservative direction**. It would not survive review, and
it should not be written.

---

## 5. A predictive formula that does work: the `n_samples` dependence

### 5.1 Exact and distribution-free

`torch.quantile(., 0.95)` with linear interpolation reads index `h = 0.95(n-1)`.
Since `E[F(X_(j:n))] = j/(n+1)`, the **effective parent quantile actually
targeted** is

```
    p_eff(n)  =  (0.95 n + 0.05) / (n + 1)
```

| n | what `q95_n` is | `p_eff` |
|---:|---|---:|
| 3 | 0.10·X₍₂:₃₎ + 0.90·X₍₃:₃₎ | 0.7250 |
| 5 | 0.20·X₍₄:₅₎ + 0.80·X₍₅:₅₎ | 0.8000 |
| 10 | 0.45·X₍₉:₁₀₎ + 0.55·X₍₁₀:₁₀₎ | 0.8682 |
| **20** | **0.95·X₍₁₉:₂₀₎ + 0.05·X₍₂₀:₂₀₎** | **0.9071** |
| 40 | 0.95·X₍₃₈:₄₀₎ + 0.05·X₍₃₉:₄₀₎ | 0.9280 |
| 1000 | — | 0.9491 |

**At the shipped `n = 20`, "q95" is the second-largest of 20 draws, and it
targets the 90.7th percentile, not the 95th.** It reaches 0.95 only as
`n -> inf`. This is exact, needs no model, and is a fair thing to state in a
paper as a characterisation of the estimator.

### 5.2 A one-parameter model for the whole curve

Under §2.1 the parent has a Gumbel-type upper tail, `Q(p) = a + b·G(p)` with
`G(p) = -ln(-ln p)`. With `rho = b/a` recovered from the sample coefficient of
variation via `CV = (pi/sqrt6)·rho/(1 + gamma·rho)`:

```
    tol_n / tol_40  =  [ 1 + rho·G(p_eff(n)) ] / [ 1 + rho·G(p_eff(40)) ]
```

Tested on all **804** banked 40-sample vectors (`probes/ncurve.py`):

| n | measured | predicted | error |
|---:|---:|---:|---:|
| 2 | 0.8535 | 0.8462 | −0.9% |
| 3 | 0.9048 | 0.8718 | −3.6% |
| 5 | 0.9445 | 0.9039 | **−4.3%** |
| 10 | 0.9748 | 0.9440 | −3.2% |
| 20 | 1.0000 | 0.9766 | −2.3% |
| 30 | 0.9955 | 0.9914 | −0.4% |

Per-invocation median `|error|`: 8.6% at n=2 falling to 2.4% at n=20. The
aggregate curve is predicted from **one number per invocation**.

This also re-derives, from theory rather than measurement, the direction the
earlier `n_samples` round observed empirically: `p_eff` is increasing in `n`, so
fewer samples ⇒ tighter band ⇒ the risk of lowering `n_samples` is false
positives, never lost catches.

---

## 6. What is NOT fittable: catch/FP rate as a function of `(n, scale)`

Because `max_err` and the full 40-vector are both recorded, and the n-sample
vector is a prefix of the 40-sample one, **every `(n, scale)` verdict is
determined exactly offline**. The full surface (`probes/surface.py`), n ∈
{1…40} × scale ∈ {1e-3 … 1e6}:

| | result |
|---|---|
| false positives, **all 117 grid points** | **0 / 785** |
| invocations with `max_err > 0` | 48 / 854 (mutant 47, **reference 1**) |
| catches at scale 3.0 | 36–38 / 69, essentially flat in `n` |
| catches at scale 1e6 | 7 / 69 — these come from the `1e-6` **floor**, not the multiplier |

**The FP response is identically constant over nine decades of `scale` and the
whole range of `n`.** No formula relating `(n, scale)` to FP rate can be fitted
or falsified on this corpus, because the corpus has one relevant data point.
This is the same saturation the earlier `n_samples` round hit, now shown to
extend across the multiplier axis too.

The multiplier is identified only to an interval:

> At n=20, `scale` can range over **(1.642, 4.360)** with **no invocation-level
> verdict change anywhere in the corpus**. Lower edge:
> `flash_attention/drop_last_tile`. Upper edge: `gelu/sigmoid_approx`. The
> shipped 3.0 sits 0.26 decades from one edge and 0.16 from the other — near
> the middle, but by coincidence, since it predates any of this evidence.

One further consequence worth recording: at small `scale` the FP count stays 0
**because of the `1e-6` floor**, which sits only 8.4× above 1 ulp(float32) for
the single live reference case. **The floor, not the multiplier, is the binding
constraint on any attempt to tighten this check.**

---

## 7. Recommendation

**A real theorem exists (§3), but it is not a theorem about `3.0`.** It says the
adaptive tolerance is `Theta(sigma·L·sqrt(log m))` — a randomised estimate of the
reference's local Lipschitz constant — and it yields a one-sided detection
guarantee with assumptions that genuinely hold. That is defensible and
verifiable, and §2.2 and §5.1 are sharp, cheap, exactly-provable companions.

**But the specific ask — a bound on the mechanism's error rate — is not
available here, and §4 explains why in a way that will not be fixed by more
work on this check.** The FP direction is unprovable by construction; the catch
direction is provable but the guarantee is stated in terms of `L`, which the
checker never computes.

Two honest options, in order of preference:

1. **Go back to the advisor with the specific ask.** "Deterministic formula,
   bound, or predictive result" is satisfiable in the *completeness* direction
   (§3 corollary) and provably not in the *soundness* direction (§4). Those are
   very different papers. It is worth 10 minutes of their time to find out which
   one they meant before building either.

2. **If §3 is wanted anyway**, the paper-shaped version is a *characterisation*:
   "the adaptive tolerance is a Lipschitz-constant estimator, here is its exact
   order-statistic identity (§5.1), here is its two-sided bound (§3), here is
   the identifiability defect in its parameterisation (§2.2), and here is why no
   false-positive bound is obtainable (§4)." The negative in §4 is a genuine
   contribution and pairs naturally with the existing tolerance-invariance
   result, which is also a statement about what tolerances *cannot* do.

**One actionable side finding, independent of the paper.** §4's margin formula
says the band sits ~10³ above the float32 noise floor. There is real, quantified
headroom to tighten this check — and §6 says the `1e-6` floor becomes binding
first. That is a concrete, testable follow-up: lower `delta_scale` (equivalently
`scale`) and lower the floor together, and see whether catch improves before FPs
appear. The current corpus cannot answer it — it needs the near-miss candidates
the earlier `n_samples` round already identified as the missing ingredient.

---

## 8. Limits of this evidence

- **RESOLVED 2026-08-25 by `GPU_COVERAGE.md` — read that first.** §2 and §3 were
  measured on CPU `torch` references rather than the shipped Triton kernels. The
  cross-check reported here (`probes/validate.py`, "agreeing only within a factor
  0.41-1.67") was **my measurement error, not a CPU/GPU phenomenon**: it compared
  `verification/specs/*.valid_shapes` inputs against a corpus that uses entirely
  different, smaller shapes. Replaying the corpus's exact inputs
  (`probes/replay.py`) gives agreement of **0.87-1.11x on 27 of 29 operators**,
  and the sandwich holds against the **Triton-measured** tolerance on **210/210**
  evaluable invocations. Coverage is now 27 of 29 operators with two
  characterised exclusions. The caveat below about the sandwich being unverified
  on the shipped kernels is **withdrawn**.
- `L` in §3 is a Monte-Carlo per-row estimate (K=400), validated against exact
  `jacrev` row norms on three cases at **2.1% / 7.1% / 10.0%** error. It is
  biased slightly low for a maximum, which makes the reported upper-bound
  looseness slightly optimistic and the lower bound slightly conservative.
- §6's surface inherits the corpus's saturation: 806/854 invocations have
  `max_err` exactly 0. Everything in §6 rests on 48 live invocations, one of
  which is the entire reference-side evidence base.
