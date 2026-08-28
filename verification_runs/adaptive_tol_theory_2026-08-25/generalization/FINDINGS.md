# Structural `L`: yes. Universal `tol` formula from (σ, L, m): no. From the row-norm profile: yes.

**Measured 2026-08-25 on a Colab T4** (session `kccgen`, provisioned and stopped).
Probes `gen_native.py` (GPU), `fit_tol.py`, `gen_report.py`; raw data in `data/`.
**The checker's tolerance computation was not touched.** Derivation and
measurement only.

---

## Verdict up front

| question | answer |
|---|---|
| Can `L` be derived structurally instead of probed? | **Yes, for all 27.** Closed form matches the probed value to **0.994–1.018×** once the estimator's own bias is removed. 9 of 27 need only the *shape*. |
| Does `tol = f(σ, L, m, n)` fit with one set of constants? | **No. R² = −0.34** — worse than predicting the mean. Spread 3.5×, 5/27 within ±10%. |
| Is `tol` predictable at all without probing? | **Yes — from the closed-form Jacobian row-norm *profile*, with zero fitted constants. R² = 0.958**, 36/38 within ±10%. |
| What extra information does that need? | The whole profile `{‖J_i‖}`, not just its max (`L`) and count (`m`). |

The honest headline: **`L` is structural; `tol` is structural too, but not from the
theorem's three summary statistics — the theorem's leading term is a *bound*, and
it does not double as an estimator.**

---

## Part A — structural `L`

### A.1 The closed forms

`L = max_i ‖J_i‖₂`. Written down from each kernel's arithmetic, on the corpus's
own replayed inputs (`np.random.default_rng(0)`). `n` = reduced length,
`W` = pool window, `p` = softmax row, `z` = normalised row, `S = Σ|x|`.

| operator(s) | `‖J_i‖₂` | needs |
|---|---|---|
| `sum_reduction` | `√n` | **shape only** |
| `mean_reduction` | `√n / n` | **shape only** |
| `max_reduction`, `min_reduction` | `1` (one-hot) | **shape only** |
| `max_pool{1,2,3}d` | `1` | **shape only** |
| `avg_pool{1,2,3}d` | `√W / W` | **shape only** |
| `matmul` | `‖B[:,j]‖₂` | the operand |
| `gelu`, `swish` | `\|φ'(x_i)\|` | the input |
| `softmax` | `p_i √(1 − 2p_i + Σ_j p_j²)` | the input |
| `log_softmax` | `√(1 − 2p_i + Σ_j p_j²)` | the input |
| `l2norm` | `√(1 − u_i²) / ‖x_r‖`, `u = x_r/‖x_r‖` | the input |
| `l1norm` | `√(1 − 2\|f_i\| + n f_i²) / S`, `f = x/S` | the input |
| `frobenius_norm` | `√(1 − u_i²) / ‖x‖` (whole tensor) | the input |
| `layernorm` | `\|γ_i\| (v+ε)^{-1/2} √(1 − 1/n − z_i²/n)` | input + γ |
| `rmsnorm` | `(\|γ_i\|/r) √(1 − 2a_i + a_i c)`, `a_i = x_i²/(n r²)` | input + γ |
| `group`/`instancenorm` | layernorm form per (batch, group\|channel) row | input + γ |
| `batchnorm` | `\|w_c\| / √(rv_c + ε)` (diagonal) | the stats |
| `cross_entropy` | `m = 1`; `(1/N) √(Σ_r ‖p_r − e_{t_r}‖²)` | input + targets |
| `flash`/`causal`/`sdpa` | `‖(1/√D) Kᵀ (p_i ⊙ (V[:,d] − f_id))‖₂` | Q, K, V |

The attention entry is worth calling out: `f_id = Σ_j p_ij V_jd` depends only on
`Q_i`, so its gradient closes in one line. **The prior round used a loose
`(1/2√D)‖V‖₂‖K‖₂` bound for the attention family; that was unnecessary — the
exact form is above and it is what is measured below.**

### A.2 Structural vs probed, on the real Triton kernels

The probed `L` is `max_i` of `m` noisy per-row RMS estimates, so it is biased
**high** by roughly `1 + √(2 ln m)/√(2K)`. Raising `K` separates "the formula is
wrong" from "the estimator is biased":

| `L_mc(K) / L_struct` | min | median | max | within ±5% |
|---|---:|---:|---:|---:|
| K = 400 *(what the native run used)* | 0.986 | 1.081 | 1.123 | 9 / 27 |
| K = 4000 | 0.996 | 1.025 | 1.039 | **27 / 27** |
| K = 20000 | 0.994 | **1.010** | 1.018 | **27 / 27** |

**The closed forms are right.** The entire K=400 gap is estimator bias, and the
probe converges onto the derived value. The bias model predicts a median 1.122
against an actual 1.081 — the right size and direction, ~4% optimistic because
it assumes every row sits at the maximum, which only the shape-only operators do.

Per operator at K=20000 (`STATIC` = derivable from shape alone):

| operator | kind | m | `L_struct` | `L_mc20k/L_struct` |
|---|:---:|---:|---:|---:|
| `sum_reduction` | STATIC | 64 | 1.1314e+01 | 1.013 |
| `mean_reduction` | STATIC | 64 | 8.8388e-02 | 1.013 |
| `max_reduction` | STATIC | 64 | 1.0000e+00 | 1.016 |
| `min_reduction` | STATIC | 64 | 1.0000e+00 | 1.016 |
| `max_pool1d` | STATIC | 48 | 1.0000e+00 | 1.013 |
| `max_pool2d` | STATIC | 96 | 1.0000e+00 | 1.015 |
| `max_pool3d` | STATIC | 384 | 1.0000e+00 | 1.013 |
| `avg_pool1d` | STATIC | 48 | 5.0000e-01 | 1.010 |
| `avg_pool2d` | STATIC | 96 | 2.5000e-01 | 1.010 |
| `avg_pool3d` | STATIC | 384 | 3.5355e-01 | 1.014 |
| `matmul` | closed | 1024 | 5.6393e+00 | 1.011 |
| `gelu` | closed | 8192 | 1.1289e+00 | 1.011 |
| `swish` | closed | 8192 | 1.0998e+00 | 1.012 |
| `softmax` | closed | 8192 | 1.8536e-01 | 0.994 |
| `log_softmax` | closed | 8192 | 1.0213e+00 | 1.011 |
| `l1norm` | closed | 8192 | 1.1331e-02 | 0.998 |
| `l2norm` | closed | 8192 | 1.0307e-01 | 1.009 |
| `frobenius_norm` | closed | 8192 | 1.1142e-02 | 1.018 |
| `layernorm` | closed | 8192 | 3.0509e+00 | 0.998 |
| `rmsnorm` | closed | 8192 | 3.2778e+00 | 1.000 |
| `groupnorm` | closed | 256 | 2.3159e+00 | 1.009 |
| `instancenorm` | closed | 128 | 1.5084e+00 | 1.000 |
| `batchnorm` | closed | 256 | 1.6034e+00 | 1.009 |
| `cross_entropy` | closed | 1 | 1.2549e-01 | 1.005 |
| `flash_attention` | closed | 2048 | 6.2433e-01 | 0.997 |
| `causal_flash_attention` | closed | 2048 | 1.1187e+00 | 1.009 |
| `scaled_dot_product_attention` | closed | 2048 | 5.1131e-01 | 1.000 |

**Conclusion (A): `L` need not be probed.** Nine operators need only the output
shape; the other eighteen need one cheap pass over the input (and γ, B, or Q/K/V)
using formulas above. Agreement with the probe is 1–2% once the probe is given
enough samples to be unbiased.

---

## Part B — a predictive formula for `adaptive_tol`

Write the dimensionless shape factor

```
        y  =  tol / (3 σ L)
```

The theorem says `tol = 3σ · q95_n( max_i |⟨J_i/L, g⟩| ) · L`, so `y` should be a
function of the Jacobian's shape and `n` alone. `n` is fixed at 40 here; its
dependence is separately established and validated to ≤4.3% in `../FINDINGS.md` §5.

### B.1 From (σ, L, m) with one constant — **NO**

Fitted on all 228 native invocations:

| model | fit | R² |
|---|---|---:|
| M0 `y = c` (null) | `c = 2.952` | 0.0000 |
| M1 `y = a√(2 ln 2m) + b` | `a = 0.1965, b = 2.1985` | **0.0570** |
| M1′ `y = a√(2 ln 2m)` | `a = 0.7537` | **−0.4146** |
| M2 `y = a(ln m)^c + b` | `a = 2.221, b = 0.535, c = 0.05` | 0.0985 |

M2's fitted exponent of 0.05 means the optimiser drove it to a constant: **`m`
explains essentially none of the variance.** M1′ — the theorem's own leading term —
is *worse than predicting the mean*.

Per-operator residuals under M1′ span **3.53×**, and they are systematic, not
noise: within-operator spread of `y` is 1.18× against a between-operator spread of
2.02×. Worst under-prediction `softmax` −37%, worst over-prediction
`cross_entropy` +121%.

This is expected in hindsight and does not contradict the theorem. `√(2 ln 2m)` is
the max-of-`2m`-Gaussians **upper bound**; it is tight only when the `m` rows are
independent and equal in norm. Neither holds: `softmax`'s row-norm profile has a
max/median spread of **38.7**, and `cross_entropy` has `m = 1`, where the bound
(0.887) sits far below the half-normal reality (1.90). **A bound is not an
estimator.**

### B.2 From the closed-form row-norm profile — **YES**

Model M3 uses the profile `{‖J_i‖}` from Part A and simulates
`E[q95_40( max_i (‖J_i‖/L)|z_i| )]` under an orthogonal-rows assumption.
**No fitted constants at all.** Measured per matched invocation (n = 38, `y`
computed with the closed-form `L` so the estimator bias of §A.2 does not
contaminate the denominator):

| model | pred/meas min | median | max | spread | ±5% | ±10% | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 (σ, L, m) | — | — | — | 3.53× | — | 5/27 | **−0.343** |
| **M3 (row-norm profile)** | 0.929 | **1.022** | 1.167 | **1.26×** | **26/38** | **36/38** | **0.958** |

The residual is interpretable and signed as predicted: M3 assumes orthogonal
rows, so it **over-predicts exactly where rows are genuinely correlated**. The
four largest deviations are `flash_attention` (+17%, +10%, +9%, +8%) — whose
output rows share a softmax denominator — and the operators with truly disjoint
input windows (`layernorm` 1.002, `groupnorm` 0.999, `log_softmax` 1.015) land on
1.00.

> **CORRECTED 2026-08-27 (`../attention_gram/ATTENTION_GRAM.md`).** The
> correlation mechanism is real but its magnitude here was over-read: the
> exact-Jacobian Gram law puts attention's structural correction at **+3–4%
> median (max +7%)**, and the four deviations quoted above reproduce exactly
> as m3/meas = 1.166/1.104/1.075/1.096 while sitting at z = −1.66/−0.28/
> −0.24/−0.73 under the exact law — i.e. **single-40-sample-draw noise around
> a small true correction**, not a +17% systematic. The genuinely large
> correlation case is the scan family (+24.7%, closed form in
> `../../theory_audit_2026-08-27/`).

**Conclusion (B): `adaptive_tol` is predictable without probing, to ±10% on 36 of
38 invocations, but the predictor is the Jacobian's row-norm *profile*, not the
triple (σ, L, m).** Collapsing the profile to its max and its length throws away
exactly the information that determines the answer.

### B.3 Sample-count-to-stabilise

This composes but was **not validated end-to-end this pass.** `../FINDINGS.md` §5
already gives the exact, distribution-free effective quantile
`p_eff(n) = (0.95n + 0.05)/(n+1)` and a one-parameter model for `tol_n/tol_40`,
validated to ≤4.3% median error on 804 banked vectors — but that model takes its
shape parameter from the *measured* sample CV. M3 now supplies that parent
distribution structurally, so the two chain into a probe-free prediction of the
whole `n` curve. **Running M3 at n ∈ {2 … 40} and checking it against the banked
prefix curve is the obvious next step and was not done here.** Do not claim the
`n` dependence is structurally validated until it is.

> **DONE 2026-08-28 — `../../theory_closure_2026-08-28/FINDINGS.md` §1.**
> The chain is validated end-to-end on all 228 replayed native invocations:
> the structural parent's DIRECT curve matches the measured aggregate to
> 0.1% at the shipped n = 20 (0.9841 predicted vs 0.9852 measured) and
> per-invocation deviations are z ~ N(0,1) against the parent's own
> single-draw noise. The DIRECT route also beats this section's Gumbel
> one-parameter model (whose −2.3% aggregate bias at n = 20 is a model
> artifact). The scan family needs the exact Brownian parent (M3's
> independent parent shows +7.5% at n = 2 and half the measured CV; the
> Brownian parent is within 1.9% everywhere) — the chain is validated WITH
> the correct Gram parent, exactly as theory_audit H1 requires.

---

## What this means for the paper

Two claims are now available that were not before:

1. **`L` is derivable, not probed** — closed form for all 27 in-scope operators,
   shape-only for 9 of them, agreeing with a converged probe to 1–2%. This
   removes the Monte-Carlo estimate that the previous round listed as a residual
   limitation, and it makes the detection-guarantee corollary of `../FINDINGS.md` §3
   **computable in advance** rather than only after measurement.
2. **`adaptive_tol` is predictable from the Jacobian, with no fitted constants** —
   R² = 0.958, ±10% on 36/38 — provided the full row-norm profile is used.

And one negative, stated as plainly as the false-positive-rate result was:

3. **There is no universal `tol = f(σ, L, m, n)` with one set of constants.**
   R² = −0.34 against the theorem's own leading term; 3.53× residual spread;
   `m` explains ~6% of the variance at best. The honest phrasing is
   **"verified per-operator, and predicted per-operator from its Jacobian"** —
   not "predicted by one universal formula". `√(2 ln 2m)` remains correct as the
   bound it was proved to be, and should be presented only as that.

---

## Limits

- **Part A's closed forms were checked against a probe, not against an
  independent exact Jacobian.** The probe converges onto them (1–2% at K=20000),
  which is strong, but an exact `jacrev` cross-check on the small-`m` cases would
  make it airtight and is cheap. The prior round did this for three CPU cases at
  2.1% / 7.1% / 10.0%; it has not been repeated against these closed forms.
- **One invocation per corpus entry** (j=0) carries Part A and the matched Part B
  comparison — 38 points, not 228. The unmatched 228-point version of Part B is in
  `fit_tol.py` output and gives the same M1 verdict (R² = 0.057 / −0.415).
- **M3 is a simulation, not a closed form.** It needs `NSIM × n × m` Gaussian
  draws. That is far cheaper than probing the kernel, and needs no GPU, but it is
  not an equation you can write on one line.
- **`groupnorm` and `instancenorm` shift substantially between invocations**
  (γ is redrawn per invocation), so per-operator medians mix configurations; the
  matched per-invocation comparison in §B.2 is the one to trust. An earlier
  ratio-of-medians pass produced spurious 1.59× / 0.80× outliers for exactly this
  reason.
- Attention's closed form is verified only on the corpus's ordinary inputs. The
  saturating and fp-floor adversarial inputs characterised in `../GPU_NATIVE.md` §4
  are outside the linear regime, so a Jacobian-based prediction is not expected to
  hold there and was not tested.
