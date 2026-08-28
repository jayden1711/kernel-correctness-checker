# The attention saturation-onset law: the per-delta ratio is the parameter-free pushforward of a derived Gaussian through φ(u) = (1−e^{−u})/u — the label revision is now derived, the "scatter" is the law's own random variable, and attention leaves scope through the floor screen, not the Gram screen

**Derived and validated 2026-08-28, CPU emulation + banked GPU arms.**
Probes `probes/onset_law.py`, `probes/onset_law_2.py`,
`probes/compare_banked.py`; data and logs in `data/`. Replaces the
single-parameter exponential-onset model of
`../gram_screen_2026-08-27/FINDINGS.md` §3 (which its own §7 demoted to "a
consistency check, not a law").

## The law

Setup, exactly as shipped: corpus attention at (N, D) = (64, 32) with
Q, K, V ~ N(0,1) entries; `large_magnitude_qk` scales Q and K by κ = 20, so
logits ℓ = (κQ)(κK)ᵀ/√D are κ²·(unit-variance Gaussians). The perturbation
check perturbs **only the primary input Q** with d = randn · δ · std(κQ),
δ = 1e-3, and the Gram screen measures r_k = ‖f(x+d_k)−f(x)‖∞ / ‖J d_k‖∞
on the same delta.

Each delta perturbs row i's logits by ~iid N(0, τ_e²) with τ_e = δκ²
(the two κ factors: one from std(κQ) in the delta, one from κK in the
logit). The ∞-norm is dominated by the row with the smallest top-2 logit
gap g (response ∝ e^{−g}), where softmax is a two-state logistic. For
g ≫ 1:

    s_meas ∝ e^{−g}·|1 − e^{−δg}|·|ΔV|   (finite difference)
    ‖Jd‖  ∝ e^{−g}·|δg|·|ΔV|             (exact derivative)

    ⇒  r = (1 − e^{−δg})/δg =: φ(δg),   δg ~ N(0, τ²),  τ = √2·δ·κ²

Everything cancels except the top-2 gap perturbation δg: **the per-delta
ratio distribution is the pushforward of N(0, τ²) through φ, with
τ = √2·10⁻³·κ² — no fitted parameter anywhere** (at κ = 20, τ = 0.5657).
The old model — one fitted a per record in φ(a) — is this law with the
δg-distribution collapsed to a point. That is exactly why it matched rank
and magnitude but not per-record values, and why the "paired scatter" was
never noise around a per-record truth: **each delta draws its own
a_k = −δg_k, so the scatter is the law's random variable.**

## Validation (attempted falsification, per the standing standard)

| test | result |
|---|---|
| **T1 per-delta** (the strong form: correlate measured log r_k with φ(δg_k) computed from the record's own dominant row, delta by delta) | corr **0.81** (sdpa) / **0.86** (causal), n = 3000 each; residual sd 0.084/0.070 dex against a law spread of 0.14 dex — the law explains ~⅔–¾ of the per-delta variance record-by-record, not just distributionally. |
| **T2 pooled quantiles at κ=20** | Three-way match, no parameters: analytic pushforward P05/P50/P95 = 0.65/1.00/1.65; CPU ensemble 0.69/0.99/1.55 (sdpa); **banked GPU arm 0.70/1.05/1.58** (sdpa, 100 ratios) and 0.68/0.96/1.51 (causal, 80). |
| **T3 the Gram statistic** (per-record median of 20) | Ensemble P05–P95 = 0.88–1.23, extremes 0.76–2.28. The nine banked GPU medians 0.878–1.34 all sit inside; the banked 1.34 is ~P97 of the law's median distribution — one draw in nine at that quantile is unremarkable, not structure. |
| **T5 the 2026-08-26 ladder defects, rederived** | Same δg's through the defect functional give per-record medians P05–P95 = **12.0–26.7%**; the banked arm-D defects on these classes were **6.6–27.7%**. (The 6.6% record sits slightly below the law's P05 — noted, not hidden; it is the mildest-g record.) |
| **T4/F1 g-independence** | The law says the ratio distribution is independent of the record's own gap once saturated. First ensemble showed corr(median, g_min) = +0.75 on n=19 (sdpa) — but it **did not replicate** (+0.03 at n=27, fresh seed), and the cross-row-competition hypothesis for it was **refuted directly**: restricting both norms to the single dominant row leaves the causal-side correlation unchanged (+0.37 → +0.32). Verdict: no reproducible g-dependence; the original signal was a small-n fluctuation. |

## What is now derived rather than measured

1. **The label revision.** At κ = 20 the derived τ = 0.566 puts 95% of
   per-delta ratios in [0.65, 1.65] and the median-of-20 within a few
   percent of 1 — an order of magnitude inside both the pre-registered 2×
   Gram line and the checker's own 3× scale factor. In 400 fresh ensemble
   records, exactly **one** median crossed 2× — and that record sat at
   s/ulp = 0.2, i.e. it is **floor-gated before the Gram flag is
   consulted** (the banked causal ref3, s/ulp = 6, is the same phenomenon).
   "Saturation onset, below any defensible flag line" is therefore a
   theorem of the input distribution, not a description of ten records.
2. **The paired scatter.** Both 2026-08-26 defects and 2026-08-27 ratios
   are order statistics of functionals of the same centered δg
   distribution, taken over *different* delta draws. The per-record values
   are not predictable even in principle; only their distributions are —
   and both are, correctly (T2, T5).
3. **The scope-exit scale and mechanism.** The Gram median concentrates at
   φ(0) = 1 by symmetry at every κ, so it is (near-)blind to this
   mechanism; the exit channel is the **fp32 floor**: s ∝ e^{−κ²·G_min}
   with G_min the minimum row top-2 gap of the unit-variance base logits,
   so ln s falls quadratically in κ with an invocation-to-invocation spread
   of κ²·spread(G_min) — which is why the banked s/ulp values span 6 to
   2.3×10⁶ at a single κ. Measured exit curve (n = 200/point, F2):

   | κ | 15 | 20 | 25 | 30 | 40 | 50 |
   |---|---|---|---|---|---|---|
   | sdpa floor-flag % | 0 | 1.5 | 3.0 | 16 | 24 | 46 |
   | causal floor-flag % | 0 | 2.0 | 9.5 | 18 | 29.5 | 56.5 |

   Onset begins at the corpus's own κ = 20 (predicted 1.5–2%, banked
   observed 1/10), the causal variant leads (fewer effective keys per
   row), and the median invocation exits around **κ ≈ 45–50**. The Gram
   median only starts firing at κ ≳ 30 and always trails the floor
   (measured 0% at κ ≤ 25; 21% vs 41% floor at κ = 50 on the non-floored
   remainder).

## Negatives, plainly

- The per-delta residual (0.07–0.08 dex) is **not derived**: it contains
  the mild-g correction to the two-state tail form and whatever multi-row
  effects survive; the closed form covers the dominant term only.
- A causal-side asymmetry — P(median > 1) ≈ 0.70 in the saturated subset,
  unchanged by the single-row restriction — is real, reproducible, and
  **unexplained**. It is bounded (medians ≤ 1.23 at P95) and does not
  threaten any flag line, but the law does not produce it.
- There is **no clean closed form for the median-of-20 flag statistic**
  with all corrections; the operative form is Monte Carlo of the law
  (seconds, deterministic seed). Reported as the task requested: the
  per-delta law is closed-form and parameter-free; the record-level
  statistic is its order statistic, computed numerically.

## Limits

- CPU fp32 softmax emulation stands in for the GPU online-softmax kernel
  on the measured side; the banked-arm agreement in T2 (GPU vs law within
  ~0.05 dex at every quantile) is the evidence this is safe at these
  margins. Floor-fraction points below ~5% carry n=200 binomial error
  (±~2%).
- The banked sample is 9 medians / 180 ratios; every distributional claim
  about it is correspondingly coarse, and is backed by the 150–400-record
  ensembles instead.
- τ's derivation assumes independent top-1/top-2 logit perturbations
  (cov ~ K_a·K_b/D ≈ 0); exact at the corpus shapes to O(1/√D).

## Reproduce

```bash
.venv/bin/python probes/onset_law.py       # law + T1..T6, writes data/onset_law.json
.venv/bin/python probes/onset_law_2.py     # F1 mechanism refutation, F2 causal sweep
.venv/bin/python probes/compare_banked.py  # banked GPU arm vs analytic pushforward
```
