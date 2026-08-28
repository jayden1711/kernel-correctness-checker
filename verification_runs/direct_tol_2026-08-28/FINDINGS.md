# Theory-driven latency: the closed form genuinely replaces the measurement — verdict-identical on the full corpus, E/draw ratio p50 = 1.008 — and the honest wall-clock answer is NEGATIVE (+18% checker wall), because the probe's "32.9% of check time" was a serialisation artifact; its pipelined cost is ≲0.3 s per corpus pass

**Built, validated and GPU-measured 2026-08-28** (T4 session `dtol`,
stopped). Implementation: `KCC_STRUCTURAL_MODE=direct` in
`verification/layer2_numeric_oracle/structural_l.py` (**default unchanged
and OFF**; the shipped probe path is untouched). Probes, driver, analysis
and all arm JSONs in `probes/` and `data/` (GPU artifacts under
`data/gpu/`). Pre-registered predictions banked before the run
(`data/preregistered_direct_predictions.json`).

This is the Part-2 question of the round — *does the theory enable a
latency lever empirical optimization couldn't find?* — pursued on its
strongest candidate and answered with a measured no, plus two
arithmetic-bounded nulls (§5) and one methodological correction that
matters beyond this round (§3).

## 1. What was built: the B.3 DIRECT route as a production estimator

The structural-L round rejected the closed-form tolerance because M3 is a
simulation (NSIM·n·m Gaussian draws; 1128× the probe). The theory-closure
round then validated the **DIRECT** route — the parent CDF
F(t) = ∏ᵢ(2Φ(t/wᵢ)−1) evaluated exactly. This round turns that into a
deterministic estimator with **no simulation anywhere**:

    E[q95_n] = Σ_k wt_k · ∫ (1 − G_k(t)) dt,
    G_k(t) = Σ_{j≥k} C(n,j) F(t)^j (1−F(t))^{n−j}

(torch.quantile's interpolation weights, by linearity of expectation),
with two controlled cost devices: log-binning of the profile (192 bins,
±0.14% multiplicative bracket) and truncation of rows below w_max/4
(dropped factor ≥ 1−1e-30 over the mass region), on a two-pass windowed
512-point grid. Validated three ways before the GPU:

- vs the M3 simulation at NSIM = 60 000 over six adversarial profile
  shapes: worst deviation **0.09%** (production transcription 0.15%) —
  inside MC error;
- vs the banked 228-invocation native bank (bit-exact input replay):
  predicted tol vs measured q95₂₀ at **R²(log) = 0.997, ratio p5/p50/p95
  = 0.88/1.01/1.14** — the residual is the draw's own noise (z ~ N(0,1),
  theory_closure);
- cost: **0.5–1.2 ms/call** on the dev machine, **0.9–2.2 ms** on the T4
  VM's CPU (vs 27–4200 ms for the M3 simulation). The 1128× objection is
  gone.

Scope under `direct`, derived not guessed: attention family excluded (the
onset law proves the response is not Jacobian-generated past onset; the
gram round measured six-decade per-delta spreads there), scan family
excluded (H1: the independent-rows parent over-predicts the level +24.7%,
the unsafe direction), argmax/argmin already out. Kink understatement
bounded by g(p) ≤ 1.44 (safe direction). Fail-closed: any exception in
the direct path declines to the probe — a wiring lesson paid for honestly:
the FIRST arm shipped with a device bug (CPU grid vs CUDA profile) whose
exception failed the check instead of declining, producing 160/200
reference FPs; the failure mode is now impossible by construction and the
arm was rerun from scratch.

## 2. GPU A/B — the swap is semantically real

Arms under `KCC_ABLATION_SEED=1`, same harness as every corpus round
(A = shipped probe; D = direct):

- **A: 40/40 catch, 0/200 FP. D: 40/40, 0/200.** Every wall-clock rep of
  both arms identical on catch/FP.
- **Verdict diffs: 0. Failing-check-set diffs: 0** across all 240 trials'
  records — swapping a random q95-of-20 draw for its parent mean moves no
  outcome anywhere on the corpus (the 461× median margins predict this;
  now it is measured).
- **Coverage**: of 854 perturbation-routed records, 186 are excluded ops
  (attention, argmax/argmin); **all 668 others took the direct path**
  (605 with tol visibly different from A's draw; 63 floor-clamped in both
  arms; zero off-floor fallbacks).
- **E/draw ratio over the 605 direct-taken records: p50 = 1.008**,
  p5/p95 = 0.89/1.27. The p95 tail is a single identified class:
  near-zero-variance variants of group/instancenorm where the parent's
  eps-regularized Jacobian sits above the measured (fp-cancellation-
  floored) response — parent tol 4e-6–7e-6 vs measured floor 1e-6, loose
  by ≤ 7× on records whose floor has a proven 6400× safe interval
  (tol_floor round). The taxonomy's resolvability boundary, seen from a
  third instrument.

## 3. The wall-clock answer, and the correction that outlives it

Five interleaved reps per arm, timing flag OFF (the honest convention):

| arm | checker wall, 5 reps (s) | median |
|---|---|---|
| A (probe) | 7.17 7.49 6.99 7.43 7.04 | **7.17** |
| D (direct) | 8.30 8.46 8.50 8.48 8.34 | **8.46** |

**D is 1.29 s SLOWER (+18.0% checker wall ≈ +2.9% corpus), far outside
the rep spread (A 7.0%, D 2.3%).** Mechanism, and the finding that
generalizes:

- The direct path costs ~1.9 ms/call net (668 calls ≈ +1.3 s — matching
  the VM CPU measurement plus a forced device sync per call), all
  host-side and un-overlappable.
- Therefore the probe path it replaced costs **≲ 0.1–0.3 s wall per
  corpus pass** — its 20 launches per call pipeline to nearly free.
  Under `KCC_CHECK_TIMING=1` the same step bills 2.74 s (45% of this
  round's serialised check time; the structural round's "2085 ms = 32.9%
  of check time, the biggest line item found so far" was the same
  artifact). **Serialised per-check shares overstate launch-dominated
  steps by ~30× here**; every timing-flag caveat in prior FINDINGS
  ("shares meaningful, absolutes are upper bounds") deserves this
  concrete number attached.
- The idealised ceiling the structural round computed from those shares
  (−24.5% checker wall) never existed as wall time. The real prize for
  ANY probe replacement — closed-form or otherwise — is bounded by the
  probe's pipelined cost, ~1–4% of checker wall ≈ **≤0.7% of corpus**,
  which no replacement with per-call host cost above ~0.3 ms can collect.

Answer to the round's question: **the theory did enable the closed form
(it is correct, validated, and cheap in the sense that mattered to the
old objection) — and the latency it was meant to save turns out not to
exist.** Null result, reported as such; the default stays the probe.

## 4. Near-miss response under the deterministic tolerance (pre-registered)

The response-law round predicts what removing draw noise does to the
boundary; predictions were banked BEFORE the run. Measured m-series catch
counts under D, against those predictions (10 seeds; binomial n = 10):

| op | m080 pred/obs | m100 pred/obs | m125 pred/obs |
|---|---|---|---|
| layernorm | 0% / 0 | 31% / 2 | 100% / 10 |
| softmax | 6% / 0 | 42% / 5 | 98% / 10 |
| gelu | 1% / 0 | 48% / 6 | 100% / 10 |
| l2norm | 0% / 0 | 27% / 2 | 100% / 10 |
| sum_reduction | 8% / 2 | 46% / 4 | 91% / 7 |

All 25 points inside binomial range; the m125 sharpening vs the probe arm
(softmax 8/10 → 10/10, layernorm 10/10 with pred 100% vs 99.5%) is the
predicted disappearance of tol-draw noise. The v-series is **identical
between arms except one floor-adjacent gelu seed** (v080 1/10 → 0/10) —
exactly what the binding-check law requires: layernorm/l2norm/sum bind on
deterministic comparators the estimator cannot touch, and softmax/gelu
bind on the floor, which both estimators clamp identically. The near-miss
family did its job: the boundary was watched, and it moved only where and
how the theory said it could.

## 5. The other two levers, closed by arithmetic (no GPU spent)

From the banked current arms (`failfast_ceiling.py`, serialised shares —
which §3 shows are UPPER bounds, making these ceilings generous):

- **Binding-ordered fail-fast within Layer 3** (order each op's battery by
  predicted δ*, stop at first fail; verdict-identical by construction):
  post-first-fail L3 time on the 22 L3-failing trials is 0.21 s = 3.46%
  of serialised check time ≈ **≤0.56% of corpus**. Declined: below any
  measurable resolution (single-run noise ~2–3%).
- **Dropping precision_coercion's fp16 arm** (licensed by the l3_margins
  dead zone; can only add catches): ~5.1% of serialised check time ≈
  **≤0.83% of corpus**. Declined for the same reason.
- **The 1e-6 floor slack licenses no cheaper computation**: the floor is a
  free max(); its (5e-7, 6.4e-3) safe interval buys tolerance-setting
  freedom (§2's ≤7× floor-adjacent looseness spends some of it), not
  compute. `n_samples` reduction is superseded by §3: the samples it
  would remove pipeline to ~free, which is why that round measured only
  2–3% for removing three quarters of them.

## 6. What survives as useful

- `KCC_STRUCTURAL_MODE=direct` stays in the tree as an **instrument**: a
  deterministic, draw-noise-free tolerance validated verdict-identical on
  this corpus — the right arm for boundary experiments (the near-miss
  family's step-response in §4 is sharper under it), not for latency.
- The E[q95_n] integral (`e_q95_direct`) is the engine the binding-check
  law and response-curve law run on (`../binding_law_2026-08-28/`,
  `../response_law_2026-08-28/`) — the same object doing derivation work
  even though it lost the latency race.
- The §3 serialisation correction retroactively adjusts how every
  KCC_CHECK_TIMING share in earlier rounds should be read.

## Limits

- One GPU class (T4), one corpus, 5 wall reps per arm; the +18% delta is
  9× the D-arm spread but the probe-cost bound (≲0.3 s) is inferred from
  the arm difference, not measured in isolation.
- The E/draw validation covers this corpus's input mix; the ≤7×
  floor-adjacent looseness is bounded by this corpus's floor slack, and
  any future corpus with sub-6.4e-3 floor-guarded catches must re-check
  it (the near-miss method applies directly).
- Wall reps share one VM; contention drift is bounded by the interleaving
  and the visible rep spreads.
- The verdict-identity result is for the corpus's 461×-margin population
  plus the near-miss families; candidates living exactly at the boundary
  flip with the estimator's ±10% draw noise by design (that is what §4
  measures).

## Reproduce

```bash
# local validation + pre-registration
.venv/bin/python probes/direct_e.py
.venv/bin/python probes/preregister_direct.py
# GPU (T4): upload kcc11.tgz + probes/{probe_redundancy.py,directab.sh}
#   + near_miss_gpu.py + v_series_gpu.py; nohup bash directab.sh; DONE
# offline:
.venv/bin/python probes/analyze_directab.py      # data/analysis.log
.venv/bin/python probes/tol_ratio_coverage.py    # coverage + E/draw
.venv/bin/python probes/failfast_ceiling.py      # §5 ceilings
```
