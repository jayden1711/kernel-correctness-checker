# Structural `L` as a latency lever: the premise has three factual errors, and the corrected ceiling is 4.0%

**Measured 2026-08-26.** GPU numbers are re-derived from the banked
`KCC_CHECK_TIMING` arms of `../n_samples_curve_2026-08-25/` (Colab T4, warm
Triton cache). Cost and accuracy of the new path measured on CPU here — **no
GPU was available this pass and the 40-mutant/200-reference ablation was
therefore NOT run.** Said plainly up front because it bounds everything below.

Implementation `verification/layer2_numeric_oracle/structural_l.py`, wired into
`perturbation.py` and `checker.py` behind `KCC_STRUCTURAL_L=1`.
**The flag is OFF and the shipped default is unchanged and verified bit-identical.**

---

## Verdict up front

| question | answer |
|---|---|
| Does the checker spend K=400 launches per candidate on L? | **No.** It spends `n_samples=20` per perturbation-routed call, and it never computes `L` at all. |
| Is there a closed-form `tol` to swap in? | **No.** The derivation tested exactly that and got R² = −0.34. What predicts `tol` is a *simulation*, not a formula. |
| Measured cost of the probing step | **2085 ms = 32.9% of check time, 23.9% of checker wall** — real, and the biggest single line item found so far. |
| Ceiling if the replacement were free | **−24.5% checker wall = −4.0% of corpus runtime** |
| Cost of the actual replacement | **1128× the path it replaces** (CPU, biased in its favour). Cacheable for ~10 of 27 operators; not for `softmax`. |
| Is this a meaningful win? | **No. It is another ≤4% checker-logic lever, and that 4% is the idealised ceiling, not a measured delta.** |

---

## Three errors in the premise, all material

Stated first because each one changes what the task even is.

**1. There is no K=400 in the checker.** `K = 400` appears in exactly one place
in this repository: `K_LIST = [400, 4000, 20000]` in
`../adaptive_tol_theory_2026-08-25/generalization/gen_native.py`, an offline
derivation probe. It is the sample count that probe used to *estimate* `L` in
order to check the closed forms against it. The checker draws **20**
perturbation samples per call (`perturbation.py`, `n_samples: int = 20`).

**2. The checker never computes `L`.** `grep -riE 'lipschitz|L_struct|jacobian'`
over `verification/` returns nothing but unrelated L1/L2 row-norm property
checks. The shipped tolerance is

```
adaptive_tol = max(3.0 * quantile(sensitivities, 0.95), 1e-6)
```

`L` does not appear. So "replace the Monte-Carlo L estimation with the
closed-form L" has no production referent — there is no `L` estimation to
replace. What exists is a Monte-Carlo estimate of **`tol` itself**.

**3. `L` closed-form does not give you `tol`.** This is the one that matters.
`../adaptive_tol_theory_2026-08-25/generalization/FINDINGS.md` §B.1 tested
`tol = 3σL·√(2 ln 2m)` — the theorem's own leading term — across 228 native
invocations:

> M1′ `y = a√(2 ln 2m)` … **R² = −0.4146** … *worse than predicting the mean.*
> **A bound is not an estimator.**

The thing that *does* predict `tol` (R² = 0.958) is model **M3**: a Monte-Carlo
simulation of `E[q95_n(max_i (‖J_i‖/L)|z_i|)]` over the whole closed-form
row-norm profile, and the same document flags it — *"M3 is a simulation, not a
closed form. It needs NSIM × n × m Gaussian draws."*

So the swap on offer is not "probe → formula". It is **"probe the kernel 20
times → simulate the profile 3000 × 20 × m times"**, and whether that is
cheaper is an empirical question nobody had asked. It is the question this pass
answers, and the answer is no.

---

## Measurement 1 — what the probing step actually costs

Isolated by regressing per-check `duration_ms` on `n` across the six banked
arms (n = 3, 5, 10, 15, 20, 40), all run under `KCC_CHECK_TIMING=1` and
`KCC_ABLATION_SEED=1` in one T4 session. Every perturbation-routed check
decomposes as `duration(n) = a + b·n`, where `b·n` is the sensitivity loop and
`a` is the fixed remainder (base reference call, candidate call, quantile,
device transfer, sync). The slope recovers the probing cost **without
assuming** it. `isolate_probe_cost.py`, `wall_regression.py`.

```
OLS   pert_path_ms(n) = 765.8 + 104.245 · n      R² = 0.9969
      checker_wall_ms(n) = 6346.1 + 101.176 · n  R² = 0.9801
      per-sample cost = 0.1218 ms / sample / call   (856 calls, 240 candidates)
```

At the shipped `n_samples = 20`:

| quantity | value |
|---|---:|
| sensitivity-loop (probing) cost | **2085 ms** |
| …as a share of the perturbation path (2944 ms) | **70.8%** |
| …as a share of all instrumented check time (6335 ms) | **32.9%** |
| …as a share of instrumented checker wall (8474 ms) | **23.9%** |

Per-check shares reproduce `../check_timing_2026-08-25/` exactly
(`weight_magnitude` 11.8% vs 11.6%, `adversarial_*` 35.4% vs 35.5%), which is
the cross-check that this decomposition is reading the same run correctly.

**Translated to corpus runtime**, using the denominator every prior round used
(60.8 s warm corpus; 9.89 s perturbation-bearing checker portion):

> **Eliminating the probing step entirely, with a free replacement, is
> −24.5% checker wall = 2.4 s = −4.0% of corpus runtime.**

That is a hard ceiling. Nothing in this direction can beat it.

---

## Measurement 2 — the replacement is 1128× more expensive than what it removes

`cost_probe.py`, CPU, corpus-matched shapes, `NSIM = 3000`, `n_samples = 20`.

**The direction of the bias matters:** a CPU torch reference launch is far
slower than the T4 Triton launch the checker actually pays (banked at
0.1218 ms/sample/call), so the Monte-Carlo arm is *handicapped* here. The
structural path loses anyway, and would lose by more on the GPU.

| operator group | MC path | structural path | ratio |
|---|---:|---:|---:|
| all 27 | 52.3 ms | 59 053 ms | **1128×** |
| 9 shape-only | 26.3 ms | 5 968 ms | 227× |
| 18 input-dependent | 26.0 ms | 53 085 ms | 2039× |

**The closed forms themselves are free — the simulation is the entire cost.**
Computing the row-norm profile costs 0.00–0.05 ms for 24 of 27 operators.
Part A of the derivation is fully vindicated as *cheap*. It is Part B's
estimator that is not:

| operator | profile | M3 | MC total |
|---|---:|---:|---:|
| `softmax` | 0.04 ms | **4122 ms** | 1.84 ms |
| `layernorm` | 0.05 ms | **4084 ms** | 1.71 ms |
| `matmul` | 0.01 ms | **511 ms** | 0.38 ms |
| `flash_attention` | 0.95 ms | **1062 ms** | 1.15 ms |

Device-independent arithmetic, which needs no hardware to settle:

```
Monte-Carlo path :       5,043,200 Gaussian draws
structural path  :   7,033,020,000 Gaussian draws     = 1395× more
```

One detail worth flagging against the derivation's claim that the 18 need only
*"one cheap pass over the input"*: for the three attention operators the pass
is a **Python loop over N rows** and costs 0.93–0.97 ms — comparable to the
*entire* Monte-Carlo call it is meant to replace, before M3 runs at all.

---

## Measurement 3 — accuracy, including the regime the derivation excluded

The derivation validated the closed forms on ordinary random inputs and
explicitly disclaimed the rest:

> *"The saturating and fp-floor adversarial inputs … are outside the linear
> regime, so a Jacobian-based prediction is not expected to hold there and was
> not tested."*

**That disclaimer covers 76.5% of the probing time.** From the banked n=20 arm,
of 844 perturbation-routed calls, 634 are `adversarial_*` variants —
`large_magnitude`, `near_zero_variance`, `extreme_dynamic_range`,
`zero_variance_rows` — accounting for 2237 ms of 2922 ms.

So this pass tested it. `regime_probe.py` compares `tol_struct / tol_mc` on
each spec's **own** `get_adversarial_inputs`, CPU, 11 operators:

| regime | n | min | median | max | ±10% | 2× |
|---|---:|---:|---:|---:|---:|---:|
| ordinary (validated) | 11 | 0.971 | **1.008** | 1.106 | 10/11 | 11/11 |
| adversarial (previously untested) | 20 | 0.673 | **1.000** | 1.069 | **19/20** | 20/20 |

**This is a positive result and it deserves to be recorded as one: the closed
forms hold on the adversarial inputs better than the derivation's caveat
predicted.** One outlier, `l1norm/second_half_dominant` at 0.673 — a 33%
*tighter* band, which is false-positive risk, not lost-catch risk. Two
saturating `softmax` variants (`max_in_last_tile`, `extreme_range`) produce an
all-zero profile and the path correctly **declines**, falling back to the probe.

The scope of that result is 11 of 27 operators, on CPU, against the torch
reference rather than the Triton kernel. It is corroboration, not the ablation.

---

## Measurement 4 — can M3 be cached? Mostly, but not for `softmax`

If `y` is constant per (operator, shape), the simulation is paid once per
corpus run and the 1128× collapses. This is the strongest form of the proposal
and dismissing it unmeasured would repeat the previous round's projection
error. `cacheability.py`, 8 independent draws per operator, simulator seed
pinned so any spread is the profile moving and not the simulator:

| operator | y spread | CV | cacheable |
|---|---:|---:|---|
| `matmul`, `sum_reduction`, `max_reduction`, `frobenius_norm` | 1.0000 | 0.00% | yes, exactly |
| `gelu`, `swish`, `layernorm`, `rmsnorm`, `l2norm`, `log_softmax` | 1.005–1.031 | ≤0.94% | yes |
| `l1norm` | 1.055 | 1.57% | marginal |
| **`softmax`** | **1.300** | **9.46%** | **no** |

So a cache would work for most operators and would genuinely remove the cost
blocker for them — at the price of a second approximation on top of M3's own
±10%, and it does not work for `softmax`, whose `y` moves 1.30× between inputs.
**Even granting the cache everywhere, the ceiling is still Measurement 1's
−4.0%**, because a cache cannot save more than the thing it replaces cost.

---

## What the ceiling looks like once the caveats are applied

| scenario | checker wall | corpus |
|---|---:|---:|
| **A** all 27 ops, ordinary + adversarial, free replacement | −24.5% | **−4.0%** |
| **B** all 27 ops, ordinary inputs only | −5.5% | −0.9% |
| **C** 9 shape-only ops, cached, ordinary + adversarial | −1.8% | −0.3% |
| **D** 9 shape-only ops, cached, ordinary only | −0.8% | −0.1% |

A is the idealisation. Measurement 3 makes A defensible on accuracy grounds in
a way it was not before this pass — but A still assumes a zero-cost estimator,
and Measurement 2 says the estimator costs 1128× what it replaces unless the
cache of Measurement 4 works, which it does not for `softmax`.

---

## Verdict, against the numbers already in hand

**This is not a meaningful win. It is another small one, and the pattern holds.**

| lever | corpus saving | measured? |
|---|---:|---|
| shipping a warm Triton cache | **75%** | yes |
| **structural `L` (idealised ceiling)** | **≤4.0%** | ceiling only |
| `n_samples` 20 → 5 | 3.2% | yes |
| `n_samples` 20 → 10 | 2.0% | yes |
| removing `check_weight_magnitude` entirely | 1.9% | yes |

Three things make the ≤4.0% weaker than the ≤3% entries above it, not stronger:

1. **It is a ceiling, not a delta.** Every other row is an arm that was
   actually run. This one assumes the replacement is free, and it is not.
2. **It is dominated by a cheaper lever already on the table.** `n_samples`
   20 → 5 gets 3.2% of the same 4.0% by deleting three quarters of the same
   loop, with no new module, no simulation, no cache, and no second
   approximation layered on the tolerance.
3. **The noise floor is ~2–3%** (`../check_timing_2026-08-25/`: arm D removed
   strictly more work than arm B and measured 1.9% *slower*). A 4.0% ceiling
   is barely two noise floors wide before any of its cost is subtracted.

**Recommendation: leave `KCC_STRUCTURAL_L` off, and do not pursue this for
latency.** The flag stays as instrumentation.

What this pass *did* produce that is worth keeping is not a latency result at
all — it is Measurement 3. The closed forms reproduce the probed tolerance to
±10% on 19 of 20 adversarial invocations, in the regime the derivation
explicitly declined to claim. That is a **paper** result (it closes the largest
stated limit of `../adaptive_tol_theory_2026-08-25/generalization/FINDINGS.md`),
not a **performance** result, and it should be reported as such.

---

## Limits

- **The 40-mutant / 200-reference ablation was not run.** No CUDA and no Triton
  on this machine; the T4 that carried every prior measurement is stopped. So
  there is **no measured catch rate, no measured FP rate, and no measured
  end-to-end latency delta for `KCC_STRUCTURAL_L=1`.** Everything above is
  either re-derived from banked GPU arms or measured on CPU. The task asked for
  that ablation and this is the part that is missing.
- **Measurement 3 covers 11 of 27 operators** — the ones whose torch reference
  runs on CPU without companions. `matmul`, the three attention operators,
  `cross_entropy`, and the norm family are untested against adversarial inputs.
- **Measurement 2 is CPU.** The 1128× ratio is directionally safe (the bias
  favours the structural path) but the GPU number is not measured. M3's cost is
  dominated by RNG throughput, so a T4 would compress it — the 1395× draw-count
  ratio, which is exact arithmetic, is the more durable half of that finding.
- **M3 was validated at `n_samples = 40`; the checker's default is 20.** The
  structural path simulates a q95 order statistic one step outside what the
  derivation checked. `structural_l.py` says so at the call site.
- **`y_profile` uses `nsim = 3000`.** No sensitivity analysis was run on that;
  a lower NSIM would cut cost proportionally at some unmeasured accuracy price.
- **Flag-off identity was verified on 5 operators × 2 verdicts**
  (`flag_off_identity.py`), comparing against a transcription of the
  pre-change function body from the same seed. Exact match on verdict and
  tolerance in all 10. The extraction into `_probe_adaptive_tol_and_sens` is
  behaviour-preserving on that evidence, not by inspection alone.

---

> **SUPERSEDED IN BOTH DIRECTIONS, 2026-08-28 — see
> `../direct_tol_2026-08-28/FINDINGS.md`.**
>
> 1. The cost objection (1128×) is gone: the theory-closure round's B.3
>    DIRECT route computes the SAME estimand as a deterministic grid
>    integral of the exact parent CDF — 0.5–2.2 ms/call, validated at
>    0.09% vs the M3 simulation and R²(log) = 0.997 vs the banked
>    measured q95₂₀. A `KCC_STRUCTURAL_MODE=direct` arm ran the full
>    corpus **verdict- and attribution-identical** (40/40, 0/200, zero
>    failing-set diffs; E/draw ratio p50 = 1.008 on 605 records).
> 2. The latency premise is also gone, more fundamentally: this round's
>    "probing step = 2085 ms = 32.9% of check time" was measured under
>    KCC_CHECK_TIMING's serialisation. Un-serialised, the direct arm ran
>    +18% SLOWER than the probe arm, which bounds the probe's true
>    pipelined wall cost at ≲0.3 s per corpus pass (~30× below its
>    serialised share). The −24.5%-checker-wall ceiling in this document
>    never existed as wall time; the honest ceiling for any probe
>    replacement is ≤~0.7% of corpus runtime.
