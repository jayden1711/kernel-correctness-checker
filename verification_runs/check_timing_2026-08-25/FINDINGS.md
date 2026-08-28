# The `weight_magnitude` overlap comment was a false lead — but the check is still fully redundant, for a different reason

**Measured 2026-08-25 on a Colab T4** (shipped-warm Triton cache). Harness
`probe_redundancy.py`, driver `redun.sh`, five ablation arms in `arms/`.
Instrumentation added to `verification/checker.py`,
`verification/layer2_numeric_oracle/shape_generalization.py`,
`benchmarks/autokernel/files/checker_adapter.py` — all opt-in, all default OFF.

---

## Verdict

Two separate answers, and they point opposite ways:

1. **The source comment's stated mechanism is wrong.** `check_weight_magnitude`'s
   `large_uniform`/`large_random` do **not** exercise the same numeric regime as
   the per-spec adversarial battery's `large_magnitude` inputs. Different
   magnitudes (10–600x apart on 10 of 11 operators), different shapes, different
   comparators.
2. **The check is nevertheless removable at zero catch-rate cost.** It fires on
   15 of 40 mutants and is **never the sole catcher on any of them**. Removing it
   outright: 40/40 catch, 0/200 FP, −11.8% checker time.

**The ceiling is ~1.9% of corpus runtime, which is not worth spending a change
on.** Detail below, including what *is* worth measuring instead.

---

## Instrumentation added (opt-in, default OFF)

| switch | effect |
|---|---|
| `KCC_CHECK_TIMING=1` | per-check `duration_ms` on `CheckResult`; per-variant `duration_ms` + `input_stats` inside `check_weight_magnitude`; `input_stats` + comparator on each `adversarial_*` |
| `KCC_DISABLE_CHECKS=a,b` | skip named checks (ablation only) |
| `KCC_DISABLE_VARIANTS=a,b` | skip named `check_weight_magnitude` variants (ablation only) |
| `KCC_ABLATION_SEED=1` | reseed torch per check name, from `zlib.crc32` |

Three notes on why these are shaped this way:

- **Timing is opt-in because honest CUDA timing requires `torch.cuda.synchronize()`
  around every check, which serialises the pipeline.** Instrumented totals are
  therefore **not** comparable to uninstrumented ones — per-check *shares* are
  meaningful, absolute per-check times are an upper bound. The flag exists so
  the benchmark's published latency numbers can never be produced under it.
- **`KCC_ABLATION_SEED` is required, not hygiene.** Checks consume RNG
  (`check_weight_magnitude` calls `spec.make_inputs` and `torch.randn`;
  `check_perturbation_tolerance` draws 20 `randn_like` per call). Skipping a
  check shifts the RNG stream for every later check, so a verdict could flip for
  a reason unrelated to the removal — the same defect class as the
  unseeded-executor finding (SESSION_HANDOFF §7). Every arm below carries it,
  so arms differ **only** by the removed check.
- **`zlib.crc32`, not `hash()`.** Python randomises string hashing per process
  unless `PYTHONHASHSEED` is pinned, which would silently give each arm
  different per-check seeds.

`CheckResult` gained `duration_ms` and `subchecks`, both defaulting to `None`.
**`subchecks` closes a real gap:** `check_weight_magnitude` has always returned
per-variant outcomes as a third return element, and `_run_check` discarded it —
which is why per-variant attribution had never existed.

---

## Finding 1 — the numeric regimes are not the same

`input_stats` fingerprints the tensor each check actually feeds the kernel, so
this is answered from tensors rather than from variant names.

| operator | wm `large_uniform` | wm `large_random` | spec large-magnitude |
|---|---:|---:|---:|
| `causal_flash_attention` | 1e4 @128×64 | 4.11e4 @128×64 | **67.9** @64×32 |
| `scaled_dot_product_attention` | 1e4 @128×64 | 4.11e4 @128×64 | **70.4** @64×32 |
| `cross_entropy` | 1e4 @64×100 | 3.94e4 @64×100 | **154** @64×32 |
| `gelu` | 1e4 @4096 | 3.76e4 @4096 | **382** @64×128 |
| `swish` | 1e4 @4096 | 3.76e4 @4096 | **430** @64×128 |
| `log_softmax` | 1e4 @512×512 | 4.44e4 @512×512 | **821** @64×128 |
| `sum/mean/max/min_reduction` | 1e4 @512×512 | 4.44e4 @512×512 | **~2e3** @64×128 |
| `rmsnorm` | 1e4 @512×512 | 4.44e4 @512×512 | **4.03e4** @64×128 |

Three independent axes of difference:

- **Magnitude** — 10–600x apart on 10 of the 11 operators that have both. Only
  `rmsnorm` is genuinely comparable (4.44e4 vs 4.03e4).
- **Shape** — `check_weight_magnitude` runs at `spec.valid_shapes[0]`; the
  adversarial battery runs at the harness's input shape.
- **Comparator** — fixed `torch.allclose(atol=1e-3, rtol=1e-3)`, one call per
  variant, versus `check_perturbation_tolerance`'s **adaptive** band (20
  perturbation samples, q95 of the reference's own sensitivity).

Only 11 of 29 operators have a spec large-magnitude check at all;
`check_weight_magnitude` runs on all 29.

**So the comment was reasoning from names, and the names mislead.** As a
statement about overlapping numeric coverage, it is not supported.

---

## Finding 2 — the check is redundant anyway, by over-determination

| arm | catch | FP | checker time | vs A |
|---|---:|---:|---:|---:|
| **A** baseline | 40/40 | 0/200 | 8096 ms | — |
| **B** no `large_uniform`+`large_random` | 40/40 | 0/200 | 7256 ms | **−10.4%** |
| **C** no `adversarial_large_magnitude*` | 40/40 | 0/200 | 7768 ms | −4.0% |
| **D** both B and C | 40/40 | 0/200 | 7391 ms | −8.7% |
| **E** no `weight_magnitude` at all | 40/40 | 0/200 | 7140 ms | **−11.8%** |

**Noise floor, from the arms themselves:** D removes strictly more work than B
yet measured **1.9% slower**. There is one run per arm and no replicate, so
differences below ~2–3% are unresolved. E's −11.8% clears that; **C's −4.0%
barely does and should be treated as directional.**

Per-mutant redundancy structure in arm A (weight_magnitude counted per variant):

```
distinct failing checks per caught mutant:
  1 check : 16 mutants     4 checks:  3        9 checks: 1
  2 checks:  8             5 checks:  1       10 checks: 1
  3 checks:  6             6 checks:  2       11 checks: 1
                           8 checks:  1
```

**`weight_magnitude` fires on 15 of 40 mutants. All 15 are also caught by at
least one other check. It is the sole catcher on zero.**

The load-bearing checks are elsewhere — the 16 mutants with exactly one catcher
depend on:

| unique catcher | mutants |
|---|---:|
| `nan_inf` | 3 |
| `permutation_invariance` | 2 |
| `adversarial_all_negative_padded` | 1 |
| `adversarial_all_negative_nonpow2` | 1 |
| `adversarial_all_positive_nonpow2` | 1 |
| `attention_weights_sum_to_one`, `unit_frobenius_norm`, `unit_l1_norm`, `unit_l2_norm`, `affine_correctness`, `distributivity`, `gamma_correctness`, `tile_coverage_softmax_positivity` | 1 each |

Mostly Layer-2 algebraic properties — and, notably, the three
`adversarial_all_*_padded/nonpow2` checks that were also the ones AutoKernel's
gate structurally cannot reach.

---

## Finding 3 — the ceiling, and why partial removal is worse than it looks

Per-check share of measured check time (arm A, 240 trials, 6.04 s total):

| check | share |
|---|---:|
| `weight_magnitude` | **11.6%** |
| `perturbation_tolerance` | 10.8% |
| `precision_coercion` | 10.5% |
| `cross_shape` | 7.2% |
| `kernel_executed` | 6.6% |
| **all `adversarial_*` combined** | **35.5%** |

Inside `check_weight_magnitude`:

| component | share of the check |
|---|---:|
| **setup** (`spec.make_inputs` + `_make_weight_variants`) | **46.8%** |
| `large_uniform` | 15.7% |
| `large_random` | 12.8% |
| `monotone_rows` | 12.4% |
| `alternating_sign` | 12.2% |

**Nearly half the check is setup that runs regardless of how many variants
survive.** Dropping the two suspected variants therefore caps at ~29% of the
check — the thing the original comment pointed at is the *smallest* available
win.

Translated to corpus runtime (60.8 s warm run; `weight_magnitude` runs in
`your_checker (full)` 5.28 s and `(numeric only)` 4.61 s = 9.89 s):

| removal | corpus saving |
|---|---:|
| whole `weight_magnitude` | **~1.15 s = 1.9%** |
| `large_uniform`+`large_random` only | ~0.33 s = **0.5%** |

Cross-checked two ways — from the per-check share (11.6% of 9.89 s) and from
arm E's wall time (−11.8% of 9.89 s). Both give ~1.9%.

---

## Correctness risk, and three caveats that bound the result

Zero measured catch-rate cost, and zero FP change. But "no unique catch" is
weaker than "safe to delete", for three reasons that must travel with it:

1. **Short-circuiting.** `KernelChecker.run` aborts between layers, so a check
   that never runs cannot appear as a unique catcher. `weight_magnitude` is
   Layer 3; every mutant caught in Layer 1 or 2 never reaches it. Its
   "never-unique" status is a property of the **current pipeline order**, not of
   the check in isolation.
2. **40 mutants is a small, fixed corpus.** The result is "no mutant *here*
   depends on it". The adversarial search generates novel inputs against a
   distribution this corpus does not sample, and none of this measures that.
3. **The arms are single runs.** See the noise floor above.

---

## Proposal — do not remove anything for latency

**Recommendation: leave `check_weight_magnitude` in place.** A 1.9% corpus win
does not justify touching a Layer-3 check whose redundancy is contingent on
pipeline order, in a subsystem that has returned an unplanned finding every time
it has been modified. For scale: the Triton-cache work in
`verification_runs/triton_cache_2026-08-25/` delivered **75%** for no semantic
change at all. This is not the next lever.

Two things that *are* worth doing, neither of which is a removal:

- **`check_perturbation_tolerance`'s `n_samples=20` has never been justified.**
  It is the hot path for 46% of all check time (`perturbation_tolerance` 10.8% +
  every `adversarial_*` at 35.5%, all of which route through it). The actionable
  question is the **catch-rate sensitivity curve versus `n_samples`** — if 10
  samples hold 40/40, that is a ~23% checker-time saving against the same
  ablation discipline used here, an order of magnitude above anything available
  from removing a check.
- **46.8% of `weight_magnitude` is input construction**, repeated per candidate
  at a fixed shape. That is a caching question, not a removal question, and it
  generalises to any other check that rebuilds inputs per call.

Neither is implemented. Both need their own before/after.
