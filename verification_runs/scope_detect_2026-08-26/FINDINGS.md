# Scope-divergence detector: the floor screen works, the saturation screen is falsified, and the corpus is what showed it

**Designed, implemented, and validated 2026-08-26 on a Colab T4** (session
`kccscope`, provisioned and stopped). Four arms in `arms/`, driver
`scopedet.sh`, scoring `analysis/validate_gpu_run.py`.
`torch 2.11.0+cu128`, `triton 3.6.0` — the same stack as every prior round.

**`KCC_SCOPE_DETECT` defaults OFF. All four arms: 40/40 catch, 0/200 FP.**

---

## Verdict up front

| question | answer |
|---|---|
| Verdicts unchanged with the detector on? | **Yes.** A and B byte-identical; 854/854 records promoted into `scope_flags`. |
| Flags `argmax`/`argmin`? | **Yes**, structurally, 12 records, no measurement. |
| Flags the attention adversarial cases? | **5 of 6 on every invocation; the 6th on 4 of 5.** |
| Silent on the other 24 operators' *ordinary* inputs? | **Yes**, with 9×–381× margin on `s/ulp`. |
| Is the `s/ulp` screen validated? | **Yes.** Large margins, zero classification change across 3/20/40-delta arms. |
| Is the linearisation-defect screen validated? | **No — falsified.** The classes **overlap** on the real corpus. |
| Ready to adopt? | **No.** One screen works; the other is load-bearing for a mechanism the working one cannot see. |

---

## 1. What held

**Verdict safety, confirmed on the corpus rather than by inspection.** Arms A
(detector off) and B (detector on) produce identical per-check outcomes across
all 240 candidates. The detector reuses the sensitivity loop's own deltas and
draws no new RNG, so the two arms consume the generator identically — the
property `scopedet.sh` was written to check, and it holds.

**The new field is genuinely exercised.** 854 scope records appear in
`scope_flags` and 854 in `subchecks` — the promotion in
`KernelChecker._run_check` neither drops nor duplicates. `checker_adapter.py`
was extended to serialise `scope_flags` explicitly so the scoring reads the
promoted field rather than the source list; reading `subchecks` would have
passed whether or not the promotion worked.

**Structural exclusion.** `argmax` and `argmin` flagged on every invocation,
with no probe run and no cost.

**The `s/ulp` screen.** Margins on the silent set, at the converged 40-delta
probe:

| | tightest | widest |
|---|---:|---:|
| `s/ulp` margin to the 32 threshold | **9×** (`cross_entropy`, 296) | **381×** (`gelu`, 12 198) |

No floor classification changed between the 3-, 20- and 40-delta arms. It
catches all three fp-floor variants (`last_tile_dropped`, `skip_rescaling`,
`equal_attention_weights`) on every one of their 22 invocations. **The
median-not-minimum choice paid off exactly as predicted:** `cross_entropy` is
the tightest operator at 296 and stays silent; on the minimum statistic it
would have fired on every run.

---

## 2. What failed — the defect screen, and it is not a tuning problem

### 2a. The statistic converges, and 3 deltas was too few

| deltas | worst in-scope defect | margin to 10% | records fired | disagreements vs 40 |
|---:|---:|---:|---:|---:|
| 3 (shipped default) | 8.613% | 1.16× | 182 | **18 / 854** |
| 20 | 9.605% | 1.04× | 169 | **0 / 854** |
| 40 | 9.605% | 1.04× | 169 | — |

**20 and 40 agree on every one of 854 records**, so the statistic is converged
by 20. Three is not: it flipped `layernorm/wrong_variance_trigger` (×11) and
`cross_entropy/large_magnitude_logits` (×3) into firing, and two attention
records the other way. **The default has been changed from 3 to 20.**

### 2b. At convergence, the classes overlap

| | value | invocation |
|---|---:|---|
| worst **in-scope** defect | **9.605%** | `cross_entropy/adversarial_large_magnitude_logits` |
| least **out-of-scope** defect | **6.6%** | `causal_flash_attention/adversarial_large_magnitude_qk` |

**The out-of-scope value is below the in-scope one. Separation 0.68×. No
threshold separates them.** Any constant either misses a real divergence or
fires on a healthy invocation; it cannot avoid both. At 40 deltas the detector
already misses `causal_flash_attention/large_magnitude_qk` on 1 of its 5
invocations — the "6 of 6" the 3-delta arm scored was partly the noise of an
unconverged statistic.

### 2c. Why the banked data promised a 6.5× gap

`GPU_NATIVE.md` §4 gave in-scope 3.66% against out-of-scope 23.7%. On the
corpus that becomes 9.605% against 6.6%. The two populations are different:

- the banked **in-scope** side was measured on the corpus's **ordinary inputs
  only**;
- the checker's real call mix is **75% adversarial**, and in-scope operators
  legitimately reach 8–9.6% defect on `near_zero_variance`,
  `wrong_variance_trigger` and `large_magnitude_logits` without leaving scope;
- the banked **out-of-scope** side was 5 seeded attention draws, which happened
  to sample the high end of `large_magnitude_qk`'s spread.

**I derived a threshold from one population and applied it to another.** That
is the error, it was mine, and the corpus run is what exposed it — no amount of
offline replay against the same banked table could have. The 47 offline tests
still pass; they were testing the rule against the population it was fitted to.

### 2d. Why dropping the screen is not the fix either

30 of 854 records fire on **saturation alone** — `s/ulp` does not see them:

```
causal_flash_attention/adversarial_large_magnitude_qk
flash_attention/adversarial_multi_tile_rescaling
scaled_dot_product_attention/adversarial_large_magnitude_qk
```

These are exactly the three softmax-saturation variants, and their `s/ulp` runs
210–642 588 — far above any floor threshold. **`s/ulp` alone covers mechanism
(ii) and is blind to mechanism (i).** So the detector cannot ship as
"floor screen only" and still claim to cover both mechanisms it was built for.

---

## 3. Full fire set (arm B, 3 deltas — the arm scored against the expectation)

| operator / check | reasons | defect% | s/ulp |
|---|---|---:|---:|
| `argmax`, `argmin` / `perturbation_tolerance` | structural | — | — |
| `flash_attention/last_tile_dropped` | floor + saturation | 1900.0 | 2.00 |
| `flash_attention/skip_rescaling` | floor + saturation | 900.0 | 2.00 |
| `flash_attention/equal_attention_weights` | floor + saturation | 900.0 | 5.00 |
| `flash_attention/multi_tile_rescaling` | saturation | 854.5 | 210.00 |
| `sdpa/large_magnitude_qk` | saturation | 26.6 | 3311.50 |
| `causal_flash_attention/large_magnitude_qk` | saturation | 16.0 | 642588.50 |
| `softmax/near_zero_variance` | floor + saturation | 900.0 | 1.00 |
| `softmax/max_in_last_tile`, `softmax/extreme_range` | floor | — | 0.00 |
| `log_softmax/near_zero_variance` | floor | — | 0.00 |
| `groupnorm/near_zero_variance`, `instancenorm/near_zero_variance` | floor | — | 0.00 |
| `swish/near_global_min` | floor + saturation | 150.0 | 7.00 |
| `gelu/near_global_min` | floor + saturation | 42.9 | 24.00 |
| `cross_entropy/large_magnitude_logits` | saturation | 100.0 | 426.00 |
| `layernorm/wrong_variance_trigger` | saturation | 11.0 | 8043.50 |

The **in-scope** attention variants `approx_denominator` and
`wrong_causal_mask` were **not** flagged — the case that falsified the
peak-attention-weight predictor is handled correctly.

**Ten of these are outside the expectation set**, on operators
`GPU_NATIVE.md` §4 never studied. They are mostly floor-driven on genuinely
degenerate inputs (`near_zero_variance` is the input × 1e-6; `near_global_min`
sits at `gelu`'s flat point where the derivative vanishes), so they are
*plausibly* true positives by the same two mechanisms. **But no ground-truth
in/out label exists for them, so they cannot be scored, and I am not claiming
them as correct detections.** The two that flipped between arms —
`cross_entropy/large_magnitude_logits` and `layernorm/wrong_variance_trigger` —
are the ones the defect overlap makes unanswerable.

---

## 4. Recommendation — revised downward: **not ready to adopt**

The design pass concluded "proposable, pending the GPU arm". The GPU arm ran
and the answer is **no, not as built**.

| component | status |
|---|---|
| annotate-only wiring, `scope_flags`, verdict safety | **validated — adopt-ready** |
| structural exclusion (`argmax`/`argmin`) | **validated — adopt-ready** |
| `s/ulp` floor screen | **validated**, 9–381× margins, arm-stable |
| linearisation-defect saturation screen | **falsified** — classes overlap 0.68× |
| `KCC_SCOPE_DEFECT_SAMPLES` default | **fixed**: 3 → 20 (converged) |

**The blocker is specific and not cosmetic.** Two of the three mechanisms are
detected reliably. The third — softmax saturation — has no working screen, and
it is the mechanism that motivated the detector in the first place. Shipping
what works would mean shipping a detector that is silent on precisely the case
`GPU_NATIVE.md` §4 was written about.

**The threshold has deliberately not been retuned.** Moving `DEFECT_MAX_PCT`
trades a miss for a false alarm and cannot do better than the overlap allows;
two new tests pin that so a later pass cannot present a nudge as a fix.

**What would unblock it** — a signal that separates saturation from ordinary
nonlinearity on adversarial inputs. The two obvious candidates, both cheap and
both measurable on the arms already banked in `arms/`:

1. **Homogeneity slope**, `log10(s(1)/s(0.1))`, which `GPU_NATIVE.md` §2 reports
   at 0.9922–1.0050 across all 228 in-scope invocations — a far tighter band
   than the defect's. It is already computed from the same two launches the
   defect uses, so it costs nothing extra to evaluate offline.
2. **Peak softmax weight taken at the saturating operator itself** rather than
   as a whole-kernel summary. The falsified predictor was peak *attention*
   weight as a proxy for the whole variant; the mechanism is specifically that
   softmax collapses to a select.

Neither is tested. Both can be evaluated against `arms/*.json.gz` without a GPU,
because the sensitivity vectors and both launch scales are banked there.

**Cost, now that the sample count is settled:** 20 deltas × 844 calls =
16 880 extra reference launches, **+24.3% checker wall = +3.9% corpus runtime**
at the banked 0.1218 ms/sample/call. That is 6× the +0.6% the design pass
quoted off a 3-delta default, and it changes the adoption calculus even if the
saturation screen is repaired — 3.9% is the same size as the entire structural-L
ceiling this thread already declined to spend.

---

## 5. Limits

- **The 10 out-of-expectation fires are unscored.** No ground-truth label exists
  for non-attention adversarial variants; `GPU_NATIVE.md` §4 studied attention
  only. Whether they are true positives or false alarms is genuinely open, and
  the honest reading of "stays silent on the other 24 operators" is that the
  detector *does not* — it fires on 10 of their adversarial variants, for
  reasons that look mechanistically right but are not confirmed.
- **One corpus, one seed, one T4.** Arms differ only by detector settings, and
  the catch/FP rates are flat at 40/40 and 0/200 across all four, so the corpus
  contributes no signal on whether flagging helps or hurts detection — the same
  saturation `../n_samples_curve_2026-08-25/` ran into.
- **`SUB_SCALE = 0.1` is still inherited and untested.** The defect's behaviour
  is a function of it, and the overlap in §2b might be sensitive to it. Not swept.
- **The `tolerance_floor_bound` advisory was not analysed** in this run beyond
  being emitted; it fires often, as predicted.
- **Timing is instrumented** (`KCC_CHECK_TIMING=1`, CUDA serialised), so the
  +3.9% figure is derived from the banked per-sample slope, not from these arms'
  wall times, which are upper bounds and not comparable to published latency.

---

> **SUPERSEDED ON THE SATURATION SCREEN, 2026-08-27 — see
> `../gram_screen_2026-08-27/FINDINGS.md`.** The defect ladder was replaced
> (not retuned) by the paired Gram-ratio screen (exact float64 directional
> derivatives along the loop's own deltas); the corpus run separates what
> this round's §2b showed the ladder could not (in-scope band 0.04–2.3% vs
> multi_tile at 2.26×, pre-registered factor-2 threshold untouched). Two of
> the three "saturation" ground-truth labels used above are REVISED by that
> round's §3: the `large_magnitude_qk` classes measure Jacobian-consistent
> to 1.02–1.34× — the 6.6–27.7% defects this round recorded for them were
> the ladder seeing mild exponential onset through its own noise band, and
> `cross_entropy/large_magnitude_logits`'s fatal 9.6% measures 1.004. The
> §4 candidate signals (homogeneity slope, per-operator peak weight) were
> made moot rather than evaluated. Everything else here — verdict safety,
> structural exclusion, the floor screen, the 3→20 delta default, the
> cost discipline — stands and was reproduced.

> **EFFECTIVE-SAMPLE ACCOUNTING, 2026-08-28 — see
> `../theory_closure_2026-08-28/FINDINGS.md` §3.** Under `KCC_ABLATION_SEED`
> per-check reseeding, 23 of this run's (op, check) classes are bit-identical
> replicas of one (input, deltas) draw (all flash_attention and matmul
> adversarial variants, all softmax variants, rmsnorm/non_power_of_two,
> max/min nonpow2, and the near_zero_variance transforms at fp32 resolution).
> The fire-set counts for those classes ("22/22", "10/10") are one draw
> replayed; the arms' 842 fingerprintable records contain ~501–505
> bit-distinct measurements. The falsification in §2b is UNAFFECTED: both of
> its load-bearing numbers come from varying classes
> (`cross_entropy/large_magnitude_logits` and
> `causal_flash_attention/large_magnitude_qk` are captured-input transforms
> with 5–6 distinct draws each). The "6 of 6 / 5 of 6" saturation-catch
> scores counted the three collapsed attention classes as if their
> invocations were independent; their per-class evidence is one draw each.
