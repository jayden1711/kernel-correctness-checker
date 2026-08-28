# The Gram screen: the rebuilt saturation screen separates what the defect ladder could not — and the exact-derivative evidence revises two of the three "saturation" labels it was built to catch

**Designed, implemented, and validated 2026-08-27 on a Colab T4** (session
`kccgram`, provisioned and stopped). Two arms in `arms/`, driver `gramdet.sh`,
probe `probe_redundancy.py` (byte-copy of the n_samples round's), scoring
`analysis/validate_gram_run.py`, cost `analysis/cost_compare.py`.
`torch 2.11.0+cu128`, `triton 3.6.0` — the same stack and the same corpus,
seeds, and RNG discipline (`KCC_ABLATION_SEED=1`) as the round that falsified
the defect screen (`../scope_detect_2026-08-26/`), so records pair
one-to-one with that round's banked arms.

**`KCC_SCOPE_DETECT` still defaults OFF. Both arms: 40/40 catch, 0/200 FP.**

## What was built

`scope_detect.py`'s saturation screen was REPLACED (not retuned): the
two-scale linearisation-defect ladder is gone, and in its place the screen
computes, per invocation, the **paired Gram ratio**

    r_k = s_meas_k / ||J(x) d_k||_inf,   k = 1..20

with `J d_k` the exact float64 directional derivative (`torch.func.jvp`) of
the operator's math definition (`math_refs.py`, 27 operators, unit-tested
against torch built-ins) along the SAME deltas the tolerance was built from.
Flag when `|median_k log10 r_k| >= log10 2` — a **pre-registered factor-2
band, not a fitted constant**. Floor-gated: the screen only runs where
median s/ulp >= 32 (below, the validated floor screen owns the record).
No new RNG, no extra kernel launches; verdict-safe by construction.
38 offline tests in `tests/verification/layer2/test_scope_detect.py`.

## Verdict up front

| question | answer |
|---|---|
| Verdicts unchanged with the detector on? | **Yes.** A and G identical on every one of 4 333 check outcomes; 854/854 records promoted into `scope_flags`. |
| Structural exclusion (argmax/argmin) and floor screen? | **Unchanged and reproduced**: 22/22 on each of the three fp-floor attention variants, same silent set. |
| Does the Gram screen separate the classes the defect ladder could not? | **Yes.** Worst genuinely-in-scope class sits at ratio 1.44x; the catastrophic saturation class sits at 2.26x median with per-delta ratios spanning SIX DECADES; the pre-registered threshold (2x) lands inside the gap untouched. The ladder's corpus run had **overlap 0.68x**; this run separates. |
| Does it flag all three 2026-08-26 "saturation" classes? | **One of three — and that is a finding, not only a miss.** `multi_tile_rescaling` fires 22/22. The two `large_magnitude_qk` classes measure at ratio 1.02–1.34: the exact-derivative evidence says those invocations are *mostly Jacobian-generated* (mild saturation onset), and the "out-of-scope" labels — which were themselves produced by the now-retired ladder — do not survive contact with a sharper instrument. §3. |
| False fires on in-scope operators, incl. the ones the ladder wrongly implicated? | **Zero.** All 29 primaries silent at |log10 r| <= 0.0016 (183x margin). `cross_entropy/large_magnitude_logits` — the invocation whose 9.6% ladder defect killed the old screen — measures **ratio 1.004**: the ladder's 9.6% was never operator curvature. §2. |
| Cost | +13 ms median per perturbation-routed check (3.3 → 16.3 ms), +17 s per 240-candidate corpus pass under serialised timing. All CPU float64 autodiff; no GPU work added. §5. |
| Ready to adopt? | **Yes, with the stated semantics** (§6): annotate-only, structural + floor + Gram, the Gram flag meaning "response not explained by the operator's Jacobian at this input" — which covers saturation AND reference-kernel bugs, deliberately. |

## 1. What held, mechanically

Arms A (detector off) and G (detector on) produce identical outcomes on all
868 perturbation-routed and 3 465 other check records; the screen draws no
RNG and launches nothing, so this is by construction and the arm confirms
the construction. Promotion 854/854, no drops, no duplicates. The floor
screen reproduces its 2026-08-26 behaviour exactly (same fires, same silent
set) — it was not touched.

## 2. The in-scope side: the Jacobian explains the corpus to 0.04–0.4%

Worst |log10 median ratio| per class, silent set (full table in
`analysis/validate_gram_run.py` output, banked in `arms/`):

| bucket | worst class | worst ratio |
|---|---|---:|
| all 29 primaries | cross_entropy | 1.004 |
| attention in-scope adversarial (approx_denominator, wrong_causal_mask) | — | 1.0002 |
| adversarial, smooth (matmul all 6, softmax equal_logits/non_pow2, rmsnorm, layernorm large_variance/wrong_variance_trigger/non_pow2, reductions, pools, batchnorm, l2norm, gelu/swish large_magnitude, frobenius, cross_entropy large_magnitude_logits) | layernorm/skip_mean_subtract | 1.055 |
| adversarial at a non-C^1 point | l1norm/second_half_dominant | **1.44** |

Three of these numbers do real work:

- **`cross_entropy/adversarial_large_magnitude_logits` at 1.004.** The old
  screen died on this invocation: 9.605% ladder defect, above its
  out-of-scope example. The exact-derivative measurement shows the response
  matches the Jacobian to 0.4% — the ladder's 9.6% was an artifact of
  comparing two fp32 kernel launches against each other at different
  scales, not curvature of the operator. The corpus-overlap that falsified
  the defect screen was substantially the ladder measuring itself.
- **`layernorm/adversarial_skip_mean_subtract` at 1.055** — inputs shifted
  100–1000, the documented fp32-cancellation regime for the kernel: the
  5.5% is real kernel-vs-float64-math divergence at hostile inputs, visible
  but 5.5x below the flag line. The screen sees the effect the norm
  adjudication round documented, at the size that round predicted.
- **`l1norm/adversarial_second_half_dominant` at 1.44** is the honest edge
  of the in-scope set: half the input is exactly zero, `|x|` has a kink at
  the evaluation point, and the C^1 assumption fails there *structurally*.
  The 1.44x is genuine (the measured response includes |delta|-terms the
  Jacobian cannot see). It is also the **binding margin of the whole
  design**: 1.39x below the threshold where every smooth case is >= 100x
  below. A future corpus with a harsher kink case could cross 2x, and the
  right response would be a structural non-C^1 exclusion (the operator is
  known to be non-differentiable at 0), not a threshold move.

## 3. The out-of-scope side, and the label revision the sharper instrument forces

The 2026-08-26 expectation set called three classes out-of-scope
"saturation", ground truth inherited from GPU_NATIVE §4 — which derived
those labels from the same defect ladder this round retired. The Gram
measurement splits them:

**`flash_attention/adversarial_multi_tile_rescaling` — catastrophic, flagged
22/22.** Median ratio 2.26x; per-delta log10 ratios span **−1.44 to +4.84**
(the measured response ranges from 25x below to 60 000x above what the
Jacobian generates, direction by direction). This is what "the tolerance is
measuring something other than ||J d||" looks like, and no in-scope record
is anywhere near it.

**The two `large_magnitude_qk` classes — NOT flagged, ratios 1.02–1.34, and
the evidence says the labels were wrong, not the screen.** Per-invocation
medians: causal 0.88–1.17 (4 invocations; the 5th sits at s/ulp = 6 and is
**floor-flagged instead** — the correct mechanism for it), sdpa 0.96–1.34.
Paired record-by-record with the 2026-08-26 arm (same seeds, same inputs —
the attention references' mask fix is bitwise-inert at N = 64/128):

    ladder defect 6.6–27.7%  <->  gram ratio 1.02–1.34x, same records

Both instruments see the same thing: **mild saturation onset** — a locally
exponential response (softmax logit-gaps of order 0.1–0.6 at the corpus's
x20 inputs), a few percent to ~30% away from linear. A single-parameter
exponential-response model maps each gram ratio to a ladder defect of the
right order (e.g. ratio 1.34 -> 23% vs 27.7% measured; 1.17 -> 13% vs 17.5%),
though per-record scatter is large — the two statistics take their medians
over different deltas. What the ladder could not do is *place* 6–28% —
its own in-scope band ran 0.1–9.6%, hence the fatal overlap. The Gram
band for smooth in-scope cases is 0.04–2.3%, so the same invocations are
now visibly distinct from in-scope AND visibly distinct from catastrophic.
At a tolerance distortion of <= 1.34x — inside the checker's own 3x scale
factor — these invocations do not warrant a divergence flag, and the
pre-registered threshold correctly leaves them silent. Their ratios travel
in the record (`gram_log10_median`) for anyone who wants the onset signal.

**The 2026-08-26 FINDINGS' "5 of 6 / 4 of 5" saturation-catch scores should
be read under this revision**: the detector was being scored against labels
its own falsified instrument produced. This round's scoring script
(`validate_gram_run.py`) keeps the old expectation set verbatim — its §2
prints the misses — and this document, not the script, carries the
adjudication. The floor trio's labels survive (mechanism confirmed); the
`multi_tile` label survives (spectacularly); the `large_magnitude_qk`
labels are revised from "out of scope, undetected" to "saturation onset,
quantified at 1.02–1.34x, below any defensible flag line".

## 4. The formerly-unscored ten, now adjudicated

The old round's 10 out-of-expectation fire classes, which it honestly
declined to score for lack of ground truth, resolve cleanly under the
exact-derivative measurement:

- **7 are pure floor cases** (softmax near_zero_variance / max_in_last_tile
  / extreme_range, log_softmax near_zero_variance, groupnorm and
  instancenorm near_zero_variance, gelu and swish near_global_min): floor
  fires 6-10/6-10 as before, and the Gram screen correctly has no signal
  there (the measured side is quantisation).
- **3 are in scope and now measured as such**: cross_entropy
  large_magnitude_logits (1.004), layernorm wrong_variance_trigger (1.008
  — the ladder had flipped this one at 3 deltas), rmsnorm constant_rows
  (1.01). The old screen's fires on these were false alarms, as suspected
  but unprovable then.

## 5. Cost, measured

Serialised-timing medians per perturbation-routed check: 3.28 ms (A) →
16.32 ms (G): **+13 ms**, all of it CPU float64 JVP (20 per invocation) and
device→CPU copies; other checks unchanged (0.43 → 0.46 ms). Whole-corpus
check wall +16.9 s (6.1 → 23.0 s serialised — an upper bound, per the
standing timing caveat). Unlike the retired screen this adds **zero
reference launches and zero GPU contention**; the cost is host-side and
overlappable in principle. The convergence table (prefix property, no extra
arms): k = 5 and 15 agree with k = 20 on all 684 gram-evaluated records;
k = 3, 8, 10 each disagree on exactly the 22 multi_tile records — the
per-delta ratios there straddle the threshold with a wild spread, so
mid-size prefixes can dip under it. **20 stays the default**; anyone
lowering `KCC_SCOPE_GRAM_SAMPLES` for cost is choosing to gamble precisely
on the class the screen exists to catch, and the banked ratios show why.

## 6. Integration — same architecture, adopted semantics

The 2026-08-26 architectural decision is reused unchanged and was not
re-litigated: **annotate-only, additive, cannot move a verdict by
construction** (the record is attached after `passed` is computed; arm A/G
identity re-verified here). What changes is only the third screen's signal:

| component | status after this round |
|---|---|
| structural exclusion (argmax/argmin) | validated (unchanged) |
| s/ulp floor screen, median >= 32 | validated (unchanged, reproduced) |
| Gram screen, median log10 r vs log10 2 | **validated on this corpus** |
| defect ladder | **retired**; removal pinned by tests |

Flag semantics, stated on the record and in the module docstring: a Gram
fire means *"the measured sensitivity is not explained by the operator's
Jacobian at this input"* — saturation and reference-kernel-vs-spec
divergence both produce it, are not distinguishable from inside one record,
and both genuinely invalidate the tolerance's guarantee. (The method found
two real reference bugs this way before this round; on this corpus, with
those bugs fixed or inert at the exercised shapes, no such fire occurred.)

## 7. Limits

- **One corpus, one seed regime, one T4**, as ever. The catch/FP rates are
  flat across arms, so the corpus still contributes no signal on whether
  flagging helps detection — the flag remains a labelling device.
- **`multi_tile_rescaling`'s 22 records are one distinct input** (its
  generator draws under the per-check reseed with hardcoded N=192), so the
  fired class has effective n = 1 input, replayed 22x. The old round had
  the same property. The per-delta spread (6 decades within one input) is
  the evidence that this is not fragile, but a second distinct draw has
  never been measured.
- **The in-scope/out-of-scope gap is 1.44x → 2.26x** (1.57x wide), with the
  pre-registered threshold sitting 1.39x above the kink case and 1.13x
  below the fired class. Far healthier than 0.68x (overlap), but not the
  11x the floor screen enjoys; the l1norm kink note in §2 says what to do
  if a future case crosses.
- **`math_refs.py` is a parallel implementation of 27 operator specs.** Its
  transcription is unit-tested against torch built-ins and corroborated by
  854 corpus records at ratio ~1, but an eps-placement error in BOTH the
  registry and the test would be invisible. The registry's docstring
  carries the known layernorm pad-lane caveat explicitly.
- **The exponential-onset model in §3 is a consistency check, not a law**:
  it reproduces the order of magnitude and the ranking of the paired
  ladder defects, with per-record scatter it does not explain.
  *(Superseded 2026-08-28: the actual law is derived and validated in
  `../attn_onset_2026-08-28/FINDINGS.md` — the per-delta ratio is the
  parameter-free pushforward of N(0, (√2·10⁻³κ²)²) through
  φ(u) = (1−e^{−u})/u; the per-record scatter is the law's own random
  variable, and the §3 label revision is now derived rather than
  measured.)*
- The +13 ms/check cost is measured under serialised CUDA timing on 2 vCPUs;
  production overlap could be better or worse.

## 8. Reproduce

```bash
# arms (T4):
export HOME=~/.colab-home
colab new --gpu T4 -s kccgram
tar --exclude='__pycache__' --exclude='.venv' -czf /tmp/kcc6.tgz \
    verification benchmarks scripts tests TritonBench
colab upload -s kccgram /tmp/kcc6.tgz /content/kcc6.tgz
colab upload -s kccgram verification_runs/gram_screen_2026-08-27/probe_redundancy.py /content/probe_redundancy.py
colab upload -s kccgram verification_runs/gram_screen_2026-08-27/gramdet.sh /content/gramdet.sh
# launch under nohup, poll for /content/gramdet/DONE, download /content/probe/*.json
colab stop -s kccgram

# scoring (local, no GPU):
python3 analysis/validate_gram_run.py arms/A_no_detector.json arms/G_gram.json
python3 analysis/cost_compare.py     arms/A_no_detector.json arms/G_gram.json

# offline tests:
.venv/bin/python -m pytest tests/verification/layer2/test_scope_detect.py -q
```

---

> **CORRECTIONS AND EFFECTIVE-SAMPLE ACCOUNTING, 2026-08-28 — see
> `../theory_closure_2026-08-28/FINDINGS.md` §3.**
>
> 1. **Effective independent n.** Under `KCC_ABLATION_SEED` per-check
> reseeding, 23 of this run's 83 (op, check) classes are bit-identical
> replicas of ONE (input, deltas) draw: all six flash_attention adversarial
> variants (fresh `_make_qkv` under a fixed per-check seed), all six matmul
> variants, all five softmax variants, rmsnorm/non_power_of_two,
> max/min all_negative/positive_nonpow2, and the three
> `full_like(x,3)+x*1e-6` near_zero_variance variants (distinct in
> construction but identical at fp32 measurement resolution). Class counts
> written "22/22", "10/10", "15/15" for those classes are one draw replayed;
> the run's 842 fingerprintable records contain **499 bit-distinct**
> measurements (adversarial: 632 → 289). This corrects evidence *weight*,
> not outcomes: no verdict, fire, or margin changes, and the §7 limit
> already flagged the multi_tile instance of this. Classes NOT collapsed
> include every primary (base inputs advance through the harness numpy rng)
> and every captured-transform variant (`large_magnitude_qk`,
> `large_magnitude_logits`, `second_half_dominant`, all layernorm variants —
> which vary through their captured gamma/beta companions even where x is a
> fresh draw).
>
> 2. **§2's table misfiled layernorm/adversarial_non_power_of_two as a
> measured smooth class.** In fact the Gram screen never evaluated
> layernorm or rmsnorm `non_power_of_two`: `gram_n_valid = 0`,
> `n_skipped = 20` on every record — `math_refs`'s companion slicing
> (`gamma[:333]` of a length-128 gamma) raised on every delta and the
> screen declined fail-open, silently. Their silence in this run is
> *absence of measurement*, not measured consistency. The floor screen's
> s/ulp values for those records are real (it needs no math function).
>
> 3. **Reference-suspect flag (flagged, NOT fixed, needs its own round):
> the width-333 `non_power_of_two` variants of layernorm and rmsnorm
> perform out-of-bounds companion reads in this corpus.** The autokernel
> corpus feeds (64,128) inputs with length-128 gamma/beta; the variant
> replaces x with a width-333 tensor while companions ride along, and the
> reference kernels load gamma/beta with `mask = col_offsets < n_cols`
> (n_cols = 333) over a 128-float allocation — columns 128–332 read out of
> bounds. Banked shapes confirm (input_stats `[64, 333]`); the bit-identical
> rmsnorm sensitivities across runs with *varying* in-bounds gamma are
> consistent with the response being dominated by stable OOB lanes. At the
> spec corpus's 512-wide shapes the same variants are in-bounds and
> unaffected. Adjudication and fix deferred per house rules.

> **The §3-correction's OOB flag was ADJUDICATED 2026-08-28 —
> `../oob_adjudication_2026-08-28/FINDINGS.md`**: spec-construction artifact
> (invalid companion lengths from the hardcoded width-333 variant), not a
> kernel bug; OOB reads byte-level-proven; no recorded number anywhere
> depends on the affected records; fix specified and deferred. The
> fail-open behaviour of `measure_gram` on these records (silent
> `n_valid = 0`) is noted there as worth a surfaced counter in any adoption
> pass.

> **FIXED DOWNSTREAM, 2026-08-28 — `../oob_fix_2026-08-28/FINDINGS.md`**: the
> two previously-unevaluated classes (layernorm/rmsnorm non_power_of_two)
> are now measured by the Gram screen post-fix — `gram_n_valid = 20` on all
> 31 records, silent at worst ratio 1.00010 — and the corpus regression
> against this round's arms is verdict- and attribution-identical.
