# Masking-bug blast radius — no recorded verdict or reported number changes; and the checker caught the bug live three times in July, booked as "invalid input"

**Investigated 2026-08-27, CPU only** (all impact determinations are exact
emulations validated against banked GPU records; no re-measurement was
required — see §4 for why). Probes in `probes/`, logs in `data/`.
**The reference kernels were not touched.** This is the impact assessment the
attention_gram round required before any fix.

Bug under assessment (`../adaptive_tol_theory_2026-08-25/attention_gram/
ATTENTION_GRAM.md` §4): `flash_attention` and `scaled_dot_product_attention`
reference kernels include padded key columns (S = 0) in the softmax
denominator whenever `N % BLOCK_N(=32) ≠ 0`. `causal_flash_attention` is
immune (its causal mask excludes padded columns; verified bitwise on GPU in
that round).

---

## Verdict up front

| question | answer |
|---|---|
| Any recorded catch/miss/FP verdict changed by the bug? | **No — 0 of every recorded verdict examined**, across the corpus benchmarks, all n_samples/scope/timing arms, the adversarial search DBs, and the theory rounds. §2–§3 |
| Any reported FINDINGS/paper-facing number resting on a wrong verdict? | **No.** One class of banked *auxiliary numbers* is quantitatively off (the `last_tile_dropped` tolerances, ×0.82) but no classification or claim built on them moves. §3.3 |
| The most consequential find | **The checker detected this bug live, three times, on 2026-07-23** — the flash search's three N=130 proposals all recorded `Reference failed: attention_weights_sum_to_one`, which is exactly the padded-mass signature (emulated deviation 0.16–0.19 vs atol 1e-3). The framework booked a true reference-bug detection as "invalid input" and moved on. §3.4 |
| Closest near-miss to a changed outcome | Proposal idx 9 of the flash search (N=130) recorded a **gap-confirmed `skip_rescaling` catch 3 seconds before the run's winning hit**, denied hit status only by the reference failing its invariant. Counterfactual under a corrected reference, emulated exactly: the gap **disappears** (the gap existed only because mutant and buggy reference share the padded denominator; naive max_err 1.7e-4 → 2.8e-3 under correction), so it is still not a hit and **the recorded search outcome stands**. §3.4 |
| Affected-shape invocations found, total | **3 exposure sites**: (i) the flash spec's own battery at N=65 (`last_tile_dropped`, every full-checker flash run + the 5 attn_native records); (ii) `cross_shape` at spec shapes (65,64) flash / (1,64),(333,64) sdpa (every full-checker run); (iii) 3 adversarial proposals at N=130. Every one assessed individually. §2 |

---

## 1. Where the bug can bite at all — the three exposure paths

A verdict can only be affected where one of these ran flash/sdpa at
`N % 32 ≠ 0`:

1. **`cross_shape`** (L3): sweeps `spec.valid_shapes` — flash (65,64);
   sdpa (1,64) and (333,64). Runs in every full-checker invocation.
2. **The flash spec's own adversarial battery**: `last_tile_dropped` is
   *constructed* at N=65 (`multi_tile_rescaling` at N=192 is a multiple of 32
   — clean, checked).
3. **Adversarial-search proposals** with free shapes.

Everything else in the project runs attention at N ∈ {64, 128, 192, 256} —
multiples of 32 — including every tolerance/sandwich/M3 measurement in the
theory rounds (`gpu_native`, `gen_native`, `attn_native` primaries, the
attention_gram corpus stage). Verified per dataset below.

## 2. Full enumeration of banked datasets containing flash/sdpa

| dataset | attention data | affected-shape exposure | impact |
|---|---|---|---|
| `n_samples_curve_2026-08-25/arms/` (CURVE + 6 VALID arms) | 6 mutant + 25 ref trials each | cross_shape shapes; N=65 variant | **None.** Every mutant caught by L1/L2 at base shape or by ≥7 checks incl. cross_shape failing at *all five* shapes (max_err 2.4–3.4 at every shape — dropping (65,64) leaves 4 failing shapes). All 175 ref-trial cross_shapes pass (reference-vs-itself is counterfactual-invariant). 0 FPs. |
| `scope_detect_2026-08-26/arms/` (A–D) | 6 mutant entries + refs per arm | same | **None** — identical catch attribution (verified in A and B), 0 FPs. The fp-floor screen validation on `last_tile_dropped` survives correction: classification identical 20/20 seeds (§3.3). |
| `check_timing_2026-08-25/arms/` | 6 entries per arm | same | **None** — all caught, 0 FPs (verified in 2 arms; same harness). |
| `benchmarks/autokernel/files/results.json` + `kernel_executed_fix`/`delegation_fix` raw results | aggregate rates; baseline per-trial data | baselines compare at base shape (64,32) only | **None.** The 100%-catch/0%-FP rows decompose into the per-verdict structure above; baseline systems never leave (64,32). |
| `adaptive_tol_theory_2026-08-25` native runs (`gpu_native`, `gen_native`, coverage/replay) | 38 attention invocations | none — all (64,32), m=2048 throughout | **None.** All sandwich/L/M3 numbers unaffected. |
| `attn_native.jsonl` (55 records) | 11 variant groups | **`flash/last_tile_dropped` ×5 at N=65** | **Numbers shift, classifications don't** — §3.3. |
| adversarial search, main DB flash run `7b0a6e8d` | 12 proposals | **3 proposals at N=130** | **Recorded verdicts were TRUE detections of this bug; counterfactual outcome unchanged** — §3.4. |
| adversarial search, cfa runs (main `403262a6`, 2 reruns) | 274 proposals, 151 at N%32≠0 | none — causal is immune | **None.** (The `wrong_causal_mask` *mutant*'s off-by-one could admit the first padded column on the last query row — a mutant being differently wrong changes nothing.) |
| `attention_gram` round | 36 + 108 measurements | 100×32 (by design) | Already characterized in that round; its conclusions account for the bug. |
| phase1/phase2, matmul_prediction, KernelBench corpus, forkserver/batch_executor infra rounds | no flash/sdpa measurements (KernelBench L1 has no attention; infra rounds are CFA) | — | **None.** |

## 3. The three exposure sites, each resolved

### 3.1 `cross_shape` at (65,64) / (1,64) / (333,64) — no verdict hinges on them

Reference trials compare the reference against itself — identical outputs
whether the kernel is buggy or fixed, so those passes are exact under any
counterfactual. Mutant trials: banked per-shape sub-records show cross_shape
failing at **all five** shapes for the two mutants that reach it
(`drop_last_tile`, `skip_rescaling`), with the affected shape's margin
(3.2–3.3) indistinguishable from the clean shapes'; the other two mutants
short-circuit at L1 (`nan_inf`) before cross_shape runs. sdpa's mutant never
reaches cross_shape at all. **No configuration, including the ablation
analyses, ever depended on the (65,64)/(1,64)/(333,64) sub-verdicts alone.**

### 3.2 The latent-FP scenario never occurred

The dangerous case — a genuinely correct, independently-written candidate
rejected at an affected shape because the reference is wrong there — requires
a candidate that is not a descendant of the reference. **No recorded run ever
had one**: every candidate in every banked run is the reference itself or a
mutant edited from it (mutants inherit the same unmasked-S padding, so
mutant-vs-reference comparisons at affected shapes still isolate the mutant's
own bug). This is the exposure that makes fixing worthwhile *going forward*
(KernelBench-style third-party candidates), not one that contaminated the
record.

### 3.3 `last_tile_dropped` at N=65 — banked numbers ×0.82, classifications intact

Emulated on the variant's own construction (fp32, 20 seeds,
`probes/ltd_impact.py`): the recorded output-scale statistics are insensitive
(out max-rel diff ≤ 2.3e-5 — the 1e4-dominated rows set the ∞-norm scale),
but the perturbation q95 under the buggy kernel is **0.82× (median)** what
the corrected kernel would give — the padded mass is exp-suppressed only on
rows whose dominant score is positive (~half). Consequences:

- the banked `tol` values for this variant are ~20% low — **no claim uses
  them as numbers**;
- the fp32-floor *classification* (the variant's actual role, in the
  exception taxonomy and the scope round's screen validation) is **identical
  under both kernels**: min-s at ulp scale both ways, median-s/ulp screen
  fires 20/20 vs 20/20, agreement 1.00 (`data/ltd_screen.log`).

### 3.4 The three N=130 search proposals — the checker caught the bug in July

All three recorded `Reference failed: ['attention_weights_sum_to_one']`.
Emulated exactly (`probes/flash_n130_counterfactual.py`, validated against
the recorded verdicts): the V=ones row-sum comes out 0.16–0.19 below 1
(30 padded columns of 160; with K=zeros exactly 130/160), against the check's
atol of 1e-3 — **a true detection of the reference kernel's bug**, recorded
on 2026-07-23, a month before the bug was diagnosed. The framework's verdict
vocabulary has no "reference is buggy" outcome, so it was booked as invalid
input, scored −5, and deprioritised.

Counterfactual under a corrected reference, per proposal:

| idx | recorded | corrected-reference counterfactual |
|---|---|---|
| 1 (iter 0) | ref failed; 4 missed | ref valid; approx_denom + wrong_mask caught (10/10 seeds), **no naive gap** (0/10) → valid non-hit; **no earlier hit created** |
| 6 (iter 1) | ref failed; 4 missed | ref valid; 3 mutants caught, no gap → same |
| 9 (iter 2) | ref failed; **gap-confirmed skip_rescaling catch**, denied hit by ref-fail — logged 3 s before the run's winning hit | ref valid; skip_rescaling still caught, but the **gap evaporates**: naive max_err 1.73e-4 vs buggy ref (pass) → 2.83e-3 vs corrected ref (fail). The recorded gap was an artifact of mutant and buggy reference *sharing* the padded denominator. Still not a hit. |

**Net: the flash search's recorded hit (proposal 11, N=96, approx_denom) and
its proposals-to-hit count stand unchanged**; so do the search-efficiency
table, the hit-invariant 120/120 consistency claim (the three verdicts'
recorded values satisfy it, and their counterfactual values would too), and
the FP-correction narrative (36.2%→17.1%→0% is CFA-only; causal is immune).

## 4. Why no GPU re-run was needed

Every affected-shape determination reduces to functions whose GPU behaviour
was already bitwise-pinned: the attention_gram round proved the CPU
emulation `kernel_faithful` reproduces the shipped kernels' outputs to
≤ 6.3e-7 at affected shapes, and this round's emulators were additionally
validated against the recorded verdicts they re-derive (reference-fail
signature 3/3; the idx-9 naive gap reproduced at 1.7e-4 < 1e-3). The
remaining verdicts are counterfactual-invariant by structure
(reference-vs-itself, or catches at unaffected shapes / by ≥7 checks).

> **FIX SHIPPED 2026-08-27** — `../attention_mask_fix_2026-08-27/`. All §5
> regression criteria met (40/200 clean, zero attribution drift, weights-sum
> passes at every affected N, 100×32 lands on the Gram predictions, the three
> July proposals replay as valid non-hit). **One prediction below is
> corrected there:** the "×~1.2" `last_tile_dropped` tolerance shift does not
> materialise as a smooth ratio — both kernels' sensitivities on that variant
> are pinned to the fp32 quantization floor (buggy exactly 2 ulp of the 1e4
> output scale, fixed ~1 ulp), so the ratio is ulp-count noise (0.47–1.01
> across seeds). The classification-invariance claim, which is what the
> taxonomy and screen validation actually rest on, is confirmed 20/20.

## 5. What this means for the fix

- **Nothing recorded needs retraction.** The "every prior banked measurement
  is unaffected" claim from attention_gram §4 is confirmed and sharpened:
  unaffected in every *verdict* and every *reported number*; the only banked
  quantities that would change under correction are the `last_tile_dropped`
  tolerances (×~1.2) and the three July verdict labels (invalid-input → valid
  non-hit).
- The fix's regression criteria, when written: (i) all 29-op corpus verdicts
  identical; (ii) `attention_weights_sum_to_one` passes on the reference at
  N ∈ {1, 65, 100, 130, 333}; (iii) the attention_gram 100×32 measurements
  re-taken should land on the *math-Jacobian* Gram predictions (meas/pred
  ~0.89 → ~1.00); (iv) the three N=130 proposals replayed give valid-non-hit.
- **Process finding worth carrying forward:** `reference failed checker` is
  serving as both "input out of domain" and "reference kernel is wrong", and
  the only recorded instances of the second were silently absorbed by the
  first. A reference-side invariant failure on an in-contract shape deserves
  its own outcome label.

## 6. Reproduce

```bash
cd verification_runs/attention_mask_bug_impact_2026-08-27
PY=../../.venv/bin/python
$PY probes/ltd_impact.py                  # N=65 variant: outputs, q95 ratio, floor stats
$PY probes/flash_n130_counterfactual.py   # 3 search proposals, recorded + counterfactual
# data/*.log additionally bank the dataset sweeps (arms, DBs, results.json)
```

## 7. Limits

- Counterfactual "caught" for the N=130 proposals is asserted via the
  weights-sum check (emulatable exactly) — other checks could only add
  catches, and no conclusion depends on catch count; the **gap** (naive
  allclose) determinations are exact.
- The two `randn`-fill proposals have no recorded seed; their conclusions
  are 10/10 across seeds and the decisive proposal (idx 9) is fully
  deterministic.
- `attn_native`'s exact GPU draws are not CPU-reproducible (torch CUDA RNG);
  §3.3's ×0.82 is distributional over the variant's own construction, and
  the classification-invariance claim is what matters — it held on every
  seed.
- The kernel_executed_fix / delegation_fix rounds' full-checker data was
  assessed structurally (same harness, catch attribution verified in four
  sibling datasets) rather than record-by-record.
