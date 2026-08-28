# Layer 2 / Layer 3 per-check ablation — findings and runbook

Item #2: *"tests scattered — analyze ablation for each check and justify
layer 2."* Layer 2 is reported as one aggregate 100%/0% and Layer 3 as one
aggregate 45%; neither says which checks are doing the work.

This file holds the findings available **without** a GPU, the pending
documentation corrections, and the runbook for the one Colab session that
produces the rest.

---

## 0. STATUS — the GPU run landed (2026-08-20)

**The re-run described in §4 has been executed. The numbers exist.** GPU access
is via the Colab CLI; see `SESSION_HANDOFF.md` §0 for the working commands.

- **Full per-check ablation:** `benchmarks/CHECK_ABLATION.md` (generated — do
  not hand-edit; re-run `python3 benchmarks/analyze_check_ablation.py`).
- **Raw input:** `benchmarks/autokernel/files/results_raw.json` — 11 systems,
  440 mutant results, **1343 check records**, 40 mutants across 29 operators.
- **Headline benchmark:** `benchmarks/autokernel/files/results.md`. The prior
  root-level `results.md` / `results.json` (Aug 6) were left untouched.

Layer 2 top catchers, from real data: `weight_magnitude` 31/40 (78%),
`cross_shape` 29/40 (72%), `perturbation_tolerance` 23/40 (58%). *(Note
2026-08-27: `cross_shape`'s count includes one bug-manufactured catch — the
`layernorm/wrong_variance_estimate` catch exists only because of the layernorm
reference kernel's padded-lane variance bug at shape (1000, 333); under a
corrected reference it is 28/40. See
`verification_runs/layernorm_mask_bug_2026-08-27/FINDINGS.md` §2. Update
2026-08-28: the reference fix shipped and the −1 correction is now verified on
GPU — the corpus arm diff shows exactly that one catch disappearing and
nothing else; `verification_runs/layernorm_mask_fix_2026-08-28/FINDINGS.md`.)* Zero errors and
zero false positives on the reference across every Layer-2 check **on the fixed
corpus inputs** — see §3.0's status notes before citing that as a general
property, and note the 36% reference-failure rate measured under
adversarial-search inputs applies to **Layer 1**, not Layer 2.

**Update 2026-08-21: §3.0's Layer-1 defect is FIXED and §3.0 is closed.** The
`check_kernel_executed` false positive is repaired and verified on a T4 (25/25
recorded false positives cleared against real Triton kernels, ghost still caught
25/25, zero verdict changes across the whole corpus). The 0%-FP scope caveats
were **not** relaxed — that needs a fresh adversarial-search run, which was
deliberately not spent. See §3.0's 2026-08-21 block.

**§2 below (what the re-run adds) and §4 (the runbook) are now historical.** They
are kept because §2's error/fail reasoning still explains how to read the table,
and §4 documents how the run was produced — but neither is pending work.

### The run surfaced one real defect — see §3.3

`analyze_check_ablation.py` **crashed** on the first attempt against real data.
Two of the 1343 check records carried an `int` in `subchecks`; the reader
iterated it and died with `TypeError: 'int' object is not iterable`, taking
attribution for all 94 checks with it. Fixed in three places, with a permanent
negative control. Full write-up in §3.3.

---

## 1. Answered already — no re-run needed

Static enumeration across all 29 specs (AST + grep) corrects three premises the
work was scoped on.

### 1.1 Layer 2 is 62 check instances, not ~6

| Component | Count |
|---|---|
| Fixed checks (`output_shape`, `perturbation_tolerance`, `cross_shape`, `weight_magnitude`) | 4 |
| `backward_pass` | 0 — never runs, see 1.2 |
| Per-spec adversarial battery | **58 instances across 36 distinct names** |

The battery is very unevenly distributed: **17 of 29 specs have exactly one**
adversarial input, while `flash_attention` and `matmul` have six each. Any claim
that Layer 2 applies "~5 adversarial inputs per operator" (`BENCHMARK_RESULTS.md`
§2, `sota_checks_registry.py`) is true only on average and false for most
operators individually.

Most-reused adversarial names, i.e. where cross-operator redundancy is most
likely to show up in the re-run:

| Name | Specs using it |
|---|---|
| `large_magnitude` | 8 |
| `near_zero_variance` | 4 |
| `non_power_of_two` | 4 |
| `padded` | 3 |
| `second_half_dominant` | 3 |
| `all_negative_padded` | 3 |

`large_magnitude` (8 specs) also overlaps by construction with
`check_weight_magnitude`'s `large_uniform` / `large_random` variants, which run
on **every** operator. That is the single most likely redundancy inside Layer 2,
and it is why `weight_magnitude` is now decomposed per-variant.

### 1.2 `backward_pass` is dead code — zero catches, provable today

`KernelSpec.requires_backward` defaults to `True` (`base_spec.py:23`), but **all
29 specs override it to `False`**. `check_backward_pass`
(`shape_generalization.py:221`) therefore never executes anywhere in the
TritonBench corpus.

This is deliverable (b)'s first answer and needs no measurement. It is either
removed, or a spec has to opt in and a mutant has to target gradients — at
present it is ~57 lines of code contributing nothing. The generated report lists
it under **"never ran"**, distinct from checks that run and never catch.

### 1.3 Layer 3 is 69 property instances, not "~100+"

69 instances across **29 distinct property names**. `BENCHMARK_RESULTS.md` §2's
*"~100+ individual checks across 29 operator specs"* overstates by ~45%.

**`batchnorm` has zero algebraic properties**, so Layer 3 structurally cannot
catch its mutant — which is exactly what `results.md` shows (algebraic-only, 0%
for batchnorm). That is a coverage gap, not a measurement.

Most-reused properties: `positive_scale_equivariance` (8 specs),
`positive_scale_invariance` (7), `shift_invariance` (5), `precision_coercion`
(5), `shift_equivariance` (5).

### 1.4 Smaller correction

`spec.valid_shapes` is 5 for most specs but **2–4 for the pooling and norm
families** (`avg_pool3d`/`max_pool3d`: 2; the 1-D/2-D pools: 3; `batchnorm`,
`groupnorm`, `instancenorm`: 4). `sota_checks_registry.py`'s "correctness holds
across 5 pre-defined shapes per operator" is wrong for 8 of 29 specs.

*(Checked and withdrawn: `output_dtype` **is** correctly set to `torch.int64`
for argmax/argmin — it is passed in `get_spec()` rather than the class body, so
index outputs do route through `_check_exact_match` as intended.)*

---

## 2. What the re-run adds

`benchmarks/analyze_check_ablation.py` generates `benchmarks/CHECK_ABLATION.md`
from `results_raw.json`:

- **(a)** per-check catch-rate table — `ran | caught | catch rate | errors |
  skips | FPs on reference`, with `cross_shape` and `weight_magnitude`
  decomposed into their per-shape and per-variant sub-probes.
- **Redundancy** — pairwise catch-set overlap across every check that catches
  anything, labelling pairs as `identical`, `subsumes`, or `partial overlap`.
  `identical` pairs are the actionable finding.
- **(b)** zero-catch checks, split into *never ran* and *ran but never caught*.
- **(c)** Layer 3 per operator-spec × property (69 rows), plus a rollup by
  property name across specs (29 rows).

### The error/fail distinction, and why it is not cosmetic

Only `outcome == "fail"` counts as a catch. `error` (the check raised) is
reported in its own column and never as a catch.

This comes directly from the item-#1 audit: an identical bare-`except` pattern
in `baselines.py:autokernel_gate` scored a `ValueError` — raised by a
wrongly-built input — as a legitimate gate failure, and that single behaviour
produced **all** of that baseline's reported 18% false-positive rate. The same
hazard already exists in this repo: `shape_generalization.py`'s `monotone_rows`
variant used to raise `RuntimeError` on 1-D primaries and, per its own comment,
*"coincidentally match[ed] the expected mutant-catch verdict."* A crashing check
must not look like a working one in the table built to find dead checks.

The generated report includes a **consistency check** per layer: the set of
mutants with ≥1 `fail` record must equal the harness's caught set. A mismatch
means some mutant's "catch" was actually a crash — reported as a finding, not
swallowed.

---

## 3. Pending documentation corrections

Held until the numbers land, so `BENCHMARK_RESULTS.md` §4 is edited once rather
than twice.

### 3.0 PRIORITY — `check_kernel_executed` false-positives on correct kernels

**Logged separately from the table below, and ranked above every entry in it,
because this is a potential false-positive source inside Layer 1 — the same
layer that produces the project's headline 0% false-positive claim.** Everything
else in §3 is a documentation error; this is a possible *checker* defect.

`check_kernel_executed` (`verification/layer1_structural/runtime_guards.py:171`)
decides a kernel "likely ignores input (hardcoded output or ghost optimization)"
when `torch.equal(f(x), f(x + randn_like(x)*0.1 + 1.0))`. Two properties of that
perturbation make it unsound for some operators:

- it is dominated by a **constant `+1.0` shift**, and softmax-family and
  arg-extreme operators are shift-invariant by construction;
- it perturbs **only the primary tensor**, so for a multi-tensor operator whose
  output does not depend on the primary under the given companions — e.g.
  attention with constant K and V, where the attention weights stay uniform and
  the output is just V — no perturbation of the primary can change the output.

A correct reference kernel then fails a Layer-1 check.

**Verified:**

- **30 occurrences** across the adversarial-search history
  (`adversarial_results/search_history.db`), every one on a **shift-invariant
  operator**: causal_flash_attention 25, argmax 3, flash_attention 1, softmax 1.
- **18 of the 25 causal_flash_attention cases used `ones`/`zeros` fills**, i.e.
  constant K/V — the exact configuration in which perturbing Q provably cannot
  change the output.
- These are failures on the **reference**, not on a mutant, so each is a false
  positive by definition.

**Verified NOT to fire on the TritonBench benchmark corpus** (this was open when
the finding was first raised; it is now closed):

- All 12 corpus generators in `benchmarks/autokernel/files/tritonbench_registry.py`
  use `rng.normal`. A grep for `np.ones` / `np.zeros` / `np.full` across them
  returns **0** — the corpus is randn-only.
- `check_kernel_executed` has exactly **two call sites** (`verification/checker.py:127`
  and `benchmarks/autokernel/files/checker_adapter.py:171`), and both pass
  `primary`, the base corpus input.
- It is **never** called inside the adversarial-input loop (`checker.py:184`) or
  with `check_weight_magnitude`'s constant variants (`large_uniform`,
  `alternating_sign`). Those constant fills exist only in Layer 2 and never reach
  this Layer-1 check.

So the reported 0% false-positive rate is **not** affected by this, and no
benchmark number needs restating. The defect is real but currently unreachable
from the corpus.

**Residual risk, not yet quantified:** for index-returning shift-invariant
operators (argmax/argmin) the check could still false-positive on randn input if
*no* row's argmax flips under the perturbation. With 64 rows that requires all 64
to be stable simultaneously, which is why it is not observed — `results.md` shows
argmax at 0% FP. It is a tail risk, not an observed failure, and it would become
materially more likely on a small-batch corpus.

**Recommended fix (not yet implemented):** perturb multiplicatively or
per-element rather than by a constant shift, and for multi-tensor operators
perturb a companion tensor when the primary provably cannot affect the output.
Track alongside the `bool(None)` skip-coercion issue (`verification/checker.py:229`)
as the second known Layer-1/Layer-2 soundness defect.

> **The multiplicative half of that recommendation was measured and does not
> work** — see the 2026-08-21 status block below. It rescues **0 of the 20**
> recorded causal_flash_attention false positives, because the mechanism is not
> the constant shift. Kept here verbatim as the original reasoning; do not act
> on it without reading the correction.

#### STATUS 2026-08-21 — FIXED IN CODE. §3.0 is CLOSED.

The perturbation defect is repaired in
`verification/layer1_structural/runtime_guards.py`. Both call sites
(`verification/checker.py`, `benchmarks/autokernel/files/checker_adapter.py`)
pass the new arguments. Guarded by
`tests/instrumentation/check_kernel_executed_probe.py`.

**The diagnosis above was wrong about the mechanism.** §3.0 attributed the
false positive to the probe being "dominated by a constant `+1.0` shift", and
recommended a multiplicative/per-element probe. Replaying the 20 recorded
causal_flash_attention proposals — first in pure Python, then against the real
Triton kernel on a T4 — each candidate probe rescues:

| probe | rescues (of the 20 CFA FPs) |
|---|---:|
| old `x + randn*0.1 + 1.0` | 0 — reproduces all 20, as recorded |
| per-element multiplicative (**§3.0's recommendation**) | **0** |
| fresh independent draw | 0 |
| negation `-x` | 10 |
| **companion `V`, multiplicative** | **20** |
| companion `K`, multiplicative | 15 |

The real mechanism is that **K is constant or saturated across key positions**,
which makes the attention weights independent of Q for *every* Q. No
perturbation of the primary — of any magnitude or form — can change the output.
Prong (b), perturbing a companion, is the load-bearing fix; prong (a) is a
reasonable general improvement that happens to rescue none of these.

**The fix is a probe LADDER, evaluated as a disjunction** (pass as soon as any
rung moves the output, which can only reduce false positives, never create
them): (A) per-element multiplicative + additive, (B) negation, (C) fresh
independent draw, (D) each float companion perturbed with the primary held
fixed, (E) a reference-sensitivity guard — if nothing moved the candidate, run
the same ladder through the reference, and if the reference is *also* still,
report "not evaluable" rather than a ghost. Rung E is the only part correct by
construction rather than empirically; it is beyond what §3.0 recommended and
was added deliberately.

Cost on the pass path is unchanged: a correct kernel on ordinary input is moved
by rung A, so the common case is one extra kernel call, exactly as before.

**Measured, 2026-08-21, Colab T4.**

- **Real Triton kernels, the 25 recorded false positives** (CFA 20, argmax 3,
  softmax 1, flash_attention 1): old check fails **25/25**; new check passes
  **25/25**. Cleared by negation 12, companion 10, multiplicative 3.
- **False-negative control, same 25 real inputs:** a hardcoded-output ghost is
  still caught **25/25**. No FP-for-FN trade.
- **Corpus regression, full `run_benchmark.py`:** **zero verdict changes**
  across 440 mutant verdicts + 2200 reference verdicts over all 11 systems.
  `kernel_executed` remains **40 ran / 0 caught / 0 errors / 0 skips / 0 FPs**.
  Artifacts: `verification_runs/kernel_executed_fix_2026-08-21/`.
- **Seed stability:** all 76 recorded cases give the same verdict on all 8
  seeds. The OLD check did not — one of the 51 recorded passes (K scaled to
  0.01, i.e. K nearly constant) flips on 4 of 8 seeds with nothing else
  changed. Its verdict on near-degenerate inputs was partly a coin flip.

**Two things this did NOT do.** The published corpus numbers are unchanged and
were not restated — zero verdict changes means there was nothing to restate.
And the **0%-FP scope caveats added on 2026-08-20 were left in place**: they
are still literally true, and whether to relax them now depends on a fresh
adversarial-search run that was deliberately not spent (see below).

**Not measured, and deliberately so:** a fresh 80-proposal causal_flash_attention
search, which would give a headline "reference-failure rate 36% → ~11%". The
deterministic 25-case replay is stronger evidence for the fix itself — an LLM
search is stochastic and cannot be compared proposal-for-proposal — so the GPU
budget went there instead. If a headline *rate* is wanted, that run is the way
to get it, and the 9× `nan_inf` + `dtype_preserved` reference failures should
come back **unchanged** as a built-in control that this fix was targeted.

**Still open, unchanged:** `checker.py:229`'s `bool(None)` skip-coercion, the
other Layer-1/Layer-2 soundness defect §3.0 named. Not touched here.

#### STATUS 2026-08-20 — RESOLVED via documentation; code fix deliberately deferred

> **SUPERSEDED 2026-08-21 — the code fix has since landed; see the status block
> above.** This block is kept because it records why the documentation path was
> taken first and what was qualified as a result, all of which still stands.
> Where it says the defect "is still present in the code", read that as true
> *as of 2026-08-20*.

**Read this before treating §3.0 as closed: the defect is still present in the
code.** What was resolved is the *claim* it endangered, not the unsoundness.

**New evidence.** The clean 80-proposal `causal_flash_attention` re-run
(`SESSION_HANDOFF.md` §2.4, `adversarial_results/cfa_rerun_2026-08-20/`) had the
**reference kernel fail 29 of 80 proposals (36%)** — **20× `kernel_executed`**,
9× `nan_inf` + `dtype_preserved`. The previous basis for §3.0 was 30 occurrences
accumulated across all recorded history; this single run adds 20 and supplies a
rate rather than a count.

**Decision taken:** the documentation path, not the code path. The 0%-FP claim
is measured on the **fixed corpus inputs** from `spec.make_inputs`; the 36%
figure comes from **adversarially-generated inputs**. The two measure different
distributions and are not arithmetically contradictory, so the corpus numbers
stand as published. The actual exposure was rhetorical: "0% false positives"
read as "on any correct kernel", which this data contradicts for at least one
operator family.

**Qualified accordingly** — every site now carries the input distribution:

- `BENCHMARK_RESULTS.md` §1 (claim sentence + a scope note above the headline
  table), §8.3 (the rule-of-three bound, with an explicit statement of the
  population it was sampled from), §7 latency comparison, §11 standing summary.
- `SESSION_HANDOFF.md` §4 decision block.
- The Layer-2 zero-FP line in §0 of this file.

**Still open, as its own scoped work:** the `check_kernel_executed` perturbation
fix described directly above. It was explicitly **not** implemented in this pass,
so that it can be done, tested, and re-benchmarked on its own terms rather than
folded into a documentation change. §3.0 stays in this file as an open *code*
defect; only its documentation consequence is closed.
*(Done 2026-08-21 — and doing it on its own terms is what surfaced that the
recommended perturbation rescues 0 of 20. Folded into the documentation change,
that would have shipped as a fix and silently changed nothing.)*

> Note for whoever picks this up: `git diff` on
> `verification/layer1_structural/runtime_guards.py` **does** show uncommitted
> changes, but they predate this work (file mtime 2026-08-06) and concern
> `check_determinism`'s atomic-add tolerance, not `check_kernel_executed`. The
> 2026-08-20 documentation pass added nothing to that file.

### 3.1 Documentation corrections

| Doc | Claim | Correction | Blocked on |
|---|---|---|---|
| `BENCHMARK_RESULTS.md` §4 | 18% FPR caused by "fixed-tolerance adversarial-stability" **and** "bitwise determinism check false-positiving on `frobenius_norm`'s atomic-add" | **Both wrong.** Every FP came from an *exception*, not a tolerance comparison: an arity bug (layernorm variants are 1-tuples; reference needs 3 args) and a dtype bug (only generator in the repo missing `.astype(np.float32)`, so fp64 reaches Triton's `tl.dot`). `frobenius_norm` has 0% FP and no operator fails at stage 4. Full trace: `benchmarks/autokernel/AUTOKERNEL_BASELINE_AUDIT.md` §3 | #1 re-run |
| `BENCHMARK_RESULTS.md` §4, §8.1–8.2 | autokernel_gate at 68% / 18% | Replace with the published-faithful gate's numbers (expect higher catch, lower FP) | #1 re-run |
| `BENCHMARK_RESULTS.md` §2 | Layer 3 is "~100+ individual checks" | **69** | nothing — correct now |
| `BENCHMARK_RESULTS.md` §2 | "~5 adversarial inputs per operator" | 58 across 29 specs; 17 specs have exactly 1 | nothing — correct now |
| `sota_checks_registry.py` | `cross_shape` covers "5 pre-defined shapes per operator" | 2–4 for 8 specs | nothing — correct now |
| `sota_checks_registry.py` | `backward_pass` scope "none currently" | Stronger: it can never run — all 29 specs set `requires_backward=False` | nothing — correct now |


### 3.2 Withdrawn claims and standing constraints

Not documentation errors in the table above — one is a claim this project made
and has since retracted, the other is a constraint on work not yet started.
Both need to outlive the session that found them.

**1. WITHDRAWN: `hit_mutants: []` does not prove the checker missed the mutant.**

This project previously read `hit_mutants: []` across all 120
causal_flash_attention proposals as evidence that the checker never caught
`wrong_causal_mask`. **That does not follow and is withdrawn.**
`_evaluate_verdict` placed a mutant in `missed_mutants` when the checker failed
to catch it **or** when the checker caught it and naive allclose caught it too —
opposite outcomes, recorded identically. Since allclose catches this mutant on
ordinary input (0% missed for this operator in `results.md`) and all three
checker layers catch it (`benchmarks/LAYER_ATTRIBUTION.md`), the likelier
reading is that the checker *did* catch it and the search correctly reported no
allclose gap.

Full reasoning: `adversarial_results/CFA_NONHIT_ROOTCAUSE.md` §4.
`benchmarks/LAYER_ATTRIBUTION.md` has been corrected. **Any other text repeating
the original reading needs the same fix.** Note this is now only a historical
hazard for runs recorded before §2.2: verdicts written after it carry
`not_caught` / `caught_no_gap` / `mutant_records` and are unambiguous.

**2. CONSTRAINT on #8(b): `OPERATOR_CONTEXT` entries ship WITH the schema
extension, never after.**

#8(b) plans to wire the 8 excluded operators (`cross_entropy`, `groupnorm`, and
the 6 pooling ops) into the adversarial search by extending the `InputProposal`
schema for int64 class-index targets and int hyperparameters. **That change must
add each operator's `OPERATOR_CONTEXT` entry in the same commit.**

Rationale, measured rather than assumed: operators *with* prompt context
averaged 10.8 proposals-to-hit and hit 100% of the time; operators *without*
averaged 20.0 and produced the only non-hit. The missing-context failure mode
scales with how non-obvious an operator's calling convention is, and these eight
are the least obvious in the corpus — int64 class indices, `num_groups`,
`kernel_size`/`stride`/`padding`. Shipping the schema without the context would
reproduce the causal_flash_attention failure eight more times.

The same applies to the other six per-operator tables (see `SESSION_HANDOFF.md`
§2.3 finding **B2**): adding an operator currently means editing seven places
with nothing enforcing agreement.

### 3.3 FIXED — overloaded return slot broke the ablation reader on real data

Found by running §4 for real on 2026-08-20. **Not a documentation error: this
made the ablation table impossible to build from a real corpus run**, which is
item #2's entire deliverable.

**Symptom.** `python3 benchmarks/analyze_check_ablation.py` died immediately:

```
TypeError: 'int' object is not iterable   (analyze_check_ablation.py:86)
```

**Cause — one slot, two meanings.** `check_all_tiles_visited`
(`verification/layer1_structural/tile_coverage.py`) returned a 3-tuple whose
third element was a **column count**. `checker_adapter._try` assigns
`subchecks = result[2]` unconditionally, and by the adapter's convention a third
element means *per-sub-check records for a compound check* — a list.
`_expand` then iterated the int.

Slot 2 was overloaded too: `-1` in most branches but a message string in the
failure branch. `verification/checker.py`'s `_run_check` does
`str(result[1])`, which is where the "garbage `-1` sentinel FAIL" already noted
in `checker.py:142` came from. The sibling
`check_all_tiles_visited_generic` returns clean `(bool, str)` 2-tuples — this
function was the lone outlier.

**Scale — why it went unnoticed.** Exactly **2 of 1343** check records were
malformed, both `tile_coverage_softmax_positivity` (values 64 and 128), both in
`your_checker (structural only)`; that system's own record count is 322. Two bad
records destroyed the whole report. The check is gated to
`spec.name == "softmax"` and only populates the slot on the partial-coverage
branch, so it fires rarely — but when it fires, nothing downstream survives.

**Why the 13 existing assertions passed anyway.** The fixture in
`tests/instrumentation/check_ablation_report.py` modelled `subchecks` as
list-or-`None` only. The reader was tested exclusively against shapes it could
already handle — a textbook instance of §5's *passed while testing the wrong
thing* pattern (logged there as instance 7).

**Fix — three places, because one is not enough:**

1. `tile_coverage.py` — `check_all_tiles_visited` now returns
   `(passed, detail)`, matching its sibling and every other Layer-1 check. The
   column count moves into the detail string, where it was already duplicated
   and where it is actually read. `passed` values are byte-for-byte unchanged,
   so catch_rate / false_positive_rate stay comparable to earlier runs.
2. `checker_adapter._try` — `subchecks` is populated only when `result[2]` is a
   `list`. Stops any future misbehaving check from corrupting the ablation
   input for all the others.
3. `analyze_check_ablation._expand` — validates rather than trusts, because this
   script also reads `results_raw.json` files from **earlier** runs, which still
   contain the int. Without this, historical data stays unreadable.

**Permanent negative control.** `check_ablation_report.py` now carries a fixture
record with `subchecks: 64` — the exact real-world shape, on the same
`softmax/first_tile` mutant it occurred on — plus three assertions: the report
generates, the parent check is still counted, and no phantom `parent[sub]` rows
are synthesised. Verified to actually fire: stripping the `_expand` guard
reproduces the `TypeError`. Attached to an existing mutant so the original 13
assertions are undisturbed; all 16 pass.

---

## 4. Runbook — one Colab session covers #1 and #2

Every corpus kernel is a real `@triton.jit`; there is no CPU path, so nothing
below runs on the dev machine.

```bash
cd benchmarks/autokernel/files
python corpus_contract.py my_corpus.py     # validate the corpus first
python run_benchmark.py                     # writes results.md, results.json, results_raw.json

cd ../../..
python3 benchmarks/analyze_check_ablation.py   # writes benchmarks/CHECK_ABLATION.md
```

Systems in this single run:

| System | Serves |
|---|---|
| `autokernel_gate` | #1 — old approximation, kept for the delta |
| `autokernel_gate (faithful)` | #1 — published spec, `rtol=1e-5` |
| `autokernel_gate (faithful, rtol=0)` | #1 — strict-literal atol-only reading |
| `allclose`, `gpuemu` ×2, `propilot` | unchanged controls |
| `your_checker` full + 3 single-layer ablations | #2 — per-check records |

The `rtol=0` variant is a **separate system in the same pass**
(`functools.partial`), not a second benchmark run.

### Acceptance checks before trusting the output

1. `results.json`'s `catch_rate` / `false_positive_rate` / `missed_mutants` for
   the **untouched** systems (`allclose`, both `gpuemu`, `propilot`) must match
   the previous run. Any drift means the instrumentation leaked into behaviour.
2. `results_raw.json` exists and its `mutant_results` entries carry a
   non-null `check_records`.
3. The consistency check in `CHECK_ABLATION.md` reports OK for both layers — or,
   if it reports MISMATCH, that is a real finding about a crash-as-catch and
   needs reading before the tables are cited.

### Runtime

The two faithful gates each add 8 shapes × 3 dtypes plus edge cases per trial,
across 40 mutants × 6 trials. If the session runs long, add a `--systems` filter
rather than silently dropping coverage — a partial run that looks complete is
the failure mode this whole item exists to prevent.

---

## 5. Verified locally (no GPU)

- `/tmp/test_item2.py` — 44 assertions on the instrumentation: four-valued
  outcome mapping, subcheck passthrough, `_summarize`'s joined string
  byte-identical to the pre-change implementation, `summarize()` output
  byte-identical with and without `check_records`, and `harness._call` accepting
  both 3- and 4-tuple systems.
- `/tmp/test_ablation.py` — 13 assertions running
  `analyze_check_ablation.py` against a fixture with hand-derived counts,
  including an error-not-counted-as-catch case, an identical-catch-set
  redundancy pair, a never-ran roster entry, an operator with no properties, and
  a deliberate crash-as-catch that the consistency check correctly flags.

Both pass. `KernelChecker.run`'s control flow is untouched; only
`_check_cross_shape`'s return shape changed (a third element that existing
callers, which index `[0]` and `[1]` only, ignore).
