# Results summary

**Assembled 2026-08-21. Every number here traces to an artifact already on
disk — nothing in this file is new.** Source is named per section. Where two
artifacts disagree, that is stated rather than resolved silently.

---

## 1. Accuracy — checker vs. SOTA baselines

TritonBench, 29 operators, 40 mutants, 200 reference trials per system.
**Source: `benchmarks/autokernel/files/results.json`** (regenerable offline via
`benchmarks/regenerate_report.py`).

| system | catch rate | false-positive rate | p50 latency |
|---|---:|---:|---:|
| **`your_checker (full)`** | **100.0%** | **0.0%** | 14.41 ms |
| `your_checker (numeric only)` | 100.0% | 0.0% | 17.82 ms |
| `your_checker (algebraic only)` | 45.0% | 0.0% | 1.13 ms |
| `your_checker (structural only)` | 10.0% | 0.0% | 3.74 ms |
| `autokernel_gate (faithful)` | 80.0% | 0.5% | 12.15 ms |
| `autokernel_gate (faithful, rtol=0)` | 80.0% | 1.0% | 14.15 ms |
| `autokernel_gate` (original approximation) | 67.5% | 18.0% | 2.75 ms |
| `gpuemu (adversarial_value)` | 82.5% | **81.5%** | 1.01 ms |
| `gpuemu (boundary_shape)` | 65.0% | 0.0% | 2.69 ms |
| `allclose` | 57.5% | 0.0% | 0.56 ms |
| `propilot` | 10.0% | 0.0% | 0.00 ms |

**The full checker is the only system at 100% catch, and it is one of six at 0%
false positives.** `gpuemu (adversarial_value)` is the only baseline whose catch
rate approaches it, at an 81.5% false-positive rate — not a usable trade.

Two caveats that must travel with this table:

- **The `autokernel_gate (faithful)` FP figures are one draw from a
  distribution, not point estimates.** `frobenius_norm`'s reference uses
  `tl.atomic_add`; float addition is non-associative, so any *bitwise*
  determinism check flips run to run. Measured across three independent runs the
  same check flagged it 1/5, 2/5 and 0/5 times with no code change
  (`SESSION_HANDOFF.md` §3). This project's own checker is immune — it uses a
  tolerance, not `torch.equal` — and showed zero determinism FPs in every run.
- **Units discrepancy — RECONCILED 2026-08-28.** `results.json` gives the
  faithful gate 0.5% / 1.0% (1 and 2 FPs of 200, matching its own per-op
  frobenius rates of 1/5 and 2/5), and `results.md`'s displayed 0% / 1% is
  the integer rounding of those same values — the two artifacts of the run
  agree. `SESSION_HANDOFF.md` §1's prose "1% / 2%" matched neither artifact
  and was a transcription error; it has been corrected in place. The
  artifact values 0.5% / 1.0% stand (still one draw from the frobenius
  flake distribution, per the caveat above). Cite `results.json`.

---

## 2. The corrected latency multiple

**`allclose` is 29x faster than the full checker — not 354x.**
**Source: `BENCHMARK_RESULTS.md` §8.1.2 (lines 396-397, 467, 483).**

The published 354x was wrong by roughly 8x, for two measurement defects that
were found and fixed rather than argued around:

1. `harness.allclose_system` timed only the numpy comparison — the kernels ran
   *before* its timer started — so it compared comparison-time against
   full-pipeline-time.
2. Triton JIT compilation was charged to whichever system the results dict
   happened to run first. **84% of the "checker latency" was compilation**, not
   checker work.

Both are fixed (`harness._warm` plus one timer convention across all seven
systems), and the honest steady state is **~14.4 ms p50 per check, 6.9 s across
the whole corpus warm**. The trade the 29x buys is 57.5% → 100% catch.

---

## 3. The false-positive correction sequence

**36.2% → 17.1% → 0.0%**, on the adversarial-search input distribution.
**Source: `SESSION_HANDOFF.md` §6.1, §7 and §5 instance 13; artifacts in
`verification_runs/kernel_executed_fix_2026-08-21/` and
`adversarial_results/cfa_rerun_postfix_2026-08-21/`.**

| stage | rate | what changed |
|---|---:|---|
| 2026-08-20 baseline | 29 of 80 = **36.2%** | as measured |
| after the `check_kernel_executed` fix | 12 of 70 = **17.1%** | `kernel_executed` FPs went to 0 of 70 |
| after diagnosing the remainder | **0 of 58 = 0.0%** | the residual was never a false positive |

**The one-line honest story: two of the three steps fixed a defect, and the
third found that the number itself had been measuring the wrong thing.** The
final 12 "failures" were *crashes on out-of-domain input* — 7 non-power-of-two
head dims and 5 rank-3/4 tensors, against a kernel documented as 2-D with a
power-of-two last dim — scored as checker failures because `check_nan_inf` and
`check_dtype_preserved` return a plain `False` for **any** exception. The 2x2 is
exact in both directions: 0 in-domain failures, 0 out-of-domain passes. The
17.1% was published and is retracted in `BENCHMARK_RESULTS.md` §8.3.1.

The corpus 0% and the adversarial 36% were never contradictory — different input
distributions — but "0% false positives" invited the reading "on any correct
kernel", and that reading is what the sequence above corrected.

---

## 4. Theory highlights

**Source: `benchmarks/NUMERICAL_THEORY.md`, derived from
`benchmarks/BUG_CLASS_THEORY.md` (120 of 120 recorded verdicts predicted; 0
falsifying cases).**

### 4.1 Tolerance invariance — the load-bearing result

For mutant `f̃`, reference `f`, input `x`, residual `R(x) = f̃(x) − f(x)`: the
`allclose` baseline is blind for **every** tolerance pair `(a, r) ≥ 0` — `a = r
= 0` included — **iff `R(x) ≡ 0`**. The proof is immediate; the content is that
the partition it induces is non-empty on both sides.

Applying the decision procedure (re-run the simulation at `a = r = 0`) to the 20
simulated confirmed hits:

- **9 of 20 are exact masking** — `R(x) ≡ 0`, so **no allclose test at any
  tolerance or precision can catch them**. Three mechanisms: algebraic identity
  (`γ ≡ 1`), floating-point absorption (discarded terms below ½ ULP of what is
  retained), discrete uniqueness (no tied maxima).
- **11 of 20 are tolerance straddling** — real but sub-tolerance. These close if
  the baseline is tightened.

**Caveat that must be quoted with the counts:** the simulation runs float64, the
kernels float32, and the absorption threshold differs (`v > 40.9` vs `v > 21.5`).
Two proposals sit at `v = 20`, so **the on-hardware split is likely 11/9, not
9/11.** The qualitative claim is unaffected; the counts are.

**Consequence:** the honest headline is two numbers, not one. The straddling
class invites "then tighten your baseline" and that objection is correct. The
exact-masking class does not — it is the only claim that argues for property
checking as a *category* rather than as a better-tuned comparator. And its
masking inputs are the *ordinary* ones: `γ ≡ 1` is the default, unique maxima
are generic, saturated softmax is routine.

### 4.2 matmul — a confirmed out-of-sample prediction, and a retraction

**Source: `verification_runs/matmul_prediction_2026-08-21/`.**

**Confirmed.** The derivation predicted, before any run, that zeroing
`A[:, K/2:]` makes `matmul:partial_k_reduct`'s residual identically zero and
therefore a hit on the first attempt. Result: **credited on 3 of 3 zeroed
proposals and 0 of 1 un-zeroed control.** The control holds shape, fill and
strides fixed and varies only the zeroing, so the masking is attributable to the
residual and nothing else. This is the first claim in the project made before
the data rather than fitted to it.

**Retracted in the same run.** `skip_boundary_tiles` came back masked at
`M = N = 100`, where the tile-alignment condition says it should be plainly
visible; a second hypothesis (constant output makes out-of-bounds stores
invisible) was tested with a non-constant-output proposal and **also failed**.
So the section's "8 of 8 cells" is **6 of 8 on conditions that survive testing**
— every recorded matmul proposal is simultaneously tile-aligned *and*
constant-output, so the data could never have distinguished those two
conditions, and the `skip_boundary` cells were right for the wrong reason.

**`skip_boundary_tiles`'s masking condition is explicitly unresolved and was not
pursued further.**

---

## 5. Process

**Negative controls caught real defects in this project's own fixes before they
shipped** — repeatedly, and in both directions: a control that passed via
`NameError` instead of detecting anything; a leave-one-out control on an
OR-composed check that reported "delta 0" for the rung doing the entire job; an
"N/N now pass" claim with no paired "the old code fails N/N"; a seeded-path
floor of "0 of 80" that was underpowered rather than exact; and, tonight, a
matmul condition that fit every observation and was still the wrong mechanism.
Full list: `SESSION_HANDOFF.md` §5, thirteen instances and three recurring
patterns.
