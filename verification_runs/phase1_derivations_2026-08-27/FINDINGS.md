# Phase 1: 27 operators added, 27 closed forms derived and autograd-verified. M3 re-fit NOT run — it needs a GPU.

**Produced 2026-08-27 on the Apple-silicon dev machine. No GPU, no Triton.**
Probes in `probes/`, raw data in `data/`, torch-level mutants in `mutants/`.
`verification/checker.py` was not touched.

---

## Verdict up front

| question | answer |
|---|---|
| Operators added | **27** (29 → **56**) |
| Closed-form `L` coverage | **27 of 27 derived.** `SUPPORTED_OPS` 27 → **54**; `STATIC_OPS` (shape-only) 10 → **14** |
| Are the derivations right? | **Yes, as calculus.** 380 invocations (27 ops × 3 input regimes × 5 seeds), worst relative error vs an autograd-exact Jacobian **2.98e-08**, 0 failures |
| Do the specs separate correct from buggy? | **54/54 mutants caught, 0 false positives on the correct implementation** — but see the calibration caveat below, this is not an out-of-sample number |
| **Updated M3 R²** | **NOT MEASURED. Requires a GPU.** See §4 — this is the one deliverable that is blocked, and it is blocked on purpose rather than approximated |
| Regressions | 0 — the pre-existing 27 operators produce identical profiles |

---

## 1. What "verified" means here, and what it does not

The 2026-08-25 round established two different things about the original 27, and
this round establishes only the **first** of them for the new 27:

| | established for original 27 | established here for new 27 |
|---|---|---|
| closed form == true Jacobian row norm | yes (implied) | **yes — 380 invocations, worst 2.98e-08** |
| closed form == converged Monte-Carlo probe **on real Triton kernels** | yes, 0.994–1.018× at K=20000 | **no — needs a GPU** |
| M3 predicts `adaptive_tol`, R² = 0.958 | yes, 38 matched invocations | **no — needs a GPU** |

The first is a **mathematical identity** and is device-independent, which is why
CPU is the right place to check it. The second and third are **measurements
against real kernels** and are not.

Anything in `structural_l.py`'s Phase-1 block is therefore
**derivation-verified and probe-unverified.** That distinction is written into
the module itself so it cannot be collapsed by a later reader.

---

## 2. The closed forms

Four are **shape-only** and join `STATIC_OPS`:

| operator | `‖J_i‖₂` | note |
|---|---|---|
| `cumsum` | `√(i+1)` | J is a triangular block of ones |
| `cumsum_reverse` | `√(n−i)` | |
| `cumsum_exclusive` | `√i` | |
| `std_reduction` | `1/√(n−1)` | **`var_reduction` is NOT shape-only** — `2‖x−m‖/(n−1)`, input-dependent. The two are not interchangeable and both are implemented |

The rest:

| operator(s) | `‖J_i‖₂` | needs |
|---|---|---|
| 9 activations (`relu`, `leaky_relu`, `sigmoid`, `tanh`, `selu`, `elu`, `softplus`, `hardsigmoid`, `new_gelu`) | `\|φ′(x_i)\|` | the input |
| `masked_cumsum` | `√(Σ_{j≤i} m_j²)` | the mask |
| `matvec` | `‖v‖₂` | the operand |
| `batched_matmul` | `‖B_b[:,j]‖₂` | the operand |
| `diagonal_matmul` | `\|B_ij\|` | the operand |
| `triangular_matmul` | `[i≥j]·‖B[:,j]‖₂` | the operand |
| 5 losses (`mse`, `huber`, `bce`, `kldiv`, `nll`) | `‖∇_x loss‖₂`, `m=1` | input + target |
| `rope` | `√(cos²+sin²)` — **exactly 1** for a real rotation | the cos/sin cache |
| `swiglu` | `√((silu′(a)b)² + silu(a)²)` | the input |
| `logsumexp` | `‖p_r‖₂`, `p = softmax` | the input |

**`rope` is the cheapest high-value entry in the table.** RoPE is an orthogonal
transform, so every row norm is exactly 1 — and it is the most-represented
uncovered operator family in TritonBench-G (12 of 184 files, see §5).

**The general form `√(cos²+sin²)` is kept rather than hardcoding `1.0`,
deliberately.** A cos/sin cache that is not a unit rotation is a real kernel bug;
the general form reports its true row norm instead of assuming orthogonality.
Verified both ways — a unit table gives exactly 1, and a deliberately scaled
non-rotation table still matches autograd exactly.

---

## 3. Measured: catch rate, and three real spec defects found by measuring it

**Final: 54/54 mutants caught, 0 false positives on the correct implementation.**
2 mutants per operator, each encoding a named real failure mode. Torch-level, not
CUDA — the shipped corpus is `load_inline` CUDA and does not compile without a
GPU, so these answer the question that *is* answerable now: do the batteries
separate a correct implementation from a buggy one?

**CALIBRATION CAVEAT, stated plainly: 100% is not an out-of-sample number.**
Three spec defects were found *because* mutants escaped, and were then fixed —
so the battery has been tuned against this mutant set. The out-of-sample facts
are the **0 false positives**, which held at every iteration, and the three
defects themselves.

Layer 1's AST checks were deliberately **not** run: they test for Triton kernel
launches and would fail a torch candidate *including the correct one*, which
would have manufactured a meaningless 100%.

### The three defects, each found by a mutant escaping

**D1 — a varied hyperparameter that was never varied.** `leaky_relu`, `elu` and
`softplus` had `make_inputs` hardcode the torch **default** for their scalar
argument. A kernel that ignores the argument and hardcodes that same default is
then **bit-identical to the reference on every input the spec can generate** —
undetectable by any check in the pipeline, by construction. Measured: the
`hardcoded_slope` and `ignores_alpha` mutants escaped the entire battery.
*Fix:* fold the hyperparameter into `valid_shapes` (the `PoolKernelSpec`
precedent) and vary it. Both mutants now caught.

**D2 — a saturating adversarial input that did not saturate.** Six activation
specs used `x = ±40` for their "saturating" variant. **fp32 `exp` overflows near
x = 88**, so at 40 (`exp(40) = 2.4e17`, comfortably in range) a naive unstable
formulation stays finite and the variant **silently tested nothing**. Measured:
`softplus`'s `naive_overflow` mutant escaped at 40 and is caught at 100.
*Fix:* raised to ±100 across all six.

**D3 — `relu` had no variant in the small-positive band.** Its variants sat at
zero, negative, and large — so a kernel that zeroes everything below a small
epsilon (a real thresholding bug) matched the reference on all of them.
*Fix:* added a `small_positive` variant.

### Two of my own mutants were degenerate, and that is worth recording

`sigmoid/unstable_exp` was first written as `1/(1+exp(−x))`. **That is not
unstable in fp32** — `exp(inf)` gives `1/(1+inf) = 0`, which is the correct
answer. The genuinely unstable form is `exp(x)/(1+exp(x))` (`inf/inf → NaN`).
Separately, my hand-written "correct" BCE did not floor `log` at −100 the way
`torch.binary_cross_entropy` documents, so **my correct implementation** was the
one disagreeing with the reference at `p ∈ {0,1}` — it produced the run's only
false positive until fixed. Both are recorded because "the mutant escaped" and
"the mutant was not actually a bug" look identical in a summary table.

---

## 4. The M3 re-fit — RUN 2026-08-27. This section's prediction was WRONG; see GPU_NATIVE.md §4

> **SUPERSEDED.** The GPU round has been run on a T4 with 27 purpose-written
> Triton kernels. **M3's R² over the full 54-operator corpus is 0.8567**, down
> from 0.9579 — so this section's *direction* was right. Its *attributed cause
> was wrong*: the `m=1` losses come out at −1.1% median (essentially unbiased),
> and the whole degradation is the **scan family at +24.7%**. Excluding scans,
> R² is 0.9635.
>
> The "`cross_entropy` is already M3's worst case at +121%" premise below is a
> **misreading of the original round**: +121% is `cross_entropy`'s residual
> under model **M1′** (the theorem's leading term used as an estimator,
> R² = −0.34), not under **M3**. Under M3 it is **−1.8%**. Everything below is
> kept as written, unedited, because the reasoning that produced a wrong
> prediction is worth being able to re-read.

### Original text, superseded

**`CORPUS_EXPANSION_PLAN.md` §4.2 predicted M3's R² would fall from 0.958 when
the 5 `m=1` loss operators were added. That prediction is not tested here and
must not be reported as if it were.**

M3's R² is computed by matching the closed-form prediction `y_M3` against a
**GPU-measured `adaptive_tol`**, which comes from `check_perturbation_tolerance`
probing a real Triton reference kernel (`data/gpu_native.jsonl` in the
2026-08-25 round: 239 invocations). There is no measured `tol` for a new
operator without running its kernel, and there is no Triton or CUDA on this
machine.

`SESSION_HANDOFF.md` §0 is explicit and this round follows it:

> Do not substitute CPU approximations, simulated results, or estimates for a
> number that requires a GPU.

**What is ready for the GPU session:** `probes/derive.py` carries all 27 forms;
`structural_l.row_norms` already dispatches them; the 2026-08-25 harness
(`gen_native.py` / `fit_tol.py`) extends by adding the new operator keys to its
op table. The run is closed-form-vs-probe at K ∈ {400, 4k, 20k} plus the M3
re-fit, exactly as before.

**What to expect, labelled as prediction and not as result:** the direction
should still be **down**, and more sharply than §4.2 estimated. The corpus goes
from **1 `m=1` operator to 6** (`cross_entropy` plus the five losses), and
`cross_entropy` is already M3's single worst over-prediction at **+121%** — the
`√(2 ln 2m)` bound is 0.887 at `m=1` against a half-normal reality of 1.90.
Three further operators (`sigmoid`, `tanh`, `swiglu`) have measured row-norm
spreads around **5e10 / 7e9 / 8e9** on saturating input, against a previous
corpus maximum of **38.7** (`softmax`). **Report whatever comes out. Do not tune.**

---

## 5. Profile degeneracy — partly wrong, and the SPREAD numbers here are an artifact

> **SUPERSEDED in part.** The zero-fraction results below stand. The **spread**
> figures (`sigmoid` 4.8e10, `tanh` 7.2e9, `swiglu` 8.0e9) **do not**: they
> divide by a *denormal* median in the saturating regime, which manufactures an
> arbitrarily large ratio. Measured natively from live kernel execution, the
> largest spread anywhere in the Phase-1 corpus is **16.3** — below the previous
> corpus maximum of 38.7. See GPU_NATIVE.md §5.

### Original text, superseded in part

`CORPUS_EXPANSION_PLAN.md` §4.1 predicted `relu`, `leaky_relu`, `hardtanh` and
`hardsigmoid` would have degenerate (mostly-zero) row-norm profiles. Measured on
saturating input, mean over 5 seeds:

| operator | zero fraction | max/median spread | verdict vs prediction |
|---|---:|---:|---|
| `hardsigmoid` | **0.967** | 1.00 | **predicted, confirmed** — genuinely degenerate |
| `tanh` | 0.657 | 7.2e9 | not predicted; sparse |
| `relu` | 0.538 | 1.00 | predicted, but **sparse, not degenerate** |
| `new_gelu` | 0.457 | 1.03 | not predicted; sparse |
| `triangular_matmul` | 0.417 | 1.62 | structural zeros above the diagonal, by construction |
| **`leaky_relu`** | **0.000** | 80.2 | **predicted, REFUTED** — the 0.01 negative slope means no row is ever dead |
| `sigmoid` | 0.148 | **4.8e10** | not predicted, and the largest spread in the corpus |
| `swiglu` | 0.000 | 8.0e9 | not predicted |

**Two corrections to the plan.** `leaky_relu` does not degenerate at all — a
nonzero negative slope keeps every row live, which the prediction missed by
grouping it with `relu`. And the more consequential failure mode is not zeros at
all: it is **spread**, where `sigmoid`, `tanh` and `swiglu` exceed the previous
corpus maximum by **eight orders of magnitude**. `y_profile`'s existing
subsampling keeps rows above 1e-3 of the max, so this is handled mechanically —
but it is far outside the regime M3 was validated in, and it is a second reason
to expect the R² to move.

`hardsigmoid` is the one operator whose profile genuinely collapses. Flagged in
`structural_l.py` at the branch itself, not fixed — M3 over an all-but-empty
profile is not a validated regime, and inventing a fallback without a
measurement would be guessing.

---

## 6. Two integration findings, one of them a live pre-existing bug

**F1 — `kernelbench_operator_registry` had three rotated stems, and they were
silently defeating the thing the registry exists to do.** The file mapped
`37_L1Norm_` / `38_L2Norm_` / `39_FrobeniusNorm_`; the real checkout is
`37_FrobeniusNorm_` / `38_L1Norm_` / `39_L2Norm_`. A stem that does not match
makes `resolve_operator_key` return `None`, which leaves
`skip_shape_guessed_layer3` **False** — so all three fell through to the
shape-based `try_*_layer3` detectors. `l1norm`, `l2norm` and `frobenius_norm`
are all "exactly one tensor argument, ≥2D", which is exactly
`try_softmax_layer3`'s detection heuristic, so a **correct** norm kernel would
have softmax's rows-sum-to-one invariant asserted against it and be **rejected**.

That is the precise failure the registry's own docstring says it exists to
prevent, reached through a typo inside the file. The docstring's standing
"UNVERIFIED: built from the directory listing ... not confirmed against your
actual checkout" was right to flag it. **Fixed, and all 49 stems now verified to
match a real file**; the docstring carries a one-line command to re-check.

**F2 — eight registry entries changed behaviour as a side effect of Phase 1,
and were checked rather than assumed.** `19_ReLU`, `20_LeakyReLU`, `21_Sigmoid`,
`22_Tanh`, `27_SELU_`, `28_HardSigmoid`, `29_Softplus`, `31_ELU` were
deliberately mapped to keys with **no spec file**, which `_run_known_operator_properties`
treats as "nothing to check". Phase 1 created those spec files, so the premise
of that comment no longer holds for them. Verified directly: all eight load,
run, and produce **0 crashes and 0 false positives** on a correct candidate —
and they now contribute **7 real algebraic checks** where they previously
contributed none. `30_Softsign` and `32_HardTanh` keep the old no-spec
behaviour; both were dropped from Phase 1 as absent from both real corpora.

12 further L1 stems were registered (batched matmul, matvec, diagonal and
triangular matmul, MinGPTNewGelu, the four scans, MSE/Huber/KLDiv), each checked
against its real `forward()` signature first. **KernelBench L1 registry coverage:
33 → 45 of 100.**

---

## 7. Test suite

`295 passed, 1 failed, 1 skipped` (468 s).

The one failure is
`test_worker_parsing.py::TestWorkerRetry::test_all_retries_exhausted_raises`,
in the adversarial-search worker's retry path. **Pre-existing and unrelated:**
`worker.py` was already modified before this session and neither it nor its test
was touched here. Recorded rather than fixed — it is someone else's in-flight
change, and folding a fix into this round would confuse two unrelated diffs.

---

## 8. `bce_loss` classification — RESOLVED 2026-08-27

Full analysis in `GPU_NATIVE.md` §7; probe `probes/bce_classification.py`
(banked data only, no GPU). Summary, because the paper needs one clear line:

**`bce_loss` is NOT a floor exception.** Its `tol` is a genuine `3σ·q95` value
at 31–38× the `1e-6` clamp, and its q95 sample is 106 ulp against the 2–3 ulp of
the fp32-floor cases. Neither documented floor mechanism applies.

**It belongs to a new third category: the `m = 1` sensitivity limit.** At `m = 1`
both `tol` and `L` are statistics of the same scalar response, so
`y = tol/(3σL)` reduces exactly to `q95(s)/√(E[s²])` — a shape ratio that
curvature moves only ~11% (1.960 → 1.737) inside a 7.17× window. The bound would
need a ~296% linearisation defect to break; `bce_loss` is at 53%.

**Recommended classification:** keep `bce_loss` in the "passes" column — the
bound holds on a real measurement — but exclude all six `m = 1` operators from
any claim that their *linearisation* was validated. Sound, but low-powered.
**Do not extend the `equal_attention_weights` footnote to cover it**; that
footnote is about a vacuous bound, and this bound is not vacuous.

> **CORRECTED 2026-08-27:** "new third category" must not be read as a third
> *disjoint region*. `../theory_audit_2026-08-27/` measured that the two floor
> categories are the absolute and relative arms of one resolvability
> criterion (15/15 flagged, and the absolute-floor cases also sit at 2.5–6
> ulp — inside the relative arm), and that 8 of 18 m=1 invocations trip the
> relative arm themselves (median s/ulp 12–25 vs the 32 screen; the 106-ulp
> figure above is the q95 sample, not the median). The m=1 *mechanism* is
> distinct and everything else in this section stands; the categories overlap
> on the real data. See GPU_NATIVE.md §7's correction box for the full
> statement.

---

## 9. Reproduce

```bash
python probes/derive.py          # 28 closed forms vs autograd, one pass
python probes/validate.py        # 380 invocations, 5 seeds x 3 regimes + degeneracy
python probes/verify_shipped.py  # same, against the SHIPPED structural_l.py
python probes/run_phase1.py      # catch rate + false positives over 54 mutants
python probes/xref.py            # TritonBench-G / KernelBenchX family breakdown
python probes/picks.py           # Phase-1 pick validation against both corpora
```

`torch.autograd.functional.jacobian(..., vectorize=True)` is **pathologically
slow** in torch 2.13 on this machine — 285 s versus 0.01 s without it. The
probes do not pass it. This cost one 5-minute timeout before it was found.
