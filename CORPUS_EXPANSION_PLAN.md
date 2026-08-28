# Corpus expansion to 50+ operators — scoping

**Planning only. Nothing generated. Written 2026-08-27.**

Every count below that is attributed to KernelBench was produced by listing or
grepping `KernelBench/KernelBench/` in this checkout. Counts attributed to
TritonBench-G and KernelBenchX were **not** — see §1.3, which is the one place
this document is blocked.

---

## 0. Verdict up front

| question | answer |
|---|---|
| Can we reach 50+ operators? | **Yes, and without conv.** 29 existing + 25 template-reuse L1 additions = **54**, all inside spec/derivation families that already exist. |
| Can we reach 50+ *with* extended theorem coverage? | **~51 of the 54.** The additions are unusually cheap on the theorem side — 4 are shape-only. But see §4.2: the new set is heavier in `m=1` and piecewise operators, and the M3 estimator's fit quality is expected to get **worse**, not better. |
| Does `KernelChecker.run()` support fused (L2) kernels? | **Not today. Three specific changes required** (§3.2 items 2, 3, 4), none a one-liner. After them, **31 of 100** L2 problems are clean; the other 69 need conv and/or stateful-module handling. |
| Does it support full architectures (L3)? | **No, and I recommend not committing to it.** The obstacles are structural (§3.3), and the project's own `is_gap` framing gets weaker at that level. |
| Biggest single gap in the corpus | **Convolution — 35 of 100 KernelBench L1 problems, zero coverage today.** It is a phase of its own, not a line item. |

**Recommended target: 54 L1 operators, ~51 with closed-form `L`, plus a 31-problem
L2 empirical-only tier as a separate later phase. Not "50+ across L1/L2/L3".**

---

## 1. Inventory

### 1.1 What exists today — 29 operators

From `verification/specs/`:

```
argmax  argmin  avg_pool1d  avg_pool2d  avg_pool3d  batchnorm
causal_flash_attention  cross_entropy  flash_attention  frobenius_norm
gelu  groupnorm  instancenorm  l1norm  l2norm  layernorm  log_softmax
matmul  max_pool1d  max_pool2d  max_pool3d  max_reduction  mean_reduction
min_reduction  rmsnorm  scaled_dot_product_attention  softmax
sum_reduction  swish
```

- **27 have a closed-form `L`** (`structural_l.SUPPORTED_OPS`); 9 of those need
  only the output shape (`STATIC_OPS`).
- **2 are structurally excluded** — `argmax`/`argmin`. `checker.py` routes them to
  `_check_exact_match`; `structural_l.py` states the exclusion is deliberate
  ("an index has no meaningful Jacobian").
- ~40 mutants across `TritonBench/cheating/`.

**The single most important cost fact in this document:** those 27 closed forms
are not 27 independent derivations. They collapse into **6 families** —

| family | members | form |
|---|---|---|
| shape-only reductions + pools | 10 | `√n`, `√n/n`, `1`, `√W/W` |
| elementwise | gelu, swish | `\|φ'(x_i)\|` |
| matmul | matmul | `‖B[:,j]‖₂` |
| row-normalisation | softmax, log_softmax, l1norm, l2norm, frobenius_norm | one shared shape |
| mean-variance normalisation | layernorm, rmsnorm, groupnorm, instancenorm, batchnorm | layernorm's form, re-indexed |
| attention | flash, causal_flash, sdpa | one shared form |
| (one-off) | cross_entropy | `m = 1` |

They were produced, probed at K ∈ {400, 4k, 20k}, and fitted **in one Colab T4
session** (`kccgen`, 2026-08-25;
`verification_runs/adaptive_tol_theory_2026-08-25/generalization/FINDINGS.md`).
**The derivation unit is the family, not the operator.** Every estimate in §4
follows from that.

### 1.2 KernelBench — verified against this checkout

`level1` 100, `level2` 100, `level3` 50, `level4` 20 (HF models, out of scope).

**L1: 27 of 100 already covered → 73 new problems, in 5 families:**

| family | L1 problem numbers | count | already covered? |
|---|---|---:|---|
| matmul layout/shape variants | 2–18 | 17 | `#1` only |
| elementwise activations | 19–22, 27–32, 88 | 11 | `#23–26` covered |
| **convolution** | 50, 54–87 | **35** | **none** |
| scan / cumulative | 89–93 | 5 | none |
| losses | 94, 96, 98, 99, 100 | 5 | `#95` covered |

The 27 already covered are the norms (33–40), pools (41–46), reductions (47–49,
51–53), softmax/log_softmax/swish/gelu (23–26), matmul (1), cross-entropy (95),
SDPA (97).

**Near-duplicates that reuse an existing derivation rather than needing a new one:**
- All 8 norm problems (33–40) already map onto the two normalisation families.
- All 17 matmul variants reuse `‖B[:,j]‖₂`; only the *masked* forms (14, 15
  triangular; 12 diagonal) change it, and only by masking which columns contribute.
- `#88 MinGPTNewGelu` is the tanh-approximation GELU — a different `φ'`, same
  one-line elementwise template.
- `#24 LogSoftmax` is already in the corpus and shares softmax's `p`-based form.

**L2 (100 fused chains) and L3 (50 architectures)** — composition measured by grep:

| | contains Conv | contains BatchNorm | contains Dropout | calls `.eval()` |
|---|---:|---:|---:|---:|
| level2 (100) | 63 | 11 | 2 | **0** |
| level3 (50) | — | 19 | 23 | — |

**31 of the 100 L2 problems contain none of Conv / BatchNorm / Dropout.** They are
all Gemm- or Matmul-rooted with elementwise/reduction epilogues (`9`, `12`, `14`,
`18`, `22`, `28`, `29`, `30`, `37`, `40`, `45`, `51`, `53`, `55`, `56`, `59`, `62`,
`63`, `64`, `68`, `70`, `75`, `76`, `80`, `81`, `86`, `88`, `94`, `95`, `98`, `99`).
That subset is the only defensible L2 target, and §3.2 says what it still needs.

### 1.3 TritonBench-G and KernelBenchX — CLOSED 2026-08-27

**Both cloned and counted. `thunlp/TritonBench` and `BonnieW05/KernelBenchX`.**
Listings and the family breakdown are reproducible via
`verification_runs/phase1_derivations_2026-08-27/probes/xref.py`.

| corpus | claimed in the first draft | **actual** |
|---|---|---|
| TritonBench-G | 184 | **184 ✓** (`data/TritonBench_G_v1`; TritonBench-T is a separate 166) |
| KernelBenchX | 176, 15 categories | **184, 15 categories ✓** — the count was wrong, the category count was right |

**Correction: KernelBenchX has 184 problems, not 176** (171 distinct after
stripping `_bf16`/`_fp16`/`_int8` dtype variants). The 176 was unverified
recollection in the first draft. The 15 categories are Activation, Convolution,
Fusion, Index, LinearAlgebra, Loss, Math, MatrixMultiply, Normalization,
Optimizer, Pooling, Quantization, Random, Reduce, SpatialOps.

**The two corpora are not the same kind of object, and this matters more than
either count.**

*TritonBench-G is a scrape of real production Triton kernels* (vLLM, lightllm,
liger, flash-linear-attention, unsloth), not a taxonomy. It is dominated by
LLM-serving infrastructure and is heavily duplicated — 9 softmax variants, ~15
matmul, ~8 layernorm:

| family | files | family | files |
|---|---:|---|---:|
| attention | 29 | softmax | 11 |
| matmul | 20 | layernorm | 9 |
| quantization | 17 | loss | 7 |
| linear-attn / SSM | 16 | reduction | 6 |
| **rope / rotary** | **12** | rmsnorm | 5 |
| elementwise math | 11 | **scan / cumsum** | **6** |
| kv-cache / copy | 11 | gated activation (swiglu/geglu) | 4 |

*KernelBenchX is the clean taxonomy*: Activation 37, elementwise Math 30,
LinearAlgebra 26, MatrixMultiply 17, Normalization 12, Index 10, Reduce 8,
Loss 7, Fusion 60.

**Effect on the Phase-1 selection — five picks dropped, four swapped in.**

Five planned picks appear in **neither** corpus. They exist in KernelBench L1
because it enumerates `torch.nn` activations systematically, not because anyone
writes GPU kernels for them:

| dropped | TB-G | KBX | why |
|---|---:|---:|---|
| `softsign` | 0 | 0 | absent from both |
| `hardtanh` | 0 | 0 | absent from both |
| `hinge_loss` | 0 | 0 | absent from both |
| `triplet_margin` | 0 | 0 | absent from both; also needs a new 3-input spec class |
| `cumsum_exclusive` | 0 | 0 | absent — but **kept anyway**, it is a free shift of `cumsum` and costs no new derivation |

Swapped in on real-corpus evidence:

| added | TB-G | KBX | why |
|---|---:|---:|---|
| **`rope`** | **12** | 0 | the most-represented uncovered family after attention/matmul/quant. Orthogonal Jacobian → `‖J_i‖₂ = 1` exactly. Highest value-per-unit-effort in the whole plan |
| **`swiglu`** | 4 | 0 | every modern LLM FFN |
| **`logsumexp`** | 1 | 0 | plus heavy presence inside KBX `Fusion` and 6 KernelBench L2 chains |
| **`std_reduction`** | 0 | 1 | sibling of the already-covered `mean_reduction`; turned out **shape-only** |
| **`bce_loss`**, **`nll_loss`** | 0 | 1 each | replace `hinge`/`triplet`; both far more common in practice |

Strongly confirmed and kept: `cumsum` (**6 in TB-G**, real fla/Mamba kernels) and
`cumsum_reverse` (2 in TB-G) — the scans are validated by production code, not
just by KernelBench. Also `kldiv` (3+1), `matvec` (2+4), `batched_matmul` (4+5),
`relu` (4+3), `sigmoid` (0+7), `tanh` (1+7).

**Not pursued, recorded as genuinely new territory:** KernelBenchX's
LinearAlgebra (26 — cholesky/svd/qr/eig), Quantization (6), Index (10), Random
(2), SpatialOps (3). Random is structurally excluded for the same reason as
Dropout (§4.3). LinearAlgebra is a Phase-2-scale effort of its own.

## 2. Two ingestion paths, with very different costs

This is the second cost fact that drives everything, and it is easy to miss.

**Path A — `KernelChecker` + a `KernelSpec`** (`benchmarks/run_checker.py`).
Full pipeline: L1 AST + tile coverage, L2 algebraic properties, L3 numeric oracle,
per-operator adversarial battery, closed-form `L`. Requires a hand-written Triton
**reference** kernel and a spec file.

**Path B — `verification/kernel_adapter.py`** (`benchmarks/kernelbench_corpus/`).
Operator-agnostic. `_check_one_call` runs executes / shape / dtype / nan-inf /
allclose / perturbation-tolerance / determinism from a KernelBench problem file with
**no spec at all**; algebraic properties are optional and dispatched by
`kernelbench_operator_registry`. **The torch `Model` is the reference, so there is no
reference kernel to write.**

Path B already works end-to-end — `benchmarks/kernelbench_corpus/` has 15 problems
and 30 candidates with a completed gap report. **Empirical corpus expansion should go
through Path B; Path A should be reserved for operators whose theorem coverage or
adversarial battery is actually wanted.** Mixing this up is how a 25-operator addition
turns into 25 hand-written Triton references.

Cost anchor from the existing gap report: **~65 s per candidate per run.** A
54-operator corpus at 2 candidates each ≈ 108 runs ≈ **2.0 GPU-hours per full sweep**.
That is a recurring cost, paid every regression.

---

## 3. Multi-level feasibility — the design question

### 3.1 What `KernelChecker.run()` assumes

`run(candidate_fn, raw_kernel, reference_fn, inputs)` + a spec. The assumptions,
each read off the code:

1. **One primary tensor is perturbed; everything else is a fixed companion.**
   `checker.py` `_cand`/`_ref` build `(x,) + inputs[1:]`.
2. **`spec.make_inputs(shape, device, dtype)` synthesises every companion from a
   flat shape tuple.**
3. **Reference and candidate are pure functions of `inputs`** — called repeatedly
   (determinism ×2, perturbation ×20, plus one perturbation battery per adversarial
   variant).
4. **Layer 1 reads the source of a single entry-point callable**
   (`ast_analysis._get_source` → `inspect.getsource`).

Assumption 1 is the good news: **it already handles multi-tensor operators
correctly**, so a fused kernel taking `(x, W, b, ...)` is not excluded by it.

### 3.2 Fused L2 — supported after three changes, none trivial

**Change 2 — parameters live inside `nn.Module`, not in `inputs`.**
Verified in `level2/1_Conv2D_ReLU_BiasAdd.py`: `nn.Conv2d(...)` and
`nn.Parameter(torch.randn(bias_shape))` are constructed in `__init__`, and
`get_init_inputs()` returns only constructor arguments. **Two separately-constructed
modules therefore hold different random weights**, so reference and candidate are not
numerically comparable. `KernelChecker.run()` has no construct-once-share-weights
step, and `spec.make_inputs` synthesising fresh tensors per call is precisely the
wrong behaviour here. *Fix:* a spec variant bound to a constructed reference module
that hands the candidate its `state_dict`. Path B already solves this; the work is
lifting it into the spec path.

**Change 3 — Layer 1's AST checks would fail *correct* fused candidates.**
`check_partial_computation(candidate_fn, max_torch_ratio=0.5)` fails any entry point
where more than half of the compute calls are torch ops. A legitimate L2 candidate
that fuses the epilogue and leaves conv in cuDNN trips it. `check_ghost_optimization`
fails an entry point containing no Triton launch and no `.apply()` indirection.
**These are false FAILs on correct kernels — the exact failure class this project has
been most careful about** (`SESSION_HANDOFF.md` §3.0, §6.1). *Fix:* follow the AST
walk into submodule sources, or gate both checks off above L1. Neither is a one-liner.

**Change 4 — statefulness breaks assumption 3.**
`nn.BatchNorm2d` in training mode mutates running statistics on every forward, so
call *n* differs from call *n+1* **for a correct kernel** — and the checker makes
20+ calls. Dropout is worse: stochastic output makes `check_determinism` fail a
correct kernel outright. **11/100 L2 contain BatchNorm, 2/100 contain Dropout, and
0/100 call `.eval()`.** The existing `batchnorm` spec sidesteps exactly this by
taking running stats as explicit arguments and documenting *"INFERENCE MODE ONLY"* —
purity is an assumption the codebase states at source, not one I am inferring.
*Fix:* force `.eval()` and seed/patch dropout — **and say plainly in any writeup that
this is a modified problem, not KernelBench's own semantics.**

**Change 6 (tax, not blocker) — flat shape tuples don't scale.**
`GroupNormKernelSpec` already encodes `(N,C,H,W,num_groups)`; `PoolKernelSpec`
encodes `(*shape,k,s,p)`. An L2 conv chain needs ~10 hyperparameters in one tuple.
Refactor `valid_shapes`/`make_inputs` to a config dataclass **before** the expansion,
not during it.

**Change 5 — the theorem does not compose. This is the real limit.**
For a chain `f = g∘h`, only a **bound** `L_f ≤ L_g·L_h` follows. The generalization
round already measured what happens when the estimator is fed a bound instead of the
exact profile: **R² = −0.34, worse than predicting the mean** (§B.1 of that FINDINGS;
"a bound is not an estimator"). So theorem coverage at L2 means re-deriving the
composed row-norm profile **per chain** — and there are 100 distinct chains with
**no family amortisation**, unlike L1 where 27 operators collapsed into 6 families.
**This is why "multi-level theorem coverage" should not be promised.**

*One middle path, worth scoping separately and not free:* compute exact row norms by
**autograd JVP against the reference module**, which gives M3 the exact profile it
needs with zero per-chain algebra. Cost is `m` passes per check (`m` = output
elements) — *more* expensive than the current Monte-Carlo probe for large outputs,
cheaper only when `m` is small. That makes it plausible **specifically for the
31-problem Gemm-rooted subset**, whose outputs are small. It is new machinery
requiring its own validation round.

### 3.3 Full architectures (L3) — no

Beyond changes 2/3/4:

- **42 of 50 contain BatchNorm or Dropout** (19 and 23 respectively).
- `check_ghost_optimization` reads only the entry function. An L3
  `ModelNew.forward()` contains no Triton launch — the launches are inside
  submodules — so it reads as pure delegation and **fails every correct candidate**.
- The `is_gap` framing the project is built on ("allclose says fine, the checker
  disagrees") **weakens** at L3: allclose on a 50-layer network's output is already a
  strong test, and the per-operator adversarial batteries that produce this checker's
  edge have no architecture-level analogue.

Recommend recording L3 as future work with these reasons, and not committing to it.

---

## 4. Buckets, by cost

Bucket **(a)** = spec/registry entry + candidate + mutant, empirical coverage only.
Bucket **(b)** = (a) plus a closed-form `L` extending theorem coverage.

### 4.1 Bucket (b) — cheap, because the family already exists

| # | family | ops | derivation cost | notes |
|---|---|---:|---|---|
| A1 | elementwise activations | 11 | **one line each**, `\|φ'(x_i)\|`, same template as gelu/swish | see caveat below |
| A2 | matmul variants | 17 problems / ~5 forms | 2 small derivations (triangular mask, diagonal); transposed/batched change **layout only, not `L`** | needs a batched spec class — `MatmulKernelSpec.make_inputs` is hardcoded 2-D `(M,K,N)` |
| A3 | scan (cumsum, reverse, exclusive, masked) | 4 | **shape-only** — `J` is (masked) triangular ones, so `‖J_i‖₂ = √(#contributing inputs)` | joins `STATIC_OPS`; **cleanest theorem win in the plan** |
| A4 | losses (MSE, Huber, Hinge, KLDiv, TripletMargin) | 5 | each ≈ `cross_entropy`-sized: one gradient vector, one norm | all `m = 1` — see §4.2 |

**A1 caveat, flagged rather than assumed.** 4 of the 11 (ReLU, LeakyReLU, HardTanh,
HardSigmoid) are piecewise-linear. Their Jacobian exists almost everywhere, so the
*derivation* is trivial — but HardTanh/HardSigmoid have `‖J_i‖ = 0` on saturated
inputs, so the row-norm profile is mostly zeros and M3's max-of-`|z|` simulation
degenerates. **Derivation trivial, validation non-trivial.** These 4 need the profile
measured, not just written down.

### 4.2 Bucket (b) — the honest cost that is not derivation time

> **CORRECTED 2026-08-27 after the GPU round.** The prediction below — that the
> `m=1` losses would degrade M3 — is **refuted by measurement**. M3's R² did
> fall (0.9579 → 0.8567 over 54 operators), but the `m=1` losses are unbiased
> (−1.1% median) and the entire drop is the **scan family (+24.7%)**, whose
> nested-prefix Jacobian is the worst case for M3's orthogonal-rows assumption.
> The "`cross_entropy` at +121%" figure below is an **M1′** residual, not an M3
> one; under M3 `cross_entropy` is −1.8%. Kept unedited below.
> See `verification_runs/phase1_derivations_2026-08-27/GPU_NATIVE.md` §4.

Adding A4 puts **5 more `m = 1` operators** into a corpus that currently has one. And
`cross_entropy` (`m = 1`) is already **M3's single worst over-prediction, at +121%** —
the FINDINGS names the mechanism: at `m = 1` the `√(2 ln 2m)` bound (0.887) sits far
below the half-normal reality (1.90).

**So extending the corpus makes the estimator's headline fit look worse.** That is a
reason to add them — it stress-tests the estimator in its known-weak regime — but the
Phase-1 validation round should be budgeted expecting **M3's R² to move down from
0.958**, and that must not be treated as a regression to be tuned away.

### 4.3 Bucket (a) only — structurally excluded from the theorem

| operator | why | analogous to |
|---|---|---|
| `argmax`, `argmin` *(existing)* | discrete index output; no meaningful Jacobian | — |
| **`cumprod`** (L1 #90) | `J_ij = ∏_{k≤i, k≠j} x_k` — input-dependent, unbounded condition number, no useful `L` | not discrete, but equally excluded, and for a **different reason**: state it as such |
| Dropout-bearing problems (L2 #66, #83; 23/50 L3) | stochastic → `check_determinism` fails a *correct* kernel. Excluded from the **whole pipeline**, not just the theorem, unless forced to eval mode | — |

Note what is **not** on this list: Huber, Hinge, ReLU, LeakyReLU, HardTanh,
HardSigmoid. Piecewise ≠ excluded — `J` exists a.e. They need the §4.1 profile check,
not exclusion.

### 4.4 Bucket B — convolution, a phase of its own

35 of 100 L1 problems; ~7 distinct forms (conv1d/2d/3d, transposed 1d/2d/3d,
depthwise, pointwise, separable) × parameterisation.

- **Derivation is *not* the expensive part.** Conv is linear, so `‖J_i‖₂` per output
  element is the norm of the filter taps reaching it — **input-independent**, the same
  good case as matmul/batchnorm. ~4 derivations cover it: standard-with-padding
  (border outputs tap fewer inputs), grouped/depthwise, transposed (scatter, not
  gather), dilated. Composable, but each needs care.
- **The expensive part is kernel authoring.** A correct Triton conv2d with
  stride/pad/dilation/groups, plus meaningful mutants, is the largest authoring task
  in this plan — and Path B removes the *reference* cost but **not** the correct-candidate
  cost.
- It also needs the widest `make_inputs` tuple yet: `(N,C_in,C_out,H,W,k,s,p,d,g)`.

**Do not fold conv into a first expansion.**

### 4.5 Effort estimates

**What these are anchored on, so they can be argued with:** the 27 closed forms were
derived, probed at three `K` values, and fitted in **one Colab T4 session**, and they
collapse to 6 families. Spec files run 36–43 lines for template operators
(`swish.py` = 38) and 129–164 for operators with their own adversarial battery and
algebraic properties (`softmax.py` = 129, `flash_attention.py` = 164). Full corpus
sweep ≈ 2.0 GPU-hours at 54 operators. **I have no wall-clock records of authoring
time**, so the session counts below are judgment, not measurement — the ratios
between buckets are better supported than the absolute numbers.

| bucket | ops | authoring | derivation | GPU |
|---|---:|---|---|---|
| A1 activations | 11 | 1 session (all reuse `swish.py`) | ~1 h algebra, 11 one-liners | folds into the Phase-1 batch |
| A2 matmul variants | 17 → ~5 forms | 1 session + **batched spec class** (~half a session) | ~2 h | folds in |
| A3 scans | 4 | ~half a session | **~1 h — shape-only** | folds in |
| A4 losses | 5 | ~1 session | ~2 h (`cross_entropy` template) | folds in |
| **Phase-1 validation** | — | — | — | **1 dedicated GPU session**, mirroring `kccgen`: closed-form vs probe at K ∈ {400, 4k, 20k} + M3 re-fit over the enlarged set |
| **B: conv** | 35 → ~7 forms | **5–8 sessions** — dominated by writing correct Triton convs and mutants | ~4 derivations, ~1 session | its own validation round |
| **L2 (31 clean)** | 31 | changes 2+3+4 first (**~2–3 sessions**), then ~2 sessions of corpus work | **none — empirical only** | ~1 h/sweep |

The asymmetry is the point: **A1+A2+A3+A4 is roughly 4 sessions of authoring plus one
GPU round for 25 operators and ~24 new closed forms. Conv alone is 6–9 sessions for
7 forms.** Operators are not interchangeable in cost.

---

## 5. Recommendation

**Phase 0 — prep, no new operators. ~1 session.**
Refactor `valid_shapes`/`make_inputs` from flat tuples to a config dataclass; add the
batched-matmul spec class. Do this first or every later phase pays the tax.

**Phase 1 — ✅ SHIPPED 2026-08-27, except the GPU validation round.**
**27 operators added: 29 → 56.** Closed-form `L` derived for **all 27**
(`SUPPORTED_OPS` 27 → 54, `STATIC_OPS` 10 → 14). Every form verified against an
autograd-exact Jacobian over 380 invocations, worst relative error **2.98e-08**,
0 regressions on the pre-existing 27. Specs carry 87 adversarial variants and
catch **54/54** torch-level mutants at **0 false positives**.
Artifacts: `verification_runs/phase1_derivations_2026-08-27/`.

**✅ GPU validation round COMPLETE 2026-08-27** (Colab T4, session `kccphase1`).
27 purpose-written Triton kernels (none existed — `TritonBench/reference/` held
only the original 29), all verified against torch before measurement.
**Sandwich 162/162 both sides. Closed-form `L` vs converged probe at K=20000:
1.000–1.023, median 1.012. M3 R² over the full 54-operator corpus: 0.8567.**
Artifacts: `verification_runs/phase1_derivations_2026-08-27/GPU_NATIVE.md`.

**Phase 2 — ✅ COMPLETE 2026-08-27.**
8 operators added (56 → **64**), covering all 35 conv problems in KernelBench L1.
**One closed-form identity covers all eight forms** — `‖J_o‖ = √(F(ones, W²)[o])`
— against the plan's estimated "~4 derivations"; autograd-verified on 19
configs at max rel err 3.8e-16. 36/36 kernel configurations correct (worst rel
err 2.5e-07), **48/48 sandwich**, 35/35 adversarial with zero violations,
16/16 mutants caught at 0 FPs. M3 over 62 operators: **0.8564**, essentially
unchanged from 0.8567. Conv needs **no exception category**.
Artifacts: `verification_runs/phase2_convolution_2026-08-27/FINDINGS.md`.
The plan's estimate that *kernel authoring*, not the maths, is the cost held.

**Phase 3 — L2, empirical-only, 31 clean Gemm-rooted problems.**
Requires changes 2/3/4 first. **No theorem claim** unless the autograd-profile path
of §3.2 is separately built and validated. This is what supports a defensible
"multi-level" statement without inheriting conv or statefulness.

**L3 — future work, not committed.** Reasons in §3.3.

**Also do, ~1 hour:** clone TritonBench-G and KernelBenchX and close §1.3. It cannot
change the phasing — KernelBench L1 alone already exceeds the target — but it can
change *which* 25 operators go into Phase 1, and it is cheap.

### What this deliberately does not promise

- Not "50+ operators across L1/L2/L3." **54 at L1**, with L2 as a later empirical-only
  tier.
- Not theorem coverage at L2. §3.2 change 5 gives the measured reason.
- Not conv in the first phase.
- Not the TritonBench-G / KernelBenchX cross-reference — §1.3 is open and labelled.
