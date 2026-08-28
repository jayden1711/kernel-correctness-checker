# AutoKernel baseline audit

> **⚠ CORRECTED 2026-08-25 — READ §7 BEFORE CITING ANY NUMBER IN THIS FILE.**
> This audit was written against the paper's prose alone. AutoKernel's actual
> implementation (`bench.py`, github.com/RightNow-AI/autokernel) has since been
> read, and it contradicts the audit on three points: the real gate uses a
> **relative tolerance** (fp32 `rtol=1e-4`), its **stage 3 applies five value
> transforms to every operator with tolerances relaxed 10x** rather than three
> probe classes to eleven, and it **sweeps no hyperparameters** (padding,
> stride, kernel_size). §7 records all three and what they do to the reported
> 80% / 0.5% figures.


Audit of `benchmarks/autokernel/files/baselines.py:autokernel_gate` against
the published five-stage harness in **arXiv 2603.21331** — Jaber & Jaber,
*"AutoKernel: Autonomous GPU Kernel Optimization via Iterative Agent-Driven
Search."*

Motivation: `autokernel_gate`'s reported **18% false-positive rate** against
this project's 0% is the single largest margin in `BENCHMARK_RESULTS.md`
§4, and the easiest number for a reviewer to challenge. If the
re-implementation is stricter, buggier, or weaker than the published gate,
that margin is an artifact of our own code rather than a property of the
method.

**It is largely an artifact.** Two of five deviations are bugs that
manufacture false positives on a *correct* reference kernel, and three more
suppress the gate's catch rate.

---

## 1. The published specification

Verbatim from the paper's harness section:

> The benchmark harness (`bench.py`, 1,416 lines) enforces correctness
> through five stages. All must pass before performance is measured.
>
> **Stage 1: Smoke test.** A single forward pass on a small input (e.g.
> 128×128) catches compilation errors, shape mismatches, and gross numerical
> bugs in under 1 second.
>
> **Stage 2: Shape sweep.** The kernel runs across 8 to 10 input
> configurations and three data types. […] This catches size-dependent bugs:
> boundary handling, tile remainder logic, and dtype-specific issues.
>
> **Stage 3: Numerical stability.** Adversarial inputs probe floating-point
> edge cases. For softmax: rows of large identical values. For matmul:
> extreme dynamic range. For normalization: near-zero variance.
>
> **Stage 4: Determinism.** Same input, three runs, bitwise identical
> outputs. Catches race conditions in parallel reductions and
> non-deterministic atomics.
>
> **Stage 5: Edge cases.** Non-power-of-two dimensions (1023, 4097, 1537)
> expose masking bugs and tile remainder errors.
>
> Tolerances are dtype-specific: FP16 uses atol=10⁻², BF16 uses 2×10⁻²,
> FP32 uses 10⁻⁴.

The paper's dtypes are FP16/BF16/FP32; each kernel "has a PyTorch reference
in `reference.py` serving as the correctness oracle." **No relative
tolerance is specified anywhere in the prose** — the tolerance is absolute
only. **This is true of the paper and FALSE of the implementation; see §7.1.**

---

## 2. Threshold-by-threshold verdict

| Published | `baselines.py:autokernel_gate` | Match | Direction of error |
|---|---|---|---|
| FP32 tolerance `atol=1e-4` | `atol=1e-2, rtol=1e-2` | **NO** | 100× **looser** → *understates* catch rate |
| FP16 `atol=1e-2`, BF16 `atol=2e-2` | not implemented (single dtype) | **NO** | — |
| rtol: unspecified (absolute only) | `rtol=1e-2` invented | **NO** | looser again |
| Shape sweep: 8–10 configs | `n_shapes=3` | **NO** | *understates* catch rate |
| Shape sweep: 3 dtypes | 1 dtype (fp32) | **NO** | *understates* catch rate |
| Shape sweep varies **shape** | every `_mk_*` generator in `tritonbench_registry.py` is **fixed-shape**, so the loop re-draws random *values* at one shape | **NO** | not a shape sweep at all |
| Stage 3 probes: softmax / matmul / normalization | correct input *classes*, but wired to 3 literal op names only | **Partial** | silent no-op for 26 of 29 ops |
| Stage 4: **three** runs, bitwise | **two** runs, bitwise (`np.array_equal`) | **NO** | negligible |
| Stage 5: non-power-of-two (1023, 4097, 1537) | **absent** | **NO** | *understates* catch rate |
| Reference oracle = PyTorch reference | ✔ | **YES** | — |

Stage 5 was omitted on the stated grounds that it was "the compile stage,
which doesn't apply to numpy stand-ins" (`baselines.py` docstring). Stage 5
is not a compile stage — it is non-power-of-two edge-case coverage, which
applies directly to this corpus and would raise the gate's catch rate.

---

## 3. The two bugs behind the 18% FPR

`results.md` reports autokernel_gate false positives on exactly two
operators — `layernorm=100%`, `matmul=100%` — both with failing stage
`adversarial_stability`, i.e. stage 3. Every other operator is at 0%.
Both are traceable to `_adversarial_stability_inputs` (`baselines.py:25-39`).

### Bug A — arity (`layernorm`)

`_adversarial_stability_inputs` rebuilds the *whole* argument tuple by
guesswork and returns **1-tuples** for layernorm:

```python
variants.append((np.full(base_shape, 3.0) + rng.normal(size=base_shape) * 1e-8,))
```

But the layernorm reference is `layernorm(x, gamma, beta, eps=1e-5)`
(`TritonBench/reference/layernorm.py:36`), and the corpus routes calls
through `_to_torch_triple`, which does `x, w1, w2 = args`. A 1-tuple raises
`ValueError` there. The bare `except Exception` at `baselines.py:84` scores
that as `return False, "adversarial_stability"` — a gate *failure*. On a
reference-vs-reference trial that is a false positive, deterministically,
100% of the time.

### Bug B — dtype (`matmul`)

`_adversarial_stability_inputs` is the **only** input generator in the repo
that never calls `.astype(np.float32)`. `rng.normal()` returns float64, so
`torch.from_numpy` yields an fp64 CUDA tensor. Triton's `tl.dot` has no fp64
path (`TritonBench/reference/mat_mult.py:22` accumulates in fp32 from fp32
operands), so the matmul reference raises — same bare `except`, same
manufactured false positive, 100% of the time.

### Why softmax is unaffected — the confirming detail

Softmax's variants are correct-arity 1-tuples, and its kernel is
elementwise + reduction with no `tl.dot`, so fp64 executes fine. That is
exactly why `results.md` shows softmax at **0%** FP while layernorm and
matmul sit at 100%. The three observed outcomes are fully explained by
Bugs A and B and nothing else.

### Correction to `BENCHMARK_RESULTS.md` §4

The doc attributes the 18% FPR to *"its fixed-tolerance adversarial-stability
stage […] and its bitwise determinism check (which false-positives on
`frobenius_norm`'s legitimate atomic-add reduction)."* The determinism half
is **not supported by the data**: `frobenius_norm` has 0% FP for
autokernel_gate in `results.md`, and no operator fails at stage 4. That
sentence should be removed. The fixed-tolerance half is also not the
mechanism — no FP was caused by a tolerance comparison; both were
exceptions.

---

## 4. What was done

`benchmarks/autokernel/files/autokernel_faithful.py` implements the gate as
published:

- **Tolerance** — `atol` per dtype (FP32 1e-4, FP16 1e-2, BF16 2e-2).
- **Stage 2** — 8 configurations per operator family × 3 dtypes, with
  genuinely varying shapes (batch-1, tiny, non-power-of-two, largest-in-family).
- **Stage 3** — probes applied by operator *kind* (softmax-like /
  matmul-like / normalization), replacing **only the primary tensor** at its
  exact shape and dtype. This is the fix for both bugs: companion tensors
  and int hyperparameters keep their correct arity and dtype by construction.
- **Stage 4** — three runs, bitwise.
- **Stage 5** — non-power-of-two edge-case shapes.

`baselines.py:autokernel_gate` is left **unchanged**, and `run_benchmark.py`
now runs both, so the correction is a measured delta on identical inputs
rather than an assertion.

### Documented deviations (not accidental)

1. **Absolute shape sizes.** The paper's matmul sweep runs to
   4096×11008×4096, sized for perf-benchmarking a handful of kernels. This
   corpus runs 40 mutants × 6 trials × 8 configs × 3 dtypes per system; at
   the paper's sizes the sweep would dominate wall time by orders of
   magnitude. Shapes here preserve the sweep's *structure and count*, at
   sizes proportionate to this corpus. The fidelity claim is on structure,
   not absolute dimensions.
2. **rtol.** The paper specifies absolute tolerance only. `RTOL = 1e-5`
   (the NumPy/PyTorch default that an `atol=1e-4` implementation would
   inherit) is used. The strict-literal `rtol=0` reading is measurable by
   setting `autokernel_faithful.RTOL = 0.0` — this is load-bearing for FP32
   outputs of magnitude > 10, where the rtol term dominates atol.
3. **Reference-infeasible configs are skipped, not failed.** If the
   *reference* kernel raises on a (shape, dtype) config — e.g. a TritonBench
   kernel with no bf16 path — that is a limitation of this corpus's
   references, not evidence about the candidate. Failing the candidate there
   would re-introduce precisely the artifact class this file removes.
   Configs where the reference succeeds and the *candidate* raises remain
   hard failures. Skips are counted and reported in the system's `detail`.
4. **Stage-5 dims.** 4097 is dropped for families whose cost is superlinear
   in that dimension (matmul, attention, the 4-D/5-D norm and pool
   families); recorded in `detail` rather than hidden.

---

## 5. Expected effect on the headline table — and it cuts against us

| Fix | Effect on autokernel_gate |
|---|---|
| atol 1e-2 → 1e-4 (100× stricter) | catch rate **↑**, FP rate possibly ↑ |
| Real 8-config × 3-dtype shape sweep | catch rate **↑** |
| Stage 5 added | catch rate **↑** |
| Bug A + Bug B fixed | FP rate **↓**, likely toward 0% |

The published-faithful gate should land at a **higher catch rate and a lower
false-positive rate** than the 68% / 18% currently reported. The 18%
figure — and with it the "every SOTA method uses a fixed tolerance and
therefore false-positives" argument as applied to AutoKernel specifically —
should not be cited again until the re-run lands.

Note that this does **not** touch the gpuemu `adversarial_value` 82% FP
result, which is a faithful reproduction of a tradeoff the gpuemu paper
reports itself, nor the `allclose`/`boundary_shape`/`propilot` rows.

---

## 6. Re-run required (blocked on GPU)

Every kernel in the corpus is a real `@triton.jit` kernel; there is no CPU
path. Re-run on a CUDA runtime (Colab):

```bash
cd benchmarks/autokernel/files
python corpus_contract.py my_corpus.py    # validate corpus first
python run_benchmark.py                    # writes results.md + results.json
```

The output will contain both `autokernel_gate` (old) and
`autokernel_gate (faithful)` (corrected) rows. Deliverables that depend on
this run: the corrected §4 table, the corrected §8.2 precision/recall table,
and the FP-mechanism paragraph in §4.

### Validation done so far (no GPU required)

Argument construction was validated against a stubbed `torch` that records
shape/dtype instead of allocating -- a better test than a real CPU run for
this purpose, since the bugs being fixed are arity/dtype bugs, not numerical
ones:

- **0 construction failures** across all 12 operator families x (8 sweep
  shapes + stage-5 edge shapes) x 3 dtypes.
- **Arity correct per family** (single 1, layernorm 3, instancenorm 3,
  rmsnorm 2, matmul 2, attention 3, groupnorm 4, batchnorm 5, cross_entropy
  2, pool1d/2d/3d 4) -- i.e. Bug A cannot recur by construction.
- **No dtype leaks**: every generated tensor carries the requested sweep
  dtype (int64 target vectors excepted) -- i.e. Bug B cannot recur.
- **Stage-3 probe coverage: 11 of 29 operators**, up from 3 of 29 in the old
  gate. The remaining 18 are operators the paper names no probe class for;
  they are reported as uncovered in the system's `detail` rather than
  silently passing.

Reproduce: `python3 /tmp/_stub_test.py` (stub harness; no numpy/torch needed).

**Status: code written and validated for argument construction; the benchmark
numbers themselves are not yet re-measured -- that needs a CUDA runtime.**


---

## 7. CORRECTIONS — the implementation was read, 2026-08-25

Everything in §§1–6 was derived from the paper's harness section. AutoKernel is
open source (`github.com/RightNow-AI/autokernel`) and `bench.py` has now been
read directly. **Source beats prose**, and it overturns three modelling choices.
All three are fixed in `autokernel_faithful.py`.

### 7.1 The real gate has a relative tolerance — fp32 `rtol=1e-4`

`bench.py` pairs every absolute tolerance with an **equal** relative one:

```
float16:  atol=1e-2,  rtol=1e-2
bfloat16: atol=2e-2,  rtol=2e-2
float32:  atol=1e-4,  rtol=1e-4
```

This file previously carried `RTOL = 1e-5`, the NumPy/PyTorch default an
"atol=1e-4" implementation would inherit — a reasonable inference, and wrong.
The real fp32 rtol is **10x looser**.

**Direction of the error: the previously reported 80% catch rate was measured
with a comparator STRICTER than AutoKernel's own.** It is an upper bound on the
real gate, not an estimate of it.

### 7.2 The `(faithful, rtol=0)` system is retired

§4's "deliberate deviation" #2 offered `rtol=0` as the strict-literal reading of
an absolute-only tolerance, and `run_benchmark.py` registered it as a **system**
in the headline table. Since the source shows a real rtol, `rtol=0` corresponds
to **no AutoKernel configuration that exists**. Publishing it beside the real
gate implied a live interpretive question that is in fact settled.

Registration removed from `run_benchmark.py`. **Do not re-add it.** Tolerance
sensitivity, if wanted, is a bound to report — not a baseline.

### 7.3 Stage 3 is broader AND looser than modelled

The paper names three probe classes by example (softmax / matmul /
normalization), so this file applied a probe to **11 of 29** operators and
reported 18 as "uncovered". `bench.py` instead applies **five input-scaling
transforms to every kernel**, keyed to nothing about the operator:

| transform | definition |
|---|---|
| `near_max` | `x * 60000` (fp16) else `x * 1e30` |
| `near_zero` | `x * 1e-6` |
| `mixed_scale` | each element scaled by `1e3` or `1e-3` at random |
| `all_zeros` | `x := 0` |
| `all_same` | `x := 0.5` |

…and **relaxes the tolerance 10x** for them.

So the correction moves stage 3 in **both** directions at once: coverage
**29 of 29** instead of 11 of 29, strictness **10x looser** instead of full.
They do not cancel and there is no argument that settles which dominates.

### 7.4 Does this change the reported 80% / 0.5%? — RE-MEASURED, IT SURVIVES

**Re-run on a Colab T4, 2026-08-25. Full write-up and artifacts:
`verification_runs/autokernel_comparison_2026-08-25/RERUN.md`.**

| reading | catch | FP |
|---|---:|---:|
| pre-correction (superseded) | 80.0% | 0.5% |
| corrected, literal | **80.0%** | **13.0%** |
| corrected + reference-infeasible skip | **80.0%** | **0.5%** |

**Catch rate is 80.0% in all three, with the identical 8 misses.** Widening
stage 3 from 3 probe classes on 11 operators to 5 value transforms on 29 moved
it by exactly zero, because the surviving bugs are conditional on
*hyperparameters* (`padding`) and no value transform reaches a hyperparameter.
The looser fp32 rtol cost nothing either.

The 13% is **one mechanism, 25 of its 26 FPs**: `near_max` scales the primary by
1e30, attention's QK^T overflows fp32 to `inf`, the reference returns non-finite
output without raising, and `_allclose` scores that as a candidate failure on a
reference-vs-reference trial. §4 deviation 3's reference-infeasible rule now
also covers the silent (non-raising) case, which returns FP to 0.5% — a single
`frobenius_norm` determinism flip, the known atomic-add flake.

**Not established:** whether the real `bench.py` skips or fails a non-finite
reference output. Both readings are banked so the choice stays visible.

**So: cite 80.0% / 0.5%, but cite it from the corrected run**, not from the
pre-correction one that happened to land on the same pair for different reasons.

### 7.5 Confirmed from source, and it matters for the comparison

`bench.py` **sweeps no hyperparameters** — "padding, stride, kernel_size are not
swept". Its input space is (shape × dtype) plus the five fixed value transforms.
This is *not* an artifact of this re-implementation: it is how the real gate
works, and it is the mechanism behind the eight mutants the gate misses on this
corpus, all of which are conditional on padding rather than on shape.

Full write-up, with the latency reconciliation and the fairness caveat about
AutoKernel's own kernel-set domain (no pooling operators, no padding parameters
anywhere), is in `verification_runs/autokernel_comparison_2026-08-25/FINDINGS.md`.
