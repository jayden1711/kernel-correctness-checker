# There is no 29x latency gap against AutoKernel — that figure is `allclose`

**Analysed 2026-08-25.** Latency and verdict data from
`verification_runs/triton_cache_2026-08-25/run2/` (Colab T4, warm Triton cache,
unmodified `run_benchmark.py`, 40 mutants + 200 reference trials per system).
AutoKernel characterised from its **own source** —
`github.com/RightNow-AI/autokernel`, `bench.py` and `reference.py`, read
2026-08-25 — not from the paper's prose alone. Paper: arXiv 2603.21331.

---

## The verdict

**`RESULTS_SUMMARY.md` §2's 29x belongs to `allclose`, not to AutoKernel.**
Re-measured here at **27.3x at p50**. Against AutoKernel there is no gap in
either direction worth calling a gap, and on the mean the sign is reversed.

| pass-path, 200 reference trials | p50 | p90 | mean |
|---|---:|---:|---:|
| `your_checker (full)` | 15.20 ms | 38.80 ms | 22.38 ms |
| `autokernel_gate (faithful)` | 12.38 ms | **327.99 ms** | **69.05 ms** |
| `allclose` | 0.56 ms | 0.68 ms | 0.56 ms |

- At **p50**, AutoKernel is **1.23x faster**.
- At the **mean**, AutoKernel is **3.09x slower**.
- At **p90**, AutoKernel is **8.5x slower**.

Reference trials are the apples-to-apples slice: both systems run their full
pipeline on a correct kernel with no short-circuit. Mutant trials are not
comparable — each system short-circuits at a different stage.

**This is not "lighter verification bought speed".** AutoKernel does *more* raw
work than the checker and catches less: ~64 kernel calls per candidate, 3.09x
the mean latency, and **53.6% of the whole corpus run's 186.3s of first-touch
Triton compile** (against the checker's 19.3%) — because its 24 (shape × dtype)
configs demand roughly 3x the distinct specializations.

---

## What AutoKernel's gate actually does, per candidate

Read from `bench.py`, not inferred:

| stage | work | calls |
|---|---|---:|
| 1 smoke | 1 config, smallest size, first dtype | 2 |
| 2 shape sweep | **8 sizes × 3 dtypes = 24 configs** | **48** |
| 3 numerical stability | 5 fixed value transforms, **tolerance relaxed 10x** | 10 |
| 4 determinism | 3 runs, seed 42, `torch.equal` | 3 |
| 5 edge cases | ~3 non-power-of-two shapes, first dtype only | 6 |
| | | **~64** |

- Tolerances **fp32 `atol=1e-4, rtol=1e-4`**; fp16 `1e-2/1e-2`; bf16 `2e-2/2e-2`.
- **In-process. No subprocess per candidate** — timeouts via SIGALRM (Unix) or
  a thread (Windows). *Less* isolation than this project's spawn-per-candidate
  executor.
- **No adversarial search.** Fixed input set, fixed seeds, no retry or search
  loop. Nothing analogous to `verification/adversarial_search/`.
- **No hyperparameter variation** — padding, stride and kernel_size are not
  swept. Quoted from source.
- Real sizes run to 4096×11008×4096 (matmul) and 4096×50257 (softmax). The
  re-implementation in this repo deliberately scales these down, so the 69 ms
  measured here **understates** the real gate's cost.

---

## The reconciliation — it is one story, and it is about AXIS, not effort

AutoKernel misses exactly 8 of 40 mutants, and they are one family:

```
avg_pool{1,2,3}d/wrong_divisor      max_pool{1,2,3}d/wrong_padding
max_reduction/wrong_padding         min_reduction/wrong_padding
```

Every one is conditional on **a hyperparameter or a value distribution**, not on
shape or dtype:

- `avg_pool`'s reference uses `count_include_pad=True`. At `padding=0` the wrong
  divisor and the right divisor are **numerically identical** — no shape and no
  dtype can expose it.
- `wrong_padding` in a max reduction pads with `0` instead of `-inf`. With
  mixed-sign `randn` the true max is positive, so the bug is invisible.

The checker catches all 8 through `cross_shape`, `weight_magnitude`, and
one-line spec overrides — e.g. `verification/specs/avg_pool1d.py:30`:

```python
def get_adversarial_inputs(self, inputs):
    x, kernel_size, stride, padding = inputs
    return [("padded", (x, kernel_size, stride, max(padding, 1)))]
```

So cost and catch rate are **the same decision seen twice**:

| | input space |
|---|---|
| AutoKernel | shape × dtype, plus 5 fixed value transforms |
| this checker | shape × value-distribution × **hyperparameter** |

75% of AutoKernel's calls (48 of ~64) go to the shape sweep. That breadth is
what makes it the corpus's largest compile driver **and** what leaves the
residual bugs unreachable — they lie outside its input space entirely, so no
number of additional shapes or dtypes would find them.

**The honest framing is therefore neither "rigor-for-latency" nor "comparable
work, faster".** It is: the two systems spend comparable budgets on *different
axes*, and on this corpus the hyperparameter axis is where the surviving bugs
are.

---

## Fairness caveat — AutoKernel's kernel-set domain

**AutoKernel's own `reference.py` contains no pooling operators and no padding
parameters anywhere.** Its 9–10 kernels are transformer ops: matmul, softmax,
layernorm, rmsnorm, flash_attention, fused_mlp, cross_entropy,
rotary_embedding, reduce_sum, reduce_max.

The 8 mutants its gate misses are therefore **operators AutoKernel never
targets**. Two things follow, and both should travel with any citation:

1. The *specific* 80% figure is a property of applying AutoKernel's gate to
   **this** corpus, which is broader than its design domain.
2. The *mechanism* — an input space of shape × dtype with no hyperparameter
   sweep — is confirmed from source and **does** generalise. A padding-
   conditional bug in an operator AutoKernel did target would be equally
   invisible to it.

Do not compress this into "AutoKernel misses 20% of bugs". It does not verify
these operators.

---

## Where each system's time goes

**Checker**, pass-path, from the ablation systems on identical inputs:

| layer | ms | share |
|---|---:|---:|
| structural only | 4.42 | 19% |
| algebraic only | 3.29 | 14% |
| **numeric only** | **19.07** | **80%** |
| full (short-circuits between layers) | 23.72 | — |

Within the numeric layer the driver is the per-spec adversarial fan-out — each
`adversarial_*` variant runs its own 20-sample perturbation:

| adversarial variants | operators | mean ms |
|---:|---:|---:|
| 1 | 17 | 14.57 |
| 2 | 7 | 14.41 |
| 5 | 8 | 26.52 |
| 6 | 8 | 48.48 |

r = 0.798, **~5.67 ms per additional variant**.

**AutoKernel**: 75% of calls are stage 2, and the mean is not the typical case —
it is **faster on 22 of 29 operators** and far slower on five. See the tail
section below.

---

## Tail structure — the two systems fail differently

| | p90/p50 | mean/p50 | top-1 op | top-3 ops |
|---|---:|---:|---:|---:|
| `your_checker (full)` | **3.03** | **1.47** | 25.1% | 50.0% |
| `autokernel_gate (faithful)` | **27.75** | **5.78** | **54.0%** | **79.5%** |

AutoKernel's mean is an artifact of three operators: `flash_attention` (384 ms),
`causal_flash_attention` (365 ms) and `scaled_dot_product_attention` (360 ms)
are **79.5%** of its total time. Attention is O(N²) and its sweep runs to
(1023, 32).

**The checker has no equivalent pathology.** Its most expensive operator,
`flash_attention` at 59.6 ms, is 2.5x its own mean and *uniformly* expensive
(p50 51.75, p90 59.90 — ratio 1.16), which is a cost profile, not a blowup. No
operator in the checker has a within-operator p90/p50 above ~1.5.

Full per-operator table: `checker_tail.md` in this directory.

---

## Corrections this analysis forced

Three modelling errors in `benchmarks/autokernel/AUTOKERNEL_BASELINE_AUDIT.md`
and `autokernel_faithful.py`, all from having only the paper's prose. Fixed
2026-08-25; see that file's **§7**.

1. **fp32 `rtol` is 1e-4, not 1e-5.** The real gate pairs every atol with an
   equal rtol. The old comparator was 10x stricter than AutoKernel's own, so the
   previously reported 80% was an **upper bound**.
2. **`(faithful, rtol=0)` retired.** It corresponded to no real configuration.
3. **Stage 3 remodelled** — 5 value transforms on all 29 operators with
   tolerance relaxed 10x, replacing 3 probe classes on 11 of 29.

**Re-measured after the corrections — see `RERUN.md`. The headline survives:
catch is 80.0% with the identical 8 misses, and FP returns to 0.5% under the
file's own reference-infeasible rule.** The stage-3 widening (3 probe classes on
11 operators → 5 value transforms on 29) moved the catch rate by exactly zero,
which is the axis argument above confirming itself: the surviving bugs are
hyperparameter-conditional, and no value transform can reach a hyperparameter.
