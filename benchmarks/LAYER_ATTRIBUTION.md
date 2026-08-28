# Per-operator layer attribution

Source: `benchmarks/autokernel/files/results.json` — the three single-layer ablations, each of which runs its whole layer unconditionally. Counts are raw mutants, not percentages.

40 mutants across 29 operators.

| Operator | # mutants | structural | numeric | algebraic | >1 layer | numeric only | uncaught |
|---|---:|---:|---:|---:|---:|---:|---:|
| argmax | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| argmin | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| avg_pool1d | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| avg_pool2d | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| avg_pool3d | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| batchnorm | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| causal_flash_attention | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| cross_entropy | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| flash_attention | 4 | 1 | 4 | 2 | 2 | 2 | 0 |
| frobenius_norm | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| gelu | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| groupnorm | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| instancenorm | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| l1norm | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| l2norm | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| layernorm | 3 | 0 | 3 | 2 | 2 | 1 | 0 |
| log_softmax | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| matmul | 4 | 0 | 4 | 2 | 2 | 2 | 0 |
| max_pool1d | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| max_pool2d | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| max_pool3d | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| max_reduction | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| mean_reduction | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| min_reduction | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| rmsnorm | 3 | 0 | 3 | 3 | 3 | 0 | 0 |
| scaled_dot_product_attention | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| softmax | 2 | 1 | 2 | 2 | 2 | 0 | 0 |
| sum_reduction | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| swish | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| **TOTAL** | **40** | **4** | **40** | **18** | **18** | **22** | **0** |

## Is numeric-layer dominance uniform, or concentrated?

- Operators where the numeric layer catches **every** mutant: **29/29**
- Operators where it catches **none**: **0**

**Uniform, not concentrated.** The numeric layer catches every mutant of every operator, so its 100% aggregate is not carried by a few mutant-heavy operators — it holds operator by operator. That is the strongest form this result could take, and it is the specific claim the advisor's question was asking to verify.

## What each layer contributes independently

| Layer | mutants caught | of which, caught by no other layer |
|---|---:|---:|
| structural | 4 | 0 |
| numeric | 40 | 22 |
| algebraic | 18 | 0 |

**Removing the numeric layer would lose 22 of 40 mutants** — structural and algebraic together catch 18. This is the concrete justification for Layer 2 as a layer, distinct from the per-check ablation in `CHECK_ABLATION.md` which asks which checks *within* it earn their place.

Operators where **only** the numeric layer catches anything: **17/29** — argmax, argmin, avg_pool1d, avg_pool2d, avg_pool3d, batchnorm, cross_entropy, gelu, groupnorm, instancenorm, log_softmax, max_pool1d, max_pool2d, max_pool3d, max_reduction, min_reduction, swish

Operators where structural or algebraic catches a mutant the numeric layer also catches are defence-in-depth, not additional recall — the `>1 layer` column quantifies that overlap, and it is only meaningful because the ablations run unconditionally.

## Layer nesting

On this corpus the catch sets are **strictly nested**: structural (4) subset of algebraic (18) subset of numeric (40).

This is stronger than "numeric alone matches the full checker". It says structural and algebraic contribute **zero** additional recall — there is no mutant either one catches that the numeric layer misses. Their value on this corpus is defence-in-depth and precision of diagnosis (naming *which* invariant broke), not coverage. Stated plainly, that is a more honest framing than a three-layer recall claim, and it is what the per-operator counts above actually support.

The caveat that matters for generalisation: nesting is a property of **this 40-mutant corpus**, not a theorem. A structural-only bug (a kernel that never launches, a tile left unwritten) would be caught by Layer 1 and could well slip a numeric comparison that happens to agree; no such mutant exists here. The corpus, not the checker, is what makes the layers nest — worth saying before a reviewer says it.

## Cross-reference: causal_flash_attention

`causal_flash_attention/wrong_causal_mask` is caught here by **3 of 3 layers** (structural, numeric, algebraic) on a plain random input.

The adversarial search, by contrast, recorded `hit_mutants: []` on all 120 proposals (`adversarial_results/causal_flash_attention_search_result.json`).

**That is not the contradiction it first appears to be, and an earlier reading of it here was wrong.** A mutant lands in `missed_mutants` when it passed the checker **or** when it failed the checker but also failed naive allclose (`coordinator.py:_evaluate_verdict`) — the second case is a *caught* mutant with no allclose gap to report. Since allclose catches this mutant on ordinary inputs (0% missed for this operator in `results.md`), the likely reading is that the checker did catch it and the search correctly found no gap. The verdict record persists nothing per-mutant, so which of the two occurred cannot be recovered from the stored run.

Root-caused separately in `adversarial_results/CFA_NONHIT_ROOTCAUSE.md`: 47% of that run's proposals were structurally invalid because `causal_flash_attention` has no entry in the search prompt's `OPERATOR_CONTEXT`.
