# kernel-correctness-checker

A lightweight correctness verification framework for LLM-generated Triton kernels, targeting the **cheating kernel problem**: kernels that pass `torch.allclose` on random inputs while computing something semantically different.

## Motivation

[AccelOpt](https://arxiv.org/pdf/2511.12638) documents LLMs generating kernels that omit necessary work while staying within numerical tolerances. [robust-kbench](https://arxiv.org/abs/2509.14279) corroborates this empirically — apparent speedups collapse from 3.13x to 1.49x after filtering cheating kernels. The root cause in both cases: random input testing with fixed tolerances is too weak a correctness criterion.

## What This Does

Implements a four-layer detection pipeline against two concrete cheating softmax kernels, both of which pass `torch.allclose` on random inputs:

| Layer | Approach | Catches |
|---|---|---|
| Structural | triton-viz access pattern tracing | Memory access bugs |
| Algebraic | Row-sum and shift invariance checks | Normalization errors |
| Adversarial Oracle | Targeted inputs exposing known failure modes | Distribution-dependent cheats |
| Numeric | `torch.allclose` baseline | Obvious errors only |

## Key Findings

- **Wrong reduction cheat** (sums over 2040/2048 columns) passes `torch.allclose`, fails algebraic property checks and adversarial oracle
- **First-tile cheat** passes all layers (shift invariance makes it mathematically equivalent to the reference, demonstrating that certain cheat classes require stronger specification to catch)
- Structural checks via [triton-viz](https://github.com/Deep-Learning-Profiling-Tools/triton-viz) are insufficient for computational bugs where all memory locations are visited but the wrong values are computed
- Adversarial inputs — motivated by [TTrace](https://arxiv.org/pdf/2506.09280)'s perturbation-based tolerance estimation and [AutoKernel](https://arxiv.org/abs/2603.21331)'s adversarial input harness — catch distribution-dependent cheats that random testing misses

## Structure

```
TritonBench/
    kernels/          # cheating kernel variants
    reference/        # trusted reference implementation
checks/
    properties.py     # algebraic property tests
    structural.py     # triton-viz access pattern checks
    oracle.py         # adversarial input generation
run_experiments.py    # reproduces all findings
```

## Related Work

- [AccelOpt](https://arxiv.org/pdf/2511.12638) — motivating paper, documents the cheating kernel problem
- [robust-kbench](https://arxiv.org/abs/2509.14279) — empirical evidence that cheating is widespread
- [TTrace](https://arxiv.org/pdf/2506.09280) — perturbation-based tolerance estimation
- [triton-viz](https://github.com/Deep-Learning-Profiling-Tools/triton-viz) — runtime instrumentation backend
- [Volta](https://arxiv.org/pdf/2511.12638) — PTX-level formal equivalence checking, long-term direction
- [AutoKernel](https://arxiv.org/abs/2603.21331) — adversarial input harness design
