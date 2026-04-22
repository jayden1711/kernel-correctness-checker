# kernel-correctness-checker

Correctness verification framework for LLM-generated Triton kernels. Targets the cheating kernel problem: kernels that pass `torch.allclose` on random inputs while computing something semantically different.

## Detection Pipeline

| Layer | Approach | Catches |
|---|---|---|
| Structural | triton-viz access pattern tracing | Memory access bugs |
| Algebraic | Row-sum and shift invariance checks | Normalization errors |
| Adversarial Oracle | Targeted inputs exposing known failure modes | Distribution-dependent cheats |
| Numeric | `torch.allclose` baseline | Obvious errors only |

## Setup

```bash
git clone https://github.com/jayden1711/kernel-correctness-checker
cd kernel-correctness-checker
pip install -e .
pip install triton triton-viz
```

## Usage

```bash
python run_experiments.py
```

Or run a single kernel benchmark:

```bash
python -m TritonBench.experiments.softmax
```

## Key Findings

No single detection layer is sufficient. Different cheat classes evade different layers. Catching them requires the full pipeline.

## Related Work

- [AccelOpt](https://arxiv.org/pdf/2511.12638) — motivating paper
- [robust-kbench](https://arxiv.org/abs/2509.14279) — empirical evidence cheating is widespread
- [TTrace](https://arxiv.org/pdf/2506.09280) — perturbation-based tolerance estimation
- [triton-viz](https://github.com/Deep-Learning-Profiling-Tools/triton-viz) — runtime instrumentation backend
- [Volta](https://arxiv.org/pdf/2311.02737) — PTX-level formal equivalence checking, long-term direction
- [AutoKernel](https://arxiv.org/abs/2603.21331) — adversarial input harness design