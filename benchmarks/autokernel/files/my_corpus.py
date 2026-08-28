"""
Real 29-operator / 41-mutant corpus, built from TritonBench/reference and
TritonBench/cheating (see tritonbench_registry.py for the full mapping and
the one excluded duplicate).

REQUIRES A REAL GPU: every ref_fn/mutant_fn here launches an actual
`@triton.jit` kernel. `pip install torch triton` and run on a CUDA runtime
(e.g. Colab) -- there is no CPU path for these. This machine (no GPU, no
triton installed) cannot execute any of it; it was written and statically
checked against the TritonBench/verification source, not runtime-tested.

    python corpus_contract.py my_corpus.py
"""

from tritonbench_registry import build_corpus

CORPUS = build_corpus()
