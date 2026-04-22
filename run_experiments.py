import torch
from TritonBench.experiments.softmax import run_softmax
from TritonBench.experiments.layernorm import run_layernorm
from TritonBench.experiments.matmul import run_matmul
from TritonBench.experiments.flash_attention import run_flash_attention

if __name__ == "__main__":
    print("\nKernel Correctness Verification Benchmark")

    print("\nSoftmax")
    run_softmax()

    print("\nLayer Norm")
    run_layernorm()

    print("\nMatrix Multiplication")
    run_matmul()

    print("\nFlash Attention")
    run_flash_attention()