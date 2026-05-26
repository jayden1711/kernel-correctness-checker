import torch
from TritonBench.experiments.softmax import run as run_softmax
from TritonBench.experiments.layernorm import run as run_layernorm
from TritonBench.experiments.matmul import run as run_matmul
from TritonBench.experiments.flash_attention import run as run_flash_attention

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