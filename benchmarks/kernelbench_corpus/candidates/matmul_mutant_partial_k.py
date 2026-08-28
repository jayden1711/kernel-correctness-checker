"""
Mutant: partial_k_reduct (same bug name/spirit as
TritonBench/cheating/matmult/partial_k_reduct.py). Only accumulates
the first K/2 terms of the dot product instead of all K -- e.g. for
all-ones A/B, correct output is all-K, buggy output is all-K/2.
"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel_cheat_partial_k(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        int half_K = K / 2;
        for (int k = 0; k < half_K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel_cheat_partial_k<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul_mod = load_inline(
    name="matmul_mutant_partial_k",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul_mod

    def forward(self, A, B):
        return self.matmul.matmul_cuda(A.contiguous(), B.contiguous())
