"""Correct CUDA ELU: x if x>0 else alpha*(exp(x)-1)."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elu_kernel(const float* x, float* out, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v = x[idx];
        out[idx] = v > 0.0f ? v : alpha * (expf(v) - 1.0f);
    }
}

torch::Tensor elu_cuda(torch::Tensor x, double alpha) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size, (float)alpha);
    return out;
}
"""

cpp_source = "torch::Tensor elu_cuda(torch::Tensor x, double alpha);"

mod = load_inline(
    name="elu_correct",
    cpp_sources=cpp_source,
    cuda_sources=source,
    functions=["elu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.mod = mod
        self.alpha = alpha

    def forward(self, x):
        return self.mod.elu_cuda(x.contiguous(), self.alpha)
