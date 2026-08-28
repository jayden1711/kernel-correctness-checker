"""Mutant: leaky -- implements LeakyReLU (slope 0.1) instead of true
ReLU (slope 0)."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel_cheat_leaky(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v = x[idx];
        out[idx] = v > 0.0f ? v : 0.1f * v;
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_kernel_cheat_leaky<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

relu_cpp_source = "torch::Tensor relu_cuda(torch::Tensor x);"

relu_mod = load_inline(
    name="relu_mutant_leaky",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = relu_mod

    def forward(self, x):
        return self.relu.relu_cuda(x.contiguous())
