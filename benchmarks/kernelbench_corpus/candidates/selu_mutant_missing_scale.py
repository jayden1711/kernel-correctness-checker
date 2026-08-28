"""Mutant: missing_scale -- computes the ELU part correctly but forgets
to multiply by SELU's outer scale constant (1.0507...)."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define SELU_ALPHA 1.6732632423543772f

__global__ void selu_kernel_cheat_missing_scale(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v = x[idx];
        out[idx] = v > 0.0f ? v : SELU_ALPHA * (expf(v) - 1.0f);
    }
}

torch::Tensor selu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    selu_kernel_cheat_missing_scale<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

cpp_source = "torch::Tensor selu_cuda(torch::Tensor x);"

mod = load_inline(
    name="selu_mutant_missing_scale",
    cpp_sources=cpp_source,
    cuda_sources=source,
    functions=["selu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        return self.mod.selu_cuda(x.contiguous())
