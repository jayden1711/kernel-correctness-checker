"""Mutant: wrong_bounds -- clamps to [-0.5, 0.5] instead of [-1, 1]."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel_cheat_wrong_bounds(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fminf(fmaxf(x[idx], -0.5f), 0.5f);
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    hardtanh_kernel_cheat_wrong_bounds<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

cpp_source = "torch::Tensor hardtanh_cuda(torch::Tensor x);"

mod = load_inline(
    name="hardtanh_mutant_wrong_bounds",
    cpp_sources=cpp_source,
    cuda_sources=source,
    functions=["hardtanh_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        return self.mod.hardtanh_cuda(x.contiguous())
