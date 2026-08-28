"""Correct CUDA HardTanh: clamp(x, -1, 1)."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fminf(fmaxf(x[idx], -1.0f), 1.0f);
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    hardtanh_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

cpp_source = "torch::Tensor hardtanh_cuda(torch::Tensor x);"

mod = load_inline(
    name="hardtanh_correct",
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
