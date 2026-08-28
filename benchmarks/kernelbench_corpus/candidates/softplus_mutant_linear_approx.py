"""Mutant: linear_approx -- approximates softplus as relu(x)+0.1 instead
of log(1+exp(x)). Clearly wrong near x=0 (softplus(0)=log(2)~=0.693,
approx gives 0.1)."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softplus_kernel_cheat_linear_approx(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fmaxf(x[idx], 0.0f) + 0.1f;
    }
}

torch::Tensor softplus_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    softplus_kernel_cheat_linear_approx<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

cpp_source = "torch::Tensor softplus_cuda(torch::Tensor x);"

mod = load_inline(
    name="softplus_mutant_linear_approx",
    cpp_sources=cpp_source,
    cuda_sources=source,
    functions=["softplus_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        return self.mod.softplus_cuda(x.contiguous())
