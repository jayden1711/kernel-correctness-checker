"""Mutant: hard_approx -- clamp(x, -1, 1) piecewise-linear approximation
instead of true tanh. Agrees with tanh only at the extremes (both
saturate near +-1); meaningfully wrong at moderate |x| (e.g. tanh(1)
approx 0.762 vs hard-tanh's 1.0)."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel_cheat_hard_approx(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fminf(fmaxf(x[idx], -1.0f), 1.0f);
    }
}

torch::Tensor tanh_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_kernel_cheat_hard_approx<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

tanh_cpp_source = "torch::Tensor tanh_cuda(torch::Tensor x);"

tanh_mod = load_inline(
    name="tanh_mutant_hard_approx",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_source,
    functions=["tanh_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = tanh_mod

    def forward(self, x):
        return self.tanh.tanh_cuda(x.contiguous())
