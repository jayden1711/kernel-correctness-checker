"""Correct CUDA LeakyReLU: x if x>0 else 0.01*x (default slope, matching
this problem's get_init_inputs()=[] -> Model()'s own default)."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* x, float* out, int size, float slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v = x[idx];
        out[idx] = v > 0.0f ? v : slope * v;
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor x, double slope) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    leaky_relu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size, (float)slope);
    return out;
}
"""

cpp_source = "torch::Tensor leaky_relu_cuda(torch::Tensor x, double slope);"

mod = load_inline(
    name="leaky_relu_correct",
    cpp_sources=cpp_source,
    cuda_sources=source,
    functions=["leaky_relu_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.mod = mod
        self.slope = negative_slope

    def forward(self, x):
        return self.mod.leaky_relu_cuda(x.contiguous(), self.slope)
