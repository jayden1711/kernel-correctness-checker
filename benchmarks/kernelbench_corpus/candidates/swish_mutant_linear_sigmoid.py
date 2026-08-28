"""
Mutant: linear_sigmoid_approx (same bug name/spirit as
TritonBench/cheating/swish/linear_sigmoid_approx.py). Replaces true
sigmoid with the "hard sigmoid" piecewise-linear approximation
clamp((x+3)/6, 0, 1) -- close to true sigmoid near x=0, visibly wrong
away from it.
"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_kernel_cheat_linear_sigmoid(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v = x[idx];
        float hard_sigmoid = fminf(fmaxf((v + 3.0f) / 6.0f, 0.0f), 1.0f);
        out[idx] = v * hard_sigmoid;
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_kernel_cheat_linear_sigmoid<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

swish_cpp_source = "torch::Tensor swish_cuda(torch::Tensor x);"

swish = load_inline(
    name="swish_mutant_linear_sigmoid",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=["swish_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.swish = swish

    def forward(self, x):
        return self.swish.swish_cuda(x.contiguous())
