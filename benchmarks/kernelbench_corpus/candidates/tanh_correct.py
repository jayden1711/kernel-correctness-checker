"""Correct CUDA tanh via the CUDA math library's tanhf()."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = tanhf(x[idx]);
    }
}

torch::Tensor tanh_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

tanh_cpp_source = "torch::Tensor tanh_cuda(torch::Tensor x);"

tanh_mod = load_inline(
    name="tanh_correct",
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
