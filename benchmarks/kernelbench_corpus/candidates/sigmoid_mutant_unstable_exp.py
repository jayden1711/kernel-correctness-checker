"""
Mutant: unstable_exp -- mathematically equal to the correct sigmoid
(exp(x)/(1+exp(x)) == 1/(1+exp(-x))) but NOT numerically equivalent in
fp32: exp(x) overflows to Inf once x > ~88, giving Inf/Inf = NaN,
where the stable formula correctly saturates to 1.0. This is the
DELIBERATE gap case for this corpus -- at the base input scale
(+-4ish) both formulas agree, so naive allclose passes; only the
checker's own weight_magnitude adversarial probe (1e4 scale) reaches
the overflow and catches it. Verified directly (not just reasoned
about): base-scale allclose passes, 1e4-scale allclose fails with
NaN present, confirmed with plain PyTorch before writing this CUDA
translation.
"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sigmoid_kernel_cheat_unstable_exp(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float e = expf(x[idx]);
        out[idx] = e / (1.0f + e);
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sigmoid_kernel_cheat_unstable_exp<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

sigmoid_cpp_source = "torch::Tensor sigmoid_cuda(torch::Tensor x);"

sigmoid_mod = load_inline(
    name="sigmoid_mutant_unstable_exp",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_source,
    functions=["sigmoid_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = sigmoid_mod

    def forward(self, x):
        return self.sigmoid.sigmoid_cuda(x.contiguous())
