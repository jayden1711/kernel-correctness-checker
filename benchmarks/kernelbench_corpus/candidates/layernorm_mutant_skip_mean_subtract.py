"""Mutant: skip_mean_subtract (same bug name/spirit as
TritonBench/cheating/layer_norm/skip_mean_subtract.py). Divides by std
without subtracting the mean first."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void layernorm_kernel_cheat_skip_mean_subtract(const float* x, float* out, int n_rows, int n_cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_x = x + (long)row * n_cols;
    float* row_out = out + (long)row * n_cols;

    __shared__ float sdata[BLOCK_SIZE];

    float local_sum = 0.0f;
    for (int i = tid; i < n_cols; i += BLOCK_SIZE) {
        local_sum += row_x[i];
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / n_cols;
    __syncthreads();

    float local_var = 0.0f;
    for (int i = tid; i < n_cols; i += BLOCK_SIZE) {
        float d = row_x[i] - mean;
        local_var += d * d;
    }
    sdata[tid] = local_var;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float var = sdata[0] / n_cols;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    // BUG: divides by std without subtracting mean first.
    for (int i = tid; i < n_cols; i += BLOCK_SIZE) {
        row_out[i] = row_x[i] * inv_std;
    }
}

torch::Tensor layernorm_cuda(torch::Tensor x, double eps) {
    int n_rows = x.size(0);
    int n_cols = x.size(1);
    auto out = torch::empty_like(x);

    layernorm_kernel_cheat_skip_mean_subtract<<<n_rows, BLOCK_SIZE>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n_rows, n_cols, (float)eps);

    return out;
}
"""

layernorm_cpp_source = "torch::Tensor layernorm_cuda(torch::Tensor x, double eps);"

layernorm_mod = load_inline(
    name="layernorm_mutant_skip_mean_subtract",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.layernorm = layernorm_mod
        self.eps = 1e-5

    def forward(self, x):
        return self.layernorm.layernorm_cuda(x.contiguous(), self.eps)
