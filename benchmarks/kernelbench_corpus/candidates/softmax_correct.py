"""
Correct CUDA row-wise softmax: one block per row, shared-memory
tree reduction for max then sum, grid-stride loop over columns so it
handles n_cols > BLOCK_SIZE (this problem uses n_cols=512, BLOCK_SIZE=256).
"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void softmax_kernel(const float* x, float* out, int n_rows, int n_cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_x = x + (long)row * n_cols;
    float* row_out = out + (long)row * n_cols;

    __shared__ float sdata[BLOCK_SIZE];

    float local_max = -INFINITY;
    for (int i = tid; i < n_cols; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, row_x[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < n_cols; i += BLOCK_SIZE) {
        local_sum += expf(row_x[i] - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    for (int i = tid; i < n_cols; i += BLOCK_SIZE) {
        row_out[i] = expf(row_x[i] - row_max) / row_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    int n_rows = x.size(0);
    int n_cols = x.size(1);
    auto out = torch::empty_like(x);

    softmax_kernel<<<n_rows, BLOCK_SIZE>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n_rows, n_cols);

    return out;
}
"""

softmax_cpp_source = "torch::Tensor softmax_cuda(torch::Tensor x);"

softmax_mod = load_inline(
    name="softmax_correct",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = softmax_mod

    def forward(self, x):
        return self.softmax.softmax_cuda(x.contiguous())
