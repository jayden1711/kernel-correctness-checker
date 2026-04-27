import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)

        #Bug: A_ptr is indexed with B's strides - reads from B instead of A
        A_ptrs = A_ptr + row_offsets[:, None] * stride_bk + k_offsets[None, :] * stride_bn
        A_mask = (row_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_block = tl.load(A_ptrs, mask=A_mask, other=0.0)

        #Bug: B_ptr is indexed with A's strides - reads from A instead of B
        B_ptrs = B_ptr + k_offsets[:, None] * stride_am + col_offsets[None, :] * stride_ak
        B_mask = (k_offsets[:, None] < K) & (col_offsets[None, :] < N)
        B_block = tl.load(B_ptrs, mask=B_mask, other=0.0)

        acc += tl.dot(A_block, B_block)

    C_ptrs = C_ptr + row_offsets[:, None] * stride_cm + col_offsets[None, :] * stride_cn
    C_mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


def matmul(A, B):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C