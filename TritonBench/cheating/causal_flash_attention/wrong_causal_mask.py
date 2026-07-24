import torch
import triton
import triton.language as tl


@triton.jit
def causal_flash_attention_kernel_cheat_wrong_boundary(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_vm, stride_vk,
    stride_om, stride_ok,
    N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    pid_m = tl.program_id(0)

    q_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, D)

    Q_ptrs = Q_ptr + q_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qk
    Q_mask = (q_offsets[:, None] < N) & (d_offsets[None, :] < D)
    Q_block = tl.load(Q_ptrs, mask=Q_mask, other=0.0)

    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for start in range(0, N, BLOCK_N):
        kv_offsets = start + tl.arange(0, BLOCK_N)

        K_ptrs = K_ptr + kv_offsets[:, None] * stride_km + d_offsets[None, :] * stride_kk
        K_mask = (kv_offsets[:, None] < N) & (d_offsets[None, :] < D)
        K_block = tl.load(K_ptrs, mask=K_mask, other=0.0)

        V_ptrs = V_ptr + kv_offsets[:, None] * stride_vm + d_offsets[None, :] * stride_vk
        V_mask = (kv_offsets[:, None] < N) & (d_offsets[None, :] < D)
        V_block = tl.load(V_ptrs, mask=V_mask, other=0.0)

        S = tl.dot(Q_block, tl.trans(K_block)) * (1.0 / (D ** 0.5))

# Bug: off-by-one causal boundary -- excludes self-attention (q_idx ==
# kv_idx) in addition to future positions, using q_idx > kv_idx + 1
# instead of the correct q_idx >= kv_idx. Adapted from the existing
# flash_attention wrong_mask mutant's exact boundary shift. Invisible
# for any query tile that only ever looks at strictly-earlier key
# tiles (far from the diagonal); wrong only in tiles that straddle the
# causal boundary itself -- exactly the region a non-adversarial random
# test is unlikely to isolate.
        q_idx = q_offsets[:, None]
        kv_idx = kv_offsets[None, :]
        wrong_causal_mask = q_idx > kv_idx + 1
        S = tl.where(wrong_causal_mask, S, float('-inf'))

        m_new = tl.maximum(m, tl.max(S, axis=1))
        acc = acc * tl.exp(m - m_new)[:, None]
        P = tl.exp(S - m_new[:, None])
        acc += tl.dot(P, V_block)
        l = l * tl.exp(m - m_new) + tl.sum(P, axis=1)
        m = m_new

    acc = acc / l[:, None]

    O_ptrs = O_ptr + q_offsets[:, None] * stride_om + d_offsets[None, :] * stride_ok
    O_mask = (q_offsets[:, None] < N) & (d_offsets[None, :] < D)
    tl.store(O_ptrs, acc, mask=O_mask)


def causal_flash_attention(Q, K, V):
    N, D = Q.shape
    O = torch.empty_like(Q)
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(N, BLOCK_M),)
    causal_flash_attention_kernel_cheat_wrong_boundary[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D
    )
    return O
