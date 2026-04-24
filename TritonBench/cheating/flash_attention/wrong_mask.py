import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel_cheat_wrong_mask(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_vm, stride_vk,
    stride_om, stride_ok,
    N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    #One instance per query tile
    pid_m = tl.program_id(0)

    #Offsets for this query tile
    q_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, D)

    #Load query tile (registers)
    Q_ptrs = Q_ptr + q_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qk
    Q_mask = (q_offsets[:, None] < N) & (d_offsets[None, :] < D)
    Q_block = tl.load(Q_ptrs, mask=Q_mask, other=0.0)

    #Initialize running statistics
    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    #Loop over K/V tiles
    for start in range(0, N, BLOCK_N):
        kv_offsets = start + tl.arange(0, BLOCK_N)

        #Load K tile
        K_ptrs = K_ptr + kv_offsets[:, None] * stride_km + d_offsets[None, :] * stride_kk
        K_mask = (kv_offsets[:, None] < N) & (d_offsets[None, :] < D)
        K_block = tl.load(K_ptrs, mask=K_mask, other=0.0)

        #Load V tile
        V_ptrs = V_ptr + kv_offsets[:, None] * stride_vm + d_offsets[None, :] * stride_vk
        V_mask = (kv_offsets[:, None] < N) & (d_offsets[None, :] < D)
        V_block = tl.load(V_ptrs, mask=V_mask, other=0.0)

        #Attention scores
        S = tl.dot(Q_block, tl.trans(K_block))

        #CHEAT: apply causal mask off by one — masks one extra position per row
        #correct causal mask would be: q_idx >= kv_idx
        #this masks q_idx == kv_idx as well, preventing self-attention
        q_idx = q_offsets[:, None]
        kv_idx = kv_offsets[None, :]
        wrong_causal_mask = q_idx > kv_idx + 1
        S = tl.where(wrong_causal_mask, S, float('-inf'))

        #Update running max
        m_new = tl.maximum(m, tl.max(S, axis=1))

        #Rescale previous accumulator
        acc = acc * tl.exp(m - m_new)[:, None]

        #Add new contribution
        P = tl.exp(S - m_new[:, None])
        acc += tl.dot(P, V_block)

        #Update normalizer
        l = l * tl.exp(m - m_new) + tl.sum(P, axis=1)

        m = m_new

    #Final normalize
    acc = acc / l[:, None]

    #Single HBM write per query tile
    O_ptrs = O_ptr + q_offsets[:, None] * stride_om + d_offsets[None, :] * stride_ok
    O_mask = (q_offsets[:, None] < N) & (d_offsets[None, :] < D)
    tl.store(O_ptrs, acc, mask=O_mask)


def flash_attention(Q, K, V):
    N, D = Q.shape
    O = torch.empty_like(Q)
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(N, BLOCK_M),)
    flash_attention_kernel_cheat_wrong_mask[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D
    )
    return O