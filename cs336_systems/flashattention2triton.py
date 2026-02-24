import torch
import triton
import triton.language as tl

from cs336_systems.flashattention2pytorch import _flashattention2_backward_impl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    q_start = query_tile_index * Q_TILE_SIZE
    q_idx = q_start + tl.arange(0, Q_TILE_SIZE)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    Qi = tl.load(Q_block_ptr).to(tl.float32)  # (Q_TILE_SIZE, D)

    mi = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)  # (Q_TILE_SIZE,)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # (Q_TILE_SIZE,)
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # (Q_TILE_SIZE, D)

    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        Kj = tl.load(K_block_ptr).to(tl.float32)  # (K_TILE_SIZE, D)
        Vj = tl.load(V_block_ptr)  # (K_TILE_SIZE, D)

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)

        if is_causal:
            k_idx = k_start + tl.arange(0, K_TILE_SIZE)
            mask = q_idx[:, None] >= k_idx[None, :]
            Sij = tl.where(mask, Sij, -1e6)

        m_new = tl.maximum(mi, tl.max(Sij, axis=1))  # (Q_TILE_SIZE,)
        P_tilde = tl.exp(Sij - m_new[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        alpha = tl.exp(mi - m_new)  # (Q_TILE_SIZE,)

        li = alpha * li + tl.sum(P_tilde, axis=1)  # (Q_TILE_SIZE,)
        Oi = alpha[:, None] * Oi + tl.dot(P_tilde.to(Vj.dtype), Vj)  # (Q_TILE_SIZE, D)
        mi = m_new

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    Oi = Oi / li[:, None]
    Li = mi + tl.log(li)

    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty))
    
    l_ptrs = L_ptr + batch_index * stride_lb + q_idx * stride_lq
    tl.store(l_ptrs, Li)



class FlashAttention2Triton(torch.autograd.Function):
    _compiled_backward = None

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, NQ, D = Q.shape
        NK = K.shape[1]
        scale = D ** -0.5

        O = torch.empty_like(Q)
        L = torch.empty((B, NQ), device=Q.device, dtype=torch.float32)

        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32
        grid = (triton.cdiv(NQ, Q_TILE_SIZE), B)

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            NQ, NK, scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        if FlashAttention2Triton._compiled_backward is None:
            FlashAttention2Triton._compiled_backward = torch.compile(_flashattention2_backward_impl)

        dQ, dK, dV = FlashAttention2Triton._compiled_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None
