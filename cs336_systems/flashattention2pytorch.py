import torch


def _causal_tile_mask(
    q_len: int,
    k_len: int,
    device: torch.device,
    q_start: int = 0,
    k_start: int = 0,
) -> torch.Tensor:
    q_idx = torch.arange(q_start, q_start + q_len, device=device)[:, None]
    k_idx = torch.arange(k_start, k_start + k_len, device=device)[None, :]
    return q_idx >= k_idx


def _flashattention2_backward_impl(Q, K, V, O, dO, L, is_causal: bool):
    d = Q.shape[-1]
    scale = d ** -0.5

    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if is_causal:
        mask = _causal_tile_mask(Q.shape[-2], K.shape[-2], Q.device)
        S = S.masked_fill(~mask, -1e6)
    P = torch.exp(S - L.unsqueeze(-1))
    D = (dO * O).sum(dim=-1, keepdim=True)

    dV = torch.matmul(P.transpose(-2, -1), dO)
    dP = torch.matmul(dO, V.transpose(-2, -1))
    dS = P * (dP - D)
    dQ = torch.matmul(dS, K) * scale
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale
    return dQ, dK, dV


class FlashAttention2PyTorch(torch.autograd.Function):
    _compiled_backward = None

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Bq, Bk = 32, 32 # >= 16, 16
        *batch, Nq, D = Q.shape
        Nk = K.shape[-2]
        scale = D ** -0.5

        O = torch.empty_like(Q)
        L = torch.empty(*batch, Nq, device=Q.device, dtype=Q.dtype)

        for q_start in range(0, Nq, Bq):
            q_end = min(q_start + Bq, Nq)
            Qi = Q[..., q_start:q_end, :]                                # (..., Bq, D)

            Oi = torch.zeros(*batch, q_end - q_start, D, device=Q.device, dtype=Q.dtype)
            li = torch.zeros(*batch, q_end - q_start, device=Q.device, dtype=Q.dtype)
            mi = torch.full((*batch, q_end - q_start), float("-inf"), device=Q.device, dtype=Q.dtype)

            for k_start in range(0, Nk, Bk):
                k_end = min(k_start + Bk, Nk)
                Kj = K[..., k_start:k_end, :]                            # (..., Bk, D)
                Vj = V[..., k_start:k_end, :]                            # (..., Bk, D)

                Sij = torch.einsum("...id,...jd->...ij", Qi, Kj) * scale # (..., Bq, Bk)
                if is_causal:
                    causal_mask = _causal_tile_mask(
                        q_end - q_start,
                        k_end - k_start,
                        Q.device,
                        q_start=q_start,
                        k_start=k_start,
                    )
                    Sij = Sij.masked_fill(~causal_mask, -1e6)
                m_new = torch.maximum(mi, Sij.max(dim=-1).values)        # (..., Bq)
                P_tilde = torch.exp(Sij - m_new.unsqueeze(-1))           # (..., Bq, Bk)
                alpha = torch.exp(mi - m_new)

                li = alpha * li + P_tilde.sum(dim=-1)                    # (..., Bq)
                Oi = alpha.unsqueeze(-1) * Oi + P_tilde @ Vj             # (..., Bq, D)
                mi = m_new

            O[..., q_start:q_end, :] = Oi / li.unsqueeze(-1)
            L[..., q_start:q_end] = mi + torch.log(li)
        
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        if FlashAttention2PyTorch._compiled_backward is None:
            FlashAttention2PyTorch._compiled_backward = torch.compile(_flashattention2_backward_impl)

        dQ, dK, dV = FlashAttention2PyTorch._compiled_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None
