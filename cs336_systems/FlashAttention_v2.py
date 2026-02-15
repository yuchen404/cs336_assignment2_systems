import torch
from einops import rearrange, einsum
import triton
import triton.language as tl

class FlashAttention2Forward_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        """
        Pure PyTorch implementation of FlashAttention-2 forward pass.
        Q, K, V: (..., seq_len, d_model)

        returns: 
        O: (..., seq_len, d_model)

        Saved tensors for backward:
        Q, K, V, O, 
        L: (..., seq_len) logsumexp for each query position, L_i = logsumexp_j(Q_i . K_j^T)
        """
        assert Q.shape == K.shape == V.shape, "Q, K, V must have the same shape"

        seq_len, d_model = Q.shape[-2], Q.shape[-1]
        
        device = Q.device
        dtypef32 = torch.float32

        # 
        *prefix, _, _ = Q.shape
        Bh = 1
        for x in prefix: Bh *= x 

        Q = rearrange(Q, "... s d -> (...) s d").contiguous()
        K = rearrange(K, "... s d -> (...) s d").contiguous()
        V = rearrange(V, "... s d -> (...) s d").contiguous()

        Bq = 32  # Block size for queries, > 16
        Bk = 32  # Block size for keys/values, > 16

        Qf = Q.to(torch.float32)    # float32 for better precision
        Kf = K.to(torch.float32)    # float32 for better precision
        Vf = V.to(torch.float32)

        scale = 1.0 / (d_model ** 0.5)

        # Output tensor
        Of = torch.empty((Bh, seq_len, d_model), device=device, dtype=dtypef32)
        L = torch.empty((Bh, seq_len), device=device, dtype=dtypef32)

        # --- block-wise online softmax attention ---
        for Q_i_start in range(0, seq_len, Bq):
            Q_i = Qf[:, Q_i_start: Q_i_start + Bq, :]   # (Bh, Bq, d_model)

            # Initialize max and l for the current query block
            m_i = torch.full((Bh, Bq), float('-inf'), device=device, dtype=dtypef32) # (Bh, Bq)
            l_i = torch.zeros((Bh, Bq), device=device, dtype=dtypef32)  # (Bh, Bq)

            # Initialize output for the current query block
            Of_i = torch.zeros((Bh, Bq, d_model), device=device, dtype=dtypef32)  # (Bh, Bq, d_model)
            for K_j_start in range(0, seq_len, Bk):
                K_j = Kf[:, K_j_start: K_j_start + Bk, :]   # (Bh, Bk, d_model)
                V_j = Vf[:, K_j_start: K_j_start + Bk, :]   # (Bh, Bk, d_model)

                scores = einsum(Q_i, K_j, "b q d, b k d -> b q k") * scale  # (Bh, Bq, Bk)
                if is_causal:
                    # Create causal mask
                    q_indices = torch.arange(Q_i_start, min(Q_i_start + Bq, seq_len), device=device).unsqueeze(1)  # (Bq, 1)
                    k_indices = torch.arange(K_j_start, min(K_j_start + Bk, seq_len), device=device).unsqueeze(0)  # (1, Bk)
                    # causal_mask = (q_indices >= k_indices).to(dtypef32)  # (Bq, Bk)
                    # scores = scores * causal_mask.unsqueeze(0) + (1.0 - causal_mask).unsqueeze(0) * float('-inf')
                    causal_mask = (q_indices >= k_indices)[None, :, :]  # (Bq, Bk)
                    scores = scores.masked_fill(~causal_mask, float("-inf"))

                # per row max for the new block
                m_i_blk = torch.max(scores, dim=-1).values  # (Bh, Bq)
                # update new max m_i
                m_i_new = torch.max(m_i, m_i_blk)  # (Bh, Bq)

                P_ij = torch.exp(scores - m_i_new.unsqueeze(-1))   # stable softmax within block (Bh, Bq, Bk)

                l_ij = torch.sum(P_ij, dim=-1) + torch.exp(m_i - m_i_new) * l_i  # (Bh, Bq) update l_i

                Of_blk = einsum(P_ij, V_j, "b q k, b k d -> b q d")  # (Bh, Bq, d_model)
                Of_i_new = Of_i * torch.exp(m_i - m_i_new).unsqueeze(-1) + Of_blk  # (Bh, Bq, d_model)

                # Update for next block
                m_i = m_i_new
                l_i = l_ij
                Of_i = Of_i_new

            # write back output for the query block and logsumexp
            Of_i = Of_i / l_i.unsqueeze(-1) # normalize output
            Of[:, Q_i_start: Q_i_start + Bq, :] = Of_i
            L[:, Q_i_start: Q_i_start + Bq] = torch.log(l_i) + m_i

        # reshape back to original prefix shape
        O = Of.view(*prefix, seq_len, d_model).to(Q.dtype)
        L = L.view(*prefix, seq_len).to(dtypef32)

        # save for backward
        Q = Q.view(*prefix, seq_len, d_model)
        K = K.view(*prefix, seq_len, d_model)
        V = V.view(*prefix, seq_len, d_model)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O
    
    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass for FlashAttention-2 using PyTorch operations.
        dO: (..., seq_len, d_model) gradient of output O
        Q, K, V, O are retrieved from ctx (..., seq_len, d_model)
        L: (..., seq_len) logsumexp for each query position, L_i = logsumexp_j(Q_i . K_j^T)
        scale: float 1/sqrt(d_model)
        is_causal: bool
        returns:
        dQ, dK, dV: gradients w.r.t. Q, K, V
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        in_dtype = Q.dtype

        Qf = Q.to(torch.float32)
        Kf = K.to(torch.float32)
        Vf = V.to(torch.float32)
        Of = O.to(torch.float32)
        dOf = dO.to(torch.float32)

        if is_causal:
            flash_bwd_complied = _flash_bwd_causal(Qf, Kf, Vf, Of, L, dOf)
        else:
            flash_bwd_complied = _flash_bwd_nocausal(Qf, Kf, Vf, Of, L, dOf)

        dQ, dK, dV = flash_bwd_complied

        return dQ, dK, dV, None
        # raise NotImplementedError("FlashAttention2Forward_pytorch backward is not implemented yet.")

_flash_bwd_causal = torch.compile(lambda Q, K, V, O, L, dO: _flash_attention_bwd(Q, K, V, O, L, dO, True))
_flash_bwd_nocausal = torch.compile(lambda Q, K, V, O, L, dO: _flash_attention_bwd(Q, K, V, O, L, dO, False))
    
def _flash_attention_bwd(Q, K, V, O, L, dO, is_causal: bool):
    """
    Backward pass for FlashAttention-2 using Triton kernels.
    Q, K, V, O: (..., seq_len, d_model)
    L: (..., seq_len) logsumexp for each query position, L_i = logsumexp_j(Q_i . K_j^T)
    dO: (..., seq_len, d_model) gradient of output O
    is_causal: bool
    returns:
    dQ, dK, dV: gradients w.r.t. Q, K, V
    """
    *prefix, seq_len, d_model = Q.shape

    scale = 1.0 / (d_model ** 0.5)

    device = Q.device
    dtype = torch.float32

    # Recompute attention scores
    scores = einsum(Q, K, "... s d, ... t d -> ... s t") * scale  # (..., seq_len, seq_len)
    if is_causal:
        q_indices = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        k_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        # causal_mask = (q_indices >= k_indices).to(dtype)  # (seq_len, seq_len)
        # scores = scores * causal_mask.unsqueeze(0) + (1.0 - causal_mask).unsqueeze(0) * float('-inf')
        causal_mask = (q_indices >= k_indices)[None, :, :]  # (1, seq_len, seq_len)
        scores = scores.masked_fill(~causal_mask, float("-inf"))

    # Compute softmax probabilities
    P = torch.exp(scores - L.unsqueeze(-1))  # exp(S_ij - L_i) (..., seq_len, seq_len)

    # Gradients w.r.t. V
    dV = einsum(P, dO, "... s t, ... s d -> ... t d") # P^T @ dO (..., seq_len, d_model)
    # Gradients w.r.t. P
    dP = einsum(dO, V, "... s d, ... t d -> ... s t")  # dO @ V^T (..., seq_len, seq_len)

    # D_i = sum_d O_id * dO_id = rowsum(O * dO)_i
    D = (O * dO).sum(dim=-1)  # (..., seq_len)

    # Gradients w.r.t. scores
    dS = P * (dP - D.unsqueeze(-1))  # (..., seq_len, seq_len)

    # Gradients w.r.t. Q
    dQ = einsum(dS, K, "... s t, ... t d -> ... s d") * scale  # (..., seq_len, d_model)
    # Gradients w.r.t. K
    dK = einsum(dS, Q, "... s t, ... s d -> ... t d") * scale  # (..., seq_len, d_model)

    return dQ, dK, dV


# def flash_bwd_complied(Q, K, V, O, L, dO, is_causal: bool):

#     if is_causal:
#         fn = torch.compile(_flash_attention_bwd)(Q, K, V, O, L, dO, True)
#     else:
#         fn = torch.compile(_flash_attention_bwd)(Q, K, V, O, L, dO, False)

#     return fn



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
    query_tile_index = tl.program_id(0) # along queries
    batch_index = tl.program_id(1)  # along batch

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
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
        strides=(stride_vk,stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )

    dtype = tl.float32

    # Load Q block
    Q_i = tl.load(Q_block_ptr) # (Q_TILE_SIZE, D)
    Q_i = Q_i.to(dtype)

    # Initialize online softmax states 
    m_i = tl.full((Q_TILE_SIZE, ), float('-inf'), dtype) # (Q_TILE_SIZE,)
    l_i = tl.zeros((Q_TILE_SIZE, ), dtype) # (Q_TILE_SIZE,)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype) # (Q_TILE_SIZE, D)

    # Iterate over K, V blocks
    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        # Load K, V block
        K_j = tl.load(K_block_ptr) # (K_TILE_SIZE, D)
        K_j = K_j.to(dtype)

        V_j = tl.load(V_block_ptr) # (K_TILE_SIZE, D)
        V_j = V_j.to(dtype)

        # Compute attention scores
        scores = tl.dot(Q_i, tl.trans(K_j)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)

        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = k_start + tl.arange(0, K_TILE_SIZE)
            masks = k_idx[None, :] <= q_idx[:, None]    # (Q_TILE_SIZE, K_TILE_SIZE)
            scores = tl.where(masks, scores, -float("inf"))

        # Compute block-wise max
        m_ij = tl.max(scores, axis=1) # (Q_TILE_SIZE,)
        m_i_new = tl.maximum(m_i, m_ij) # (Q_TILE_SIZE,)

        # Compute exp factor 
        exp_m = tl.exp(m_i - m_i_new) # (Q_TILE_SIZE,)

        # Compute exp(scores - m_i_new)
        P_ij = tl.exp(scores - m_i_new[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)

        # Update logsumexp
        l_ij = tl.sum(P_ij, axis=1) + exp_m * l_i # (Q_TILE_SIZE,)

        # Update output
        O_blk = tl.dot(P_ij, V_j) # (Q_TILE_SIZE, D)
        O_i_new = O_i * exp_m[:, None] + O_blk # (Q_TILE_SIZE, D)

        # Update states for next block
        m_i = m_i_new
        l_i = l_ij
        O_i = O_i_new
        
        # Advance K, V block pointers
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # Normalize output
    O_i = O_i / l_i[:, None] # (Q_TILE_SIZE, D)

    L_i = tl.log(l_i) + m_i # (Q_TILE_SIZE,)
    # Store output and logsumexp
    tl.store(O_block_ptr, O_i.to(Q_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty))

class FlashAttention2Forward_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        """
        FlashAttention-2 forward pass using Triton kernels.
        Q, K, V: (..., seq_len, d_model)

        returns: 
        O: (..., seq_len, d_model)

        Saved tensors for backward:
        Q, K, V, O, 
        L: (..., seq_len) logsumexp for each query position, L_i = logsumexp_j(Q_i . K_j^T)
        """

        *prefix, seq_len, d_model = Q.shape

        Bh = 1
        for x in prefix: Bh *= x

        Q = rearrange(Q, "... s d -> (...) s d").contiguous()
        K = rearrange(K, "... s d -> (...) s d").contiguous()
        V = rearrange(V, "... s d -> (...) s d").contiguous()

        device = Q.device
        dtypef32 = torch.float32

        # attention scale
        scale = 1.0 / (d_model ** 0.5)

        # Output
        O = torch.empty((Bh, seq_len, d_model), device=device, dtype=Q.dtype)
        L = torch.empty((Bh, seq_len), device=device, dtype=dtypef32)

        # Triton kernel launch parameters
        Bq: int = 32
        Bk: int = 32

        grid = ( (seq_len + Bq - 1) // Bq, Bh)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L, 
            Q.stride(0), Q.stride(1), Q.stride(2), 
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            seq_len, seq_len,
            scale,
            D=d_model,
            Q_TILE_SIZE=Bq, # type: ignore[arg-type]
            K_TILE_SIZE=Bk, # type: ignore[arg-type]
            is_causal=is_causal,    # type: ignore[arg-type]
        )

        # reshape back to original prefix shape
        O = O.view(*prefix, seq_len, d_model)
        L = L.view(*prefix, seq_len)
        # save for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O
    
    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass for FlashAttention-2 using PyTorch operations/Triton kernels.
        dO: (..., seq_len, d_model) gradient of output O
        Q, K, V, O are retrieved from ctx (..., seq_len, d_model)
        L: (..., seq_len) logsumexp for each query position, L_i = logsumexp_j(Q_i . K_j^T)
        scale: float 1/sqrt(d_model)
        is_causal: bool
        returns:
        dQ, dK, dV: gradients w.r.t. Q, K, V
        """
        # --- Using PyTorch implementation for backward ---
        # Q, K, V, O, L = ctx.saved_tensors
        # is_causal = ctx.is_causal

        # in_dtype = Q.dtype

        # Qf = Q.to(torch.float32)
        # Kf = K.to(torch.float32)
        # Vf = V.to(torch.float32)
        # Of = O.to(torch.float32)
        # dOf = dO.to(torch.float32)

        # if is_causal:
        #     flash_bwd_complied = _flash_bwd_causal(Qf, Kf, Vf, Of, L, dOf)
        # else:
        #     flash_bwd_complied = _flash_bwd_nocausal(Qf, Kf, Vf, Of, L, dOf)

        # dQ, dK, dV = flash_bwd_complied

        # return dQ, dK, dV, None
        # raise NotImplementedError("FlashAttention2Forward_triton backward is not implemented yet.")

        # --- Using Triton kernels for backward ---
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        *prefix, seq_len, d_model = Q.shape
        Bh = 1
        for x in prefix: Bh *= x

        Q = rearrange(Q, "... s d -> (...) s d").contiguous()
        K = rearrange(K, "... s d -> (...) s d").contiguous()
        V = rearrange(V, "... s d -> (...) s d").contiguous()
        O = rearrange(O, "... s d -> (...) s d").contiguous()
        dO = rearrange(dO, "... s d -> (...) s d").contiguous()
        L = rearrange(L, "... s -> (...) s").contiguous()
        device = Q.device
        dtypef32 = torch.float32
        scale = 1.0 / (d_model ** 0.5)
        # Output gradients
        dQ = torch.empty((Bh, seq_len, d_model), device=device, dtype=dtypef32)
        dK = torch.empty((Bh, seq_len, d_model), device=device, dtype=dtypef32)
        dV = torch.empty((Bh, seq_len, d_model), device=device, dtype=dtypef32)

        Bq: int = 16
        Bk: int = 16

        # Launch dQ kernel
        grid_dq = ( (seq_len + Bq - 1) // Bq, Bh)
        flash_bwd_dq_kernel[grid_dq](
            Q, K, V,
            O, dO,
            L,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), 
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            seq_len, seq_len,
            scale,
            D=d_model,
            Q_TILE_SIZE=Bq, # type: ignore[arg-type]
            K_TILE_SIZE=Bk, # type: ignore[arg-type]
            is_causal=is_causal,    # type: ignore[arg-type]
        )
        # Launch dK, dV kernel
        grid_dkv = ( (seq_len + Bk - 1) // Bk, Bh)
        flash_bwd_dkdv_kernel[grid_dkv](
            Q, K, V,
            O, dO,
            L,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            seq_len, seq_len,
            scale,
            D=d_model,
            Q_TILE_SIZE=Bq, # type: ignore[arg-type]
            K_TILE_SIZE=Bk, # type: ignore[arg-type]
            is_causal=is_causal,    # type: ignore[arg-type]
        )

        dQ = dQ.view(*prefix, seq_len, d_model)
        dK = dK.view(*prefix, seq_len, d_model)
        dV = dV.view(*prefix, seq_len, d_model)

        return dQ, dK, dV, None

        
# ======
# flash attention v2 backward triton kernel
# ======
@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr,    #(..., seq_len, d_model)
    O_ptr, dO_ptr,  #(..., seq_len, d_model)
    L_ptr,      #(..., seq_len)
    dQ_ptr,     #(..., seq_len, d_model)
    stride_qb, stride_qq, stride_qd,    
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    q_tile = tl.program_id(0) # along queries
    batch_idx = tl.program_id(1)  # along batch

    dtf32 = tl.float32

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_idx * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_tile * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_idx * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(q_tile * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(q_tile * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_idx * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(q_tile * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Load Q, O, dO, L blocks
    Q_i = tl.load(Q_block_ptr).to(dtf32) # (Q_TILE_SIZE, D)
    O_i = tl.load(O_block_ptr).to(dtf32) # (Q_TILE_SIZE, D)
    dO_i = tl.load(dO_block_ptr).to(dtf32) # (Q_TILE_SIZE, D)
    L_i = tl.load(L_block_ptr).to(dtf32) # (Q_TILE_SIZE,)

    # Compute D_i = sum_d O_id * dO_id
    D_i = tl.sum(O_i * dO_i, axis=1) # (Q_TILE_SIZE,)

    dQ_i = tl.zeros((Q_TILE_SIZE, D), dtf32) # (Q_TILE_SIZE, D)
    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        # Load K, V block
        K_j = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero").to(dtf32) # (K_TILE_SIZE, D)
        V_j = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero").to(dtf32) # (K_TILE_SIZE, D)
        
        # Compute attention scores
        scores = tl.dot(Q_i, tl.trans(K_j)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        if is_causal:
            q_idx = q_tile * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = k_start + tl.arange(0, K_TILE_SIZE)
            masks = k_idx[None, :] <= q_idx[:, None]    # (Q_TILE_SIZE, K_TILE_SIZE)
            scores = tl.where(masks, scores, -float("inf"))
        
        # Compute exp(scores - L_i)
        P = tl.exp(scores - L_i[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        # Compute dP = dO_i @ V_j^T
        dP = tl.dot(dO_i, tl.trans(V_j)) # (Q_TILE_SIZE, K_TILE_SIZE)
        # Compute dS = P * (dP - D_i)
        dS = P * (dP - D_i[:, None]) * scale # (Q_TILE_SIZE, K_TILE_SIZE)

        # Update dQ_i
        dQ_i += tl.dot(dS, K_j) # (Q_TILE_SIZE, D)

        # Advance K, V block pointers
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    # Store dQ_i
    tl.store(dQ_block_ptr, dQ_i.to(dQ_block_ptr.type.element_ty), boundary_check=(0,1))

@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr,    #(..., seq_len, d_model)
    O_ptr, dO_ptr,  #(..., seq_len, d_model)
    L_ptr,      #(..., seq_len)
    dK_ptr,     #(..., seq_len, d_model)
    dV_ptr,     #(..., seq_len, d_model)
    stride_qb, stride_qq, stride_qd,    
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_dkb, stride_dkq, stride_dkd,
    stride_dvb, stride_dvq, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    k_tile = tl.program_id(0) # along keys
    batch_idx = tl.program_id(1)  # along batch
    dtf32 = tl.float32

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(k_tile * K_TILE_SIZE, 0),  
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(k_tile * K_TILE_SIZE, 0),  
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_idx * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_idx * stride_dob,    
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(0, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_idx * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkq, stride_dkd),
        offsets=(k_tile * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_idx * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvq, stride_dvd),
        offsets=(k_tile * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Load K, V blocks
    K_j = tl.load(K_block_ptr).to(dtf32) # (K_TILE_SIZE, D)
    V_j = tl.load(V_block_ptr).to(dtf32) # (K_TILE_SIZE, D)

    dK_j = tl.zeros((K_TILE_SIZE, D), dtf32) # (K_TILE_SIZE, D)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtf32) # (K_TILE_SIZE, D)

    for q_start in range(0, N_QUERIES, Q_TILE_SIZE):
        # Load Q, O, dO, L blocks
        Q_i = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero").to(dtf32) # (Q_TILE_SIZE, D)
        O_i = tl.load(O_block_ptr, boundary_check=(0,1), padding_option="zero").to(dtf32) # (Q_TILE_SIZE, D)
        dO_i = tl.load(dO_block_ptr, boundary_check=(0,1), padding_option="zero").to(dtf32) # (Q_TILE_SIZE, D)
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(dtf32) # (Q_TILE_SIZE,)

        # Compute D_i = sum_d O_id * dO_id
        D_i = tl.sum(O_i * dO_i, axis=1) # (Q_TILE_SIZE,)

        # Compute attention scores
        scores = tl.dot(Q_i, tl.trans(K_j)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        if is_causal:
            q_idx = q_start + tl.arange(0, Q_TILE_SIZE)
            k_idx = k_tile * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            masks = k_idx[None, :] <= q_idx[:, None]    # (Q_TILE_SIZE, K_TILE_SIZE)
            scores = tl.where(masks, scores, -float("inf"))
        
        # Compute exp(scores - L_i)
        P = tl.exp(scores - L_i[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)

        # Compute dV += P^T @ dO_i
        dV_j += tl.dot(tl.trans(P), dO_i) # (K_TILE_SIZE, D)

        # Compute dP = dO_i @ V_j^T
        dP = tl.dot(dO_i, tl.trans(V_j)) # (Q_TILE_SIZE, K_TILE_SIZE)
        # Compute dS = P * (dP - D_i)
        dS = P * (dP - D_i[:, None]) * scale # (Q_TILE_SIZE, K_TILE_SIZE)

        # Update dK_j += dS^T @ Q_i
        dK_j += tl.dot(tl.trans(dS), Q_i) # (K_TILE_SIZE, D)

        # Advance Q, O, dO, L block pointers
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        O_block_ptr = tl.advance(O_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE, ))

    # Store dK_j, dV_j
    tl.store(dK_block_ptr, dK_j.to(dK_block_ptr.type.element_ty), boundary_check=(0,1))
    tl.store(dV_block_ptr, dV_j.to(dV_block_ptr.type.element_ty), boundary_check=(0,1))






    





    