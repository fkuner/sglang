import torch

from cula.ops.lightning_attn import lightning_attn_fwd_varlen
from cula.ops.la_decode import linear_attention_decode
from cula.lightning import (
    linear_attention_verify_kvbuffer,
    linear_attention_state_update_kvbuffer,
)


def cula_prefill(q, k, v, temporal, cache_indices, cu_seqlens, decay, scale):
    """Prefill via cuLA varlen Lightning Attention.

    Args:
        q, k, v: [total_tokens, H, D] bf16 packed
        temporal: [pool_size, H, V, K] fp32 V-major state pool (per-layer slice)
        cache_indices: [N] int32 indices into pool
        cu_seqlens: [N+1] int32 cumulative seq lens
        decay: [H, 1, 1] fp32 positive slopes
        scale: float softmax scale
    Returns:
        o: [total_tokens, H, D] bf16
    """
    total_tokens = q.shape[0]
    o, _ = lightning_attn_fwd_varlen(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        decay.view(-1),
        cu_seqlens,
        scale=scale,
        state_pool=temporal,
        initial_state_indices=cache_indices.to(torch.int32),
    )
    return o.squeeze(0)


def cula_decode(q, k, v, temporal, cache_indices, decay, scale, out):
    """Single-token decode via cuLA.

    Args:
        q, k, v: [B, H, D] bf16
        temporal: [pool_size, HV, V, K] fp32 V-major state pool (per-layer slice)
        cache_indices: [B] int32 indices into pool
        decay: [H, 1, 1] fp32 positive slopes
        scale: float softmax scale
        out: [B, H, D] bf16 preallocated output
    Returns:
        out: [B, H, D] bf16
    """
    HEAD_DIM = q.shape[-1]
    V_DIM = v.shape[-1]
    pool_size = temporal.shape[0]
    HV = temporal.shape[1]
    temporal_3d = temporal.view(pool_size * HV, temporal.shape[2], temporal.shape[3])
    linear_attention_decode(
        q,
        k,
        v,
        temporal_3d,
        out,
        softmax_scale=scale,
        stride_q=q.stride(0),
        stride_k=k.stride(0),
        stride_v=v.stride(0),
        stride_s=temporal_3d.stride(0),
        stride_o=out.stride(0),
        s_offsets=cache_indices.to(torch.int32),
        decay_scales=decay.view(-1),
        HEAD_DIM=HEAD_DIM,
        K_SPLIT_DIM=HEAD_DIM,
        V_SPLIT_DIM=V_DIM,
    )
    return out


def cula_verify(q, k, v, temporal, cache_indices, decay, scale, T, out):
    """Parallel verify via cuLA KVBuffer.

    Args:
        q, k, v: [B*T, H, D] bf16 packed (uniform T per request)
        temporal: [pool_size, HV, V, K] fp32 V-major state pool (per-layer slice)
        cache_indices: [B] int32 indices into pool
        decay: [H, 1, 1] fp32 positive slopes
        scale: float softmax scale
        T: int draft_token_num
        out: [B*T, HV, V] bf16 preallocated output
    Returns:
        out reshaped to [B*T, HV, V]
    """
    B = cache_indices.shape[0]
    H = q.shape[1]
    K = q.shape[2]
    HV = v.shape[1]
    V = v.shape[2]
    q4 = q.view(B, T, H, K)
    k4 = k.view(B, T, H, K)
    v4 = v.view(B, T, HV, V)
    out4 = out.view(B, T, HV, V)
    linear_attention_verify_kvbuffer(
        q4, k4, v4, temporal, out4,
        decay.view(-1), cache_indices.to(torch.int32), scale, T,
    )
    return out


def cula_commit(draft_k, draft_v, temporal, cache_indices, accepted_len, decay, T):
    """Commit accepted state via cuLA KVBuffer state_update.

    Args:
        draft_k: [B, T, H, K] bf16
        draft_v: [B, T, HV, V] bf16
        temporal: [pool_size, HV, V, K] fp32 state pool
        cache_indices: [B] int32 indices into pool
        accepted_len: [B] int32 in [0, T]
        decay: [H, 1, 1] fp32 positive slopes
        T: int draft_token_num
    """
    linear_attention_state_update_kvbuffer(
        draft_k, draft_v, temporal, decay.view(-1),
        cache_indices.to(torch.int32), accepted_len.to(torch.int32), T,
    )
