"""L1 operator equivalence: cuLA verify/commit vs seg_la verify/scatter.

No model load required — tests the adapter conventions (layout, decay sign,
scale, off-by-one) by running the same random (q, k, v, h0) through both
paths and asserting outputs and committed state match.
"""

import unittest

import torch

from sglang.srt.layers.attention.linear.seg_la import SegLaMeta, seg_la_fwd

try:
    from cula.lightning import (
        linear_attention_verify_kvbuffer,
        linear_attention_state_update_kvbuffer,
    )

    CULA_AVAILABLE = True
except ImportError:
    CULA_AVAILABLE = False


def _skip_if_no_gpu():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")


def _run_seg_la_mtp(q, k, v, state_kmajor, decay, scale, B, T, H, D):
    """Run seg_la MTP verify and return (output, caches, cache_idx)."""
    s_seg = state_kmajor.clone()
    caches = torch.zeros(B * T, H, D, D, device="cuda", dtype=torch.float32)
    pool_idx = torch.arange(B, device="cuda", dtype=torch.int32)
    flat_cache_idx = pool_idx * T
    meta = SegLaMeta(
        batch_size=B,
        max_q_length=T,
        q_offsets=torch.arange(0, B * T + 1, T, device="cuda", dtype=torch.int32),
        s_offsets=pool_idx,
        q_lengths=torch.full((B,), T, device="cuda", dtype=torch.int32),
        s_scales=torch.ones(B, device="cuda", dtype=torch.int32),
    )
    o_seg = seg_la_fwd(
        q, k, v, s_seg, decay.view(H, 1, 1), meta,
        caches=caches, cache_indices=flat_cache_idx, softmax_scale=scale,
    )
    return o_seg, caches, flat_cache_idx


@unittest.skipUnless(CULA_AVAILABLE, "cuLA not installed (pip install cuda-linear-attention)")
class TestCulaVsSegLaEquivalence(unittest.TestCase):
    """Validate cuLA verify/commit produces the same results as seg_la."""

    def test_cula_verify_matches_seg_la_mtp(self):
        _skip_if_no_gpu()
        B, T, H, D = 2, 4, 4, 128
        torch.manual_seed(0)
        scale = D**-0.5
        decay = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

        q = torch.randn(B * T, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B * T, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B * T, H, D, device="cuda", dtype=torch.bfloat16)
        state_kmajor = torch.randn(B, H, D, D, device="cuda", dtype=torch.float32) * 0.01

        o_seg, _, _ = _run_seg_la_mtp(q, k, v, state_kmajor, decay, scale, B, T, H, D)

        s_cula = state_kmajor.transpose(-1, -2).contiguous()
        pool_idx = torch.arange(B, device="cuda", dtype=torch.int32)
        out_cula = torch.zeros(B, T, H, D, device="cuda", dtype=torch.bfloat16)
        linear_attention_verify_kvbuffer(
            q.view(B, T, H, D), k.view(B, T, H, D), v.view(B, T, H, D),
            s_cula, out_cula, decay, pool_idx, scale, T,
        )

        rel = (
            (out_cula.view(B * T, H, D).float() - o_seg.float()).pow(2).mean().sqrt()
            / (o_seg.float().abs().max() + 1e-8)
        )
        self.assertLess(rel.item(), 1e-2, f"verify output mismatch vs seg_la: {rel:.5f}")

    def test_cula_commit_matches_seg_la_scatter(self):
        """After full accept (L=T), committed state must match seg_la's last intermediate."""
        _skip_if_no_gpu()
        B, T, H, D = 2, 4, 4, 128
        torch.manual_seed(42)
        scale = D**-0.5
        decay = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

        q = torch.randn(B * T, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B * T, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B * T, H, D, device="cuda", dtype=torch.bfloat16)
        state_kmajor = torch.randn(B, H, D, D, device="cuda", dtype=torch.float32) * 0.01

        _, caches, flat_cache_idx = _run_seg_la_mtp(
            q, k, v, state_kmajor, decay, scale, B, T, H, D
        )
        seg_last_state = torch.stack(
            [caches[int(flat_cache_idx[b]) + T - 1] for b in range(B)]
        )

        pool_idx = torch.arange(B, device="cuda", dtype=torch.int32)
        s_cula = state_kmajor.transpose(-1, -2).contiguous()
        accepted_len = torch.full((B,), T, device="cuda", dtype=torch.int32)
        linear_attention_state_update_kvbuffer(
            k.view(B, T, H, D), v.view(B, T, H, D),
            s_cula, decay, pool_idx, accepted_len, T,
        )
        cula_committed = s_cula.transpose(-1, -2).contiguous()

        rel = (
            (cula_committed - seg_last_state).pow(2).mean().sqrt()
            / (seg_last_state.abs().max() + 1e-8)
        )
        self.assertLess(rel.item(), 1e-3, f"commit state mismatch vs seg_la: {rel:.5f}")

    def test_cula_commit_partial_accept(self):
        """Partial accept (L<T) committed state matches seg_la's intermediate[L-1]."""
        _skip_if_no_gpu()
        B, T, H, D = 2, 4, 4, 128
        L = 2
        torch.manual_seed(123)
        scale = D**-0.5
        decay = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

        q = torch.randn(B * T, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B * T, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B * T, H, D, device="cuda", dtype=torch.bfloat16)
        state_kmajor = torch.randn(B, H, D, D, device="cuda", dtype=torch.float32) * 0.01

        _, caches, flat_cache_idx = _run_seg_la_mtp(
            q, k, v, state_kmajor, decay, scale, B, T, H, D
        )
        seg_state_at_L = torch.stack(
            [caches[int(flat_cache_idx[b]) + L - 1] for b in range(B)]
        )

        pool_idx = torch.arange(B, device="cuda", dtype=torch.int32)
        s_cula = state_kmajor.transpose(-1, -2).contiguous()
        accepted_len = torch.full((B,), L, device="cuda", dtype=torch.int32)
        linear_attention_state_update_kvbuffer(
            k.view(B, T, H, D), v.view(B, T, H, D),
            s_cula, decay, pool_idx, accepted_len, T,
        )
        cula_committed = s_cula.transpose(-1, -2).contiguous()

        rel = (
            (cula_committed - seg_state_at_L).pow(2).mean().sqrt()
            / (seg_state_at_L.abs().max() + 1e-8)
        )
        self.assertLess(rel.item(), 1e-3, f"partial commit (L={L}) state mismatch: {rel:.5f}")


if __name__ == "__main__":
    unittest.main()
