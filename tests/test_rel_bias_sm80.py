"""Relative-bias forward on SM80 (A100) against an fp32 PyTorch reference.

Varlen, causal, with cached context (seqlen_k > seqlen_q), so the sheared-bias
column shift is exercised for chunked-prefill and decode shapes as well as full
prefill.
"""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 8:
    pytest.skip("SM80 relative-bias path", allow_module_level=True)

from flash_attn.cute import flash_attn_varlen_func  # noqa: E402

D = 128


def reference(q, k, v, rel, scale, past):
    """q (Tq, Hq, D), k/v (Tk, Hkv, D), rel (Tq, Hq, E): fp32 causal attention where
    query t at position past + t adds rel[t, h, pos - j] for key j when
    0 <= pos - j < E (zero beyond the extent)."""
    tq, hq, _ = q.shape
    tk, hkv, _ = k.shape
    extent = rel.shape[-1]
    rep = hq // hkv
    kf = k.float().repeat_interleave(rep, 1)
    vf = v.float().repeat_interleave(rep, 1)
    s = torch.einsum("thd,khd->htk", q.float(), kf) * scale
    pos = past + torch.arange(tq, device=q.device)
    dist = pos[:, None] - torch.arange(tk, device=q.device)[None, :]
    in_extent = (dist >= 0) & (dist < extent)
    bias = (
        rel.float().permute(1, 0, 2).gather(2, dist.clamp(0, extent - 1)[None].expand(hq, tq, tk))
    )
    s = s + torch.where(in_extent[None], bias, torch.zeros_like(bias))
    s = s.masked_fill(~(dist >= 0)[None], float("-inf"))
    return torch.einsum("htk,khd->thd", torch.softmax(s, -1), vf)


CASES = {
    # name: (query lengths, cached context lengths)
    "prefill": ([256], [0]),
    "prefill_unaligned": ([200], [0]),
    "chunked": ([512], [1500]),
    "chunked_unaligned": ([200], [1000]),
    "decode": ([1, 1, 1], [700, 2047, 129]),
    "mixed": ([257, 1, 64], [0, 3000, 700]),
}


@pytest.mark.parametrize("extent", [128, 1024])
@pytest.mark.parametrize("heads", [(8, 8), (16, 2)])
@pytest.mark.parametrize("case", list(CASES))
@torch.inference_mode()
def test_rel_bias_varlen_matches_reference(case, heads, extent):
    tq_list, ctx_list = CASES[case]
    hq, hkv = heads
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    tk_list = [c + t for c, t in zip(ctx_list, tq_list)]
    q = torch.randn(sum(tq_list), hq, D, generator=g, device=dev).bfloat16() / D**0.25
    k = torch.randn(sum(tk_list), hkv, D, generator=g, device=dev).bfloat16() / D**0.25
    v = torch.randn(sum(tk_list), hkv, D, generator=g, device=dev).bfloat16()
    rel = (torch.randn(sum(tq_list), hq, extent, generator=g, device=dev) * 2.0).bfloat16()
    cu_q = torch.tensor([0] + tq_list, device=dev).cumsum(0).int()
    cu_k = torch.tensor([0] + tk_list, device=dev).cumsum(0).int()
    scale = D**-0.5

    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        rel_bias=rel,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(tq_list),
        max_seqlen_k=max(tk_list),
        softmax_scale=scale,
        causal=True,
    )
    assert torch.isfinite(out).all()
    for r, past in enumerate(ctx_list):
        qs, qe = int(cu_q[r]), int(cu_q[r + 1])
        ks, ke = int(cu_k[r]), int(cu_k[r + 1])
        ref = reference(q[qs:qe], k[ks:ke], v[ks:ke], rel[qs:qe], scale, past)
        no_bias = reference(q[qs:qe], k[ks:ke], v[ks:ke], torch.zeros_like(rel[qs:qe]), scale, past)
        err = (out[qs:qe].float() - ref).abs()
        # The bias must matter for this check to mean anything.
        assert (ref - no_bias).abs().mean() > 10 * err.mean()
        torch.testing.assert_close(out[qs:qe].float(), ref, atol=2e-2, rtol=2e-2)
