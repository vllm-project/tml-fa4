# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'll need install nvidia-cutlass-dsl==4.2.0.

import os
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional, Tuple, Callable

import torch


import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from quack.compile_utils import make_fake_tensor as fake_tensor
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.testing import is_fake_mode


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from flash_attn.cute import cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    cute_dsl_ptxas.patch()


from flash_attn.cute import utils
from flash_attn.cute import fa_logging
from flash_attn.cute.cute_dsl_utils import (
    to_cute_tensor, to_cute_aux_tensor, get_aux_tensor_metadata, get_broadcast_dims,
)
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
from flash_attn.cute.flash_fwd_combine import FlashAttentionForwardCombine
from flash_attn.cute.flash_fwd_mla_sm100 import FlashAttentionMLAForwardSm100
from flash_attn.cute.shearing_bias import ShearingBias
from flash_attn.cute.prepare_scheduler import FlashPrepareScheduler, SchedulerMetadataTensorsTorch
from flash_attn.cute.cu_blocks_kernels import CuSeqlensToBlocksKernel, CuBlocksToBatchKernel

# SM100 head_dim=256 2CTA kernel imports
from flash_attn.cute.sm100_hd256_2cta_fmha_forward import BlackwellFusedMultiHeadAttentionForward

from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    get_sparse_q_block_size,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_config,
)

from flash_attn.cute.mixed_dtype_gemm import MixedDtypeGemmKernel, EpilogueFunction

def _parse_arch_str(arch_str):
    """Parse arch string (e.g. 'sm_80', 'sm_90a', '80', '100') to int (e.g. 80, 90, 100)."""
    import re
    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch():
    """Cached device arch check.

    Override with FLASH_ATTENTION_ARCH (e.g. 'sm_80' or '80') to select which
    kernel path to use (SM80/SM90/SM100/SM120) independently of the compilation
    target (CUTE_DSL_ARCH).

    For CPU-only compilation (no GPU), set both:
      FLASH_ATTENTION_ARCH=sm_80  (kernel selection)
      CUTE_DSL_ARCH=sm_80         (compilation target)
    """
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


@lru_cache(maxsize=None)
def _l2_cache_bytes_cached(device_id: int):
    props = torch.cuda.get_device_properties(device_id)
    return props.L2_cache_size


def _l2_cache_bytes():
    return _l2_cache_bytes_cached(torch.cuda.current_device())


def _validate_head_dims(head_dim: int, head_dim_v: int, compute_capability: int, alignment: int) -> None:
    """Validate head dimension constraints based on compute capability."""
    is_deepseek_shape = head_dim == 192 and head_dim_v == 128
    is_deepseek_mla_absorbed_shape = head_dim == 64 and head_dim_v == 512
    is_dedicate_kernel_shape = head_dim == 256 and head_dim_v == 256
    is_standard_range = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128

    is_sm90_range = 8 <= head_dim <= 256 and 8 <= head_dim_v <= 256
    if compute_capability == 9:
        assert is_sm90_range and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM90. "
            f"head_dim and head_dim_v must be between 8 and 256 and divisible by {alignment}."
        )
    elif compute_capability in [10, 11]:
        assert (is_standard_range or is_deepseek_shape or is_deepseek_mla_absorbed_shape or is_dedicate_kernel_shape) and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM100/SM110. "
            f"head_dim and head_dim_v must be between 8 and 128 and divisible by {alignment}, or (192, 128) for DeepSeek, or (256, 256) for hd256."
        )


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool


def _tile_size_fwd_sm90(head_dim, head_dim_v, is_causal, is_local, sparse_block_size_q=None):
    """Return FwdConfig for SM90 forward.

    Tile sizes and flags based on tile_size_fwd_sm90 in hopper/tile_size.h, adjusted
    for the Python kernel's different register/smem tradeoffs (benchmarked on H100 SXM).

    When sparse_block_size_q is set, tile_m must divide it. For head_dim <= 96 the
    optimal tile_m=192 is used when compatible, otherwise we fall back to 128.
    """
    if head_dim <= 64:
        # C++: 192×192 non-causal, 192×128 causal/local.
        # Python: 192×128 RS+OL is consistently best across seqlens.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, True, True)
        return FwdConfig(192, 128, True, True)
    elif head_dim <= 96:
        # C++: 192×144 noRS+OL for all cases.
        # Python: RS is catastrophic with 192× tiles (~300 vs ~600 TFLOPS).
        # noRS+OL is always required. Causal: 192×128 slightly better short seqlen.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, False, True)
        if is_causal or is_local:
            return FwdConfig(192, 128, False, True)
        else:
            return FwdConfig(192, 144, False, True)
    elif head_dim <= 128:
        return FwdConfig(128, 128, True, True)
    elif head_dim <= 192:
        tile_n = 96 if is_local else (128 if head_dim_v <= 128 else 112)
        return FwdConfig(128, tile_n, True, True)
    else:  # hdim 256
        tile_n = 64 if is_local else 80
        return FwdConfig(128, tile_n, True, True)

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert t.shape == expected_shape, f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    if not is_fake_mode():
        assert t.is_cuda, f"{name} must be on CUDA"


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
}


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    # If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if num_n_blocks <= 4:
        return 1

    # NOTE: We should revisit this heuristic after persistence is supported for split KV.
    # Sometimes, it's ideal to over-schedule splits for better efficiency.
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)


def _resolve_causal_local_window(causal, window_size_left, window_size_right, mask_mod=None):
    """Resolve causal/local/window settings into canonical form.

    Returns (causal, local, window_size_left, window_size_right).
    """
    if mask_mod is not None:
        return False, False, window_size_left, window_size_right
    if causal:
        window_size_right = 0
    if window_size_left is not None and window_size_right is not None and window_size_left + window_size_right < 0:
        window_size_left = None
        window_size_right = None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right


def _pack_gqa_heuristic(
    qhead_per_kvhead,
    tile_m=128,
    window_size_left=None,
    window_size_right=None,
    has_bias=False,
    is_bwd=False,
    requires_grad=False,
) -> bool:
    if tile_m % qhead_per_kvhead != 0:
        return False

    is_swa = window_size_left is not None and window_size_right is not None
    window_size = window_size_left + window_size_right if is_swa else None
    pack_gqa_swa = is_swa and window_size <= 1024  # TODO: tune this threshold
    pack_gqa_bwd = qhead_per_kvhead > 1 and pack_gqa_swa
    pack_gqa_fwd = qhead_per_kvhead > 1

    if is_bwd:
        return pack_gqa_bwd
    else:
        bwd_equals_fwd = has_bias and requires_grad
        return pack_gqa_fwd if not bwd_equals_fwd else pack_gqa_bwd


def _pack_gqa_swizzle_heuristic(
    pack_gqa,
    causal,
    window_size_left=None,
    window_size_right=None,
    has_bias=False,
) -> bool:
    is_swa = window_size_left is not None and window_size_right is not None
    pack_gqa_swizzle = causal or (is_swa and has_bias)  # TODO: tune this
    return pack_gqa_swizzle and pack_gqa


def _group_tile_bias(qhead_per_kvhead_packgqa=1):
    # return 128 * qhead_per_kvhead_packgqa
    return 128


def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    tile_mn: Optional[Tuple[int, int]] = None,
    mma_pv_is_rs: Optional[bool] = None,
    intra_wg_overlap: Optional[bool] = None,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _arch: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    return_logits_max: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    logits_max: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    rel_bias: Optional[torch.Tensor] = None,
    scheduler_metadata: Optional[SchedulerMetadataTensorsTorch] = None,
    seqlen_k_per_split: Optional[int] = None,
    disable_scheduler_metadata: bool = False,
    zfill_padded_output: bool = True,
    sfq: Optional[torch.Tensor] = None,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    qk_sf_vec_size: Optional[int] = None,
    v_sf_vec_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Forward pass for FlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity.
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
            The returned LSE supports taking gradient.
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.
        scheduler_metadata: Optional metadata for tile scheduler with varlen sequences. 
    """
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
        # TODO: allow for user-defined parameter
        q_sf_interleaved = seqused_q is None
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = max_seqlen_q
        total_q = q.shape[0]
        q_sf_interleaved = False
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
        kv_sf_interleaved = page_size == 128
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]
        kv_sf_interleaved = True
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        total_k = k.shape[-3]
        assert k.shape == (total_k, num_head_kv, head_dim)
        assert v.shape == (total_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )
    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )
    blockscaled = sfq is not None
    v_blockscaled = sfv is not None
    if page_table is not None and blockscaled:
        assert v_blockscaled, "paged KV with qk blockscaled requires v blockscaled"
    if v_blockscaled:
        assert v.dtype in [torch.float8_e4m3fn], "v_blockscaled V must be float8_e4m3fn"
        assert sfv.dtype == torch.float8_e8m0fnu, "sfv must be float8_e8m0fnu"
        assert v_sf_vec_size is not None, "v_sf_vec_size must be provided for v_blockscaled"
    if blockscaled:
        assert sfk is not None, "sfq and sfk must both be provided for blockscaled"
        assert qk_sf_vec_size is not None, "qk_sf_vec_size must be provided for blockscaled"
        assert q.dtype in [torch.float8_e4m3fn], "blockscaled Q must be float8_e4m3fn"
        assert q.dtype == k.dtype, "blockscaled Q and K must have the same dtype"
        assert sfq.dtype == torch.float8_e8m0fnu, "sfq must be float8_e8m0fnu"
        assert sfk.dtype == torch.float8_e8m0fnu, "sfk must be float8_e8m0fnu"
        if not v_blockscaled:
            assert v.dtype in [torch.float16, torch.bfloat16], "blockscaled V must be float16 or bfloat16"
    else:
        assert sfk is None, "sfq and sfk must both be provided for blockscaled"
        if v_blockscaled:
            assert q.dtype in [torch.float16, torch.bfloat16], "Q/K must be float16 or bfloat16"
            assert q.dtype == k.dtype, "Q and K must have the same dtype"
        else:
            assert q.dtype in [
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
            ], "inputs must be float16, bfloat16, or float8_e4m3fn"
            assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

    if not is_fake_mode():
        assert all(
            t is None or t.is_cuda
            for t in (
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                learnable_sink,
            )
        ), "inputs must be on CUDA device"
    arch = _get_device_arch() if _arch is None else _arch
    if q.dtype == torch.float8_e4m3fn and not blockscaled:
        assert arch in [100, 103], (
            "unscaled FP8 attention is only supported on SM100/SM103"
        )
        assert not (head_dim == 256 and head_dim_v == 256), (
            "unscaled FP8 attention does not support head_dim 256 "
            "(the dedicated hd256 kernel has no fp8 path)"
        )
    assert arch // 10 in [8, 9, 10, 11, 12], "Unsupported compute capability. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // q.element_size()
    if arch // 10 not in [8, 12]:
        _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim) if qv is None else 1.0 / math.sqrt(head_dim + head_dim_v)
    if softcap == 0.0:
        softcap = None

    plain_fp8 = q.dtype == torch.float8_e4m3fn and not blockscaled
    out_torch_dtype = (
        torch.bfloat16
        if v_blockscaled or plain_fp8
        else (v.dtype if blockscaled else q.dtype)
    )
    device = q.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    requires_grad = False

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device
        )
    else:
        _validate_tensor(
            out, "out", (*q_batch_seqlen_shape, num_head, head_dim_v), out_torch_dtype, device
        )

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    if logits_max is not None:
       return_logits_max = True
       _validate_tensor(logits_max, "logits_max", lse_shape, torch.float32, device)
    else:
        logits_max = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if return_logits_max
            else None
        )

    if return_logits_max:
        assert return_lse, f"{return_logits_max = } but {return_lse = }"

    if seqlen_k == 0:
        out.zero_()
        if lse is not None:
            lse.fill_(float("-inf"))
        if return_logits_max:
            logits_max.fill_(float("-inf"))
        return out, lse, logits_max, bias, None, None

    dtype = torch2cute_dtype_map[q.dtype]
    use_block_sparsity = block_sparse_tensors is not None

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right, mask_mod
    )

    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = _pack_gqa_heuristic(
            qhead_per_kvhead,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            has_bias=rel_bias is not None,
            requires_grad=requires_grad,
        )
    
    if pack_gqa and qv is not None and 128 % qhead_per_kvhead != 0:
        pack_gqa = False
    if arch // 10 in [10, 11] and (128 % qhead_per_kvhead != 0):
        pack_gqa = False
    
    if pack_gqa:
        q_sf_interleaved = False
    qhead_per_kvhead_packgqa = qhead_per_kvhead if pack_gqa else 1

    # requested_use_clc_scheduler = utils._get_use_clc_scheduler_default()
    requested_use_clc_scheduler = True
    requested_disable_2cta = utils._get_disable_2cta_default(is_fwd=True)

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # SM80/SM120: uses SM80 MMA, 128 threads (4 warps)
    if arch // 10 in [8, 12]:
        num_threads = 128

    fwd_cfg = FwdConfig(128, 128, True, True)  # default
    if tile_mn is None:
        if arch // 10 == 12:
            # SM120 tile sizes tuned for 99 KB SMEM capacity:
            # D<=64:  128x128 → 48 KB (good occupancy)
            # D>64:   128x64  → 64 KB (128x128 would use 96 KB, hurting occupancy)
            if head_dim <= 64:
                fwd_cfg = FwdConfig(128, 128, True, True)
            else:
                fwd_cfg = FwdConfig(128, 64, True, True)
        elif arch // 10 == 8:
            fwd_cfg = FwdConfig(128, 64, True, True)  # SM80, should tune
        elif arch // 10 == 9:
            sparse_q = get_sparse_q_block_size(block_sparse_tensors, seqlen_q)
            fwd_cfg = _tile_size_fwd_sm90(head_dim, head_dim_v, causal, local, sparse_block_size_q=sparse_q)
    else:
        fwd_cfg = FwdConfig(tile_mn[0], tile_mn[1], fwd_cfg.mma_pv_is_rs, fwd_cfg.intra_wg_overlap)
    tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
    if mma_pv_is_rs is None:
        mma_pv_is_rs = fwd_cfg.mma_pv_is_rs
    if intra_wg_overlap is None:
        intra_wg_overlap = fwd_cfg.intra_wg_overlap

    if max_seqlen_q is None:
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
    if max_seqlen_k is None:
        max_seqlen_k = seqlen_k
    if cu_seqlens_k is None and seqused_k is None:
        min_seqlen_k = seqlen_k 
    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    if arch // 10 == 10:
        # q_stage=2 hangs on sm100 for blockscaled; force q_stage=1 there.
        q_stage = 1 if blockscaled else (2 if seqlen_q_packgqa > tile_m else 1)
    else:
        q_stage = 1
    max_m_blocks_leq_one = seqlen_q_packgqa <= tile_m * q_stage
    tile_bias = (seqlen_q_packgqa + 8 - 1) // 8 * 8 if seqlen_q_packgqa < tile_m else tile_m

    m_block_size_effective = q_stage * tile_m
    seqlen_k_loaded = max_seqlen_k if not local else max(0, min(max_seqlen_k, (window_size_right or max_seqlen_k) + (window_size_left or max_seqlen_k) + 1 + tile_m))
    num_m_blocks = (seqlen_q_packgqa + m_block_size_effective - 1) // m_block_size_effective
    total_mblocks = batch_size * num_head_kv * num_m_blocks
    num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
    num_SMs = 132 if is_fake_mode() else torch.cuda.get_device_properties(device).multi_processor_count
    if num_splits < 1:
        num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)
        # print(f"num splits by heuristic = {num_splits} for {seqlen_q = }, {seqlen_k = }, {total_mblocks =}, {num_m_blocks = }, {num_n_blocks =},")

    # SplitKV uses float32 partial output, which doubles the O buffer size
    # in shared memory, causing OOM for diff-headdim (192, 128)
    if arch // 10 in [10, 11] and head_dim != head_dim_v and num_splits > 1:
        if num_n_blocks >= 64 and head_dim_v != 512:
            tile_n = 64
            num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
            num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)
        else:
            num_splits = 1

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = torch.empty(
            num_splits,
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=torch.float32,
            device=device,
        )
        lse_partial = torch.empty(num_splits, *lse_shape, dtype=torch.float32, device=device)
        logits_max_partial = torch.empty(num_splits, *lse_shape, dtype=torch.float32, device=device) if return_logits_max else None

    use_2cta_instrs = (
        arch // 10 in [10, 11]
        and not requested_disable_2cta
        and not causal
        and not local
        and not is_split_kv
        and cu_seqlens_q is None
        and seqused_q is None
        and not use_block_sparsity
        and page_size in [None, 128]
        and int(math.ceil(head_dim / 16) * 16) in [128, 192]
        and int(math.ceil(head_dim_v / 16) * 16) == 128
        and seqlen_q_packgqa > 2 * tile_m
        and (tile_m % qhead_per_kvhead == 0 or not pack_gqa)
        and rel_bias is None
        and not blockscaled
    )

    # hd=256 2CTA forward uses dedicated kernel (SM100 only)
    use_dedicated_hd256_kernel = arch // 10 == 10 and head_dim == 256 and head_dim_v == 256
    use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)
    elif score_mod is not None:
        if arch // 10 == 8:
            raise NotImplementedError("Custom user-provided score_mod is not supported on SM8x architectures.")
        
    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )

    # CLC regressed for varlen MHA and dense noncausal. Imbalanced varlen shapes
    # keep more K/V blocks in flight and hurt L2; dense noncausal mostly just
    # pays work-stealing overhead.
    is_varlen_mha = is_varlen and qhead_per_kvhead == 1
    is_dense_noncausal = not is_varlen and not causal and not local
    use_clc_scheduler = requested_use_clc_scheduler and not is_varlen_mha and not is_dense_noncausal

    if use_block_sparsity:
        # NB: pack_gqa requires block sparse head dim == 1 (broadcasted)
        head_dim_idx = 0 if block_sparse_tensors.mask_block_cnt.ndim == 2 else 1
        if pack_gqa and block_sparse_tensors.mask_block_cnt.shape[head_dim_idx] != 1:
            pack_gqa = False
        if is_split_kv:
            raise NotImplementedError(
                "Block sparsity is not yet supported with SplitKV. TODO: partition sparse block lists per split."
            )
        if cu_seqlens_q is not None:
            assert block_sparse_tensors.cu_total_m_blocks is not None, (
                "Varlen block sparsity requires block_sparse_tensors.cu_total_m_blocks."
            )

    # See get_broadcast_dims for why this is needed in compile key
    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    q_subtile_factor = None
    if block_sparse_tensors is not None:
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
            q_subtile_factor,
        ) = normalize_block_sparse_config(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(tile_m, tile_n),
            q_stage=q_stage,
        )
    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None
    
    # rel_bias -> bias
    cu_total_m_blocks_bias = None
    blocks_to_batch_idx = None

    if rel_bias is not None:
        rel_extent = rel_bias.shape[-1]
        rel_extent_padded = rel_extent + 256
        assert rel_extent % 128 == 0
        assert tile_m == 128
        assert tile_n == 128
        assert (
            causal
            or window_size_left is None
            or (window_size_right is not None and window_size_left + window_size_right + 1 == rel_extent)
        ), "for relative bias, require causal (with possibly shifted diagonal) or window length == rel_extent"
        if cu_seqlens_q is None:
            bias_seqlen_q_rounded = (seqlen_q + tile_m - 1) // tile_m * tile_m
            assert rel_bias.shape == (batch_size, seqlen_q, num_head, rel_extent)
            bias = torch.empty(
                batch_size,
                bias_seqlen_q_rounded,
                num_head,
                rel_extent_padded,
                dtype=rel_bias.dtype,
                device=device,
            )
        else:
            bias_total_q_padded = total_q + tile_m
            assert rel_bias.shape == (total_q, num_head, rel_extent)
            bias = torch.empty(
                bias_total_q_padded,
                num_head,
                rel_extent_padded,
                dtype=rel_bias.dtype,
                device=device,
            )

        rows_per_cta = 4
        group_tile_bias = _group_tile_bias(qhead_per_kvhead_packgqa)

        use_prepare_bias_kernel = (
            cu_seqlens_q is not None
            and max_m_blocks_leq_one is False
            and batch_size <= 1024
            # and False
        )

        if use_prepare_bias_kernel:
            cu_total_m_blocks_bias = torch.empty(batch_size + 1, dtype=torch.int32, device=device)

            compile_key_prepare = (
                group_tile_bias,
                qhead_per_kvhead_packgqa,
            )

            if compile_key_prepare not in _flash_attn_fwd.compile_cache_prepare_shear_bias:
                (
                    cu_total_m_blocks_bias_tensor,
                    cu_seqlens_q_tensor
                ) = [
                    to_cute_tensor(t, assumed_align=4, leading_dim=0)
                    for t in (cu_total_m_blocks_bias, cu_seqlens_q)
                ]

                prepare_shear_bias = CuSeqlensToBlocksKernel(
                    tile=group_tile_bias,
                    seqlen_multiple=qhead_per_kvhead_packgqa,
                )

                _flash_attn_fwd.compile_cache_prepare_shear_bias[compile_key_prepare] = (
                    cute.compile(
                        prepare_shear_bias,
                        cu_total_m_blocks_bias_tensor,
                        cu_seqlens_q_tensor,
                        current_stream,
                        options="--enable-tvm-ffi",
                    )
                )
            
            if not is_fake_mode():
                _flash_attn_fwd.compile_cache_prepare_shear_bias[compile_key_prepare](
                    cu_total_m_blocks_bias,
                    cu_seqlens_q,
                )

        precompute_batch_from_block = use_prepare_bias_kernel
        
        if precompute_batch_from_block:
            assert use_prepare_bias_kernel

            total_group_blocks_max = (
                total_q * qhead_per_kvhead_packgqa + batch_size * (group_tile_bias - 1)
            ) // group_tile_bias
            blocks_to_batch_idx = torch.empty(total_group_blocks_max, dtype=torch.int32, device=device)

            compile_key_blocks_to_batch = ()

            if compile_key_blocks_to_batch not in _flash_attn_fwd.compile_cache_blocks_to_batch:
                cu_total_m_blocks_bias_tensor = to_cute_tensor(cu_total_m_blocks_bias, assumed_align=4, leading_dim=0)
                blocks_to_batch_idx_tensor = to_cute_tensor(blocks_to_batch_idx, assumed_align=4, leading_dim=0)
                
                blocks_to_batch_kernel = CuBlocksToBatchKernel()

                _flash_attn_fwd.compile_cache_blocks_to_batch[compile_key_blocks_to_batch] = (
                    cute.compile(
                        blocks_to_batch_kernel,
                        cu_total_m_blocks_bias_tensor,
                        blocks_to_batch_idx_tensor,
                        current_stream,
                        options="--enable-tvm-ffi",
                    )
                )

            if not is_fake_mode():
                _flash_attn_fwd.compile_cache_blocks_to_batch[compile_key_blocks_to_batch](
                    cu_total_m_blocks_bias,
                    blocks_to_batch_idx,
                )

        compile_key = (
            rel_bias.dtype,
            rel_extent,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            pack_gqa,
            qhead_per_kvhead,
            rows_per_cta,
            group_tile_bias,
            max_m_blocks_leq_one,
            cu_total_m_blocks_bias is not None,
            blocks_to_batch_idx is not None,
        )
        if compile_key not in _flash_attn_fwd.compile_cache_shear_bias:
            (
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                cu_total_m_blocks_bias_tensor,
                blocks_to_batch_idx_tensor,
            ) = [
                to_cute_tensor(t, assumed_align=4, leading_dim=0) if t is not None else None
                for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cu_total_m_blocks_bias, blocks_to_batch_idx)
            ]
            rel_bias_tensor = to_cute_tensor(rel_bias)
            bias_tensor = to_cute_tensor(bias)

            shearing_bias = ShearingBias(
                rel_extent,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                qhead_per_kvhead=qhead_per_kvhead,
                rows_per_cta=rows_per_cta,
                tile_m=group_tile_bias,
                max_m_blocks_leq_one=max_m_blocks_leq_one,
            )
            _flash_attn_fwd.compile_cache_shear_bias[compile_key] = cute.compile(
                shearing_bias,
                rel_bias_tensor,
                bias_tensor,
                max_seqlen_q,
                max_seqlen_k,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                cu_total_m_blocks_bias_tensor,
                blocks_to_batch_idx_tensor,
                window_size_left,
                window_size_right,
                current_stream,
                options="--enable-tvm-ffi",
            )

        if not is_fake_mode():
            _flash_attn_fwd.compile_cache_shear_bias[compile_key](
                rel_bias,
                bias,
                max_seqlen_q,
                max_seqlen_k,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                cu_total_m_blocks_bias,
                blocks_to_batch_idx,
                window_size_left,
                window_size_right,
            )
    else:
        rel_extent = 0
        rel_extent_padded = 0
        bias = None

    # if bias is not None:
    #     rel_extent_padded = bias.shape[-1]
    #     assert rel_extent_padded % 128 == 0
    #     if cu_seqlens_q is None:
    #         assert bias.shape == (batch_size, seqlen_q, num_head, rel_extent_padded)
    #     else:
    #         assert bias.shape == (total_q, num_head, rel_extent_padded)
    # else:
    #     rel_extent_padded = 0

    if qv is not None:
        assert arch // 10 in [10, 11], "only support Blackwell arch with qv"
        assert qv.shape[:-1] == q.shape[:-1]
        assert qv.shape[-1] == head_dim_v
        assert head_dim == 64 and head_dim_v == 512, "only support MLA weight absorbed shape with qv"
        assert not local, "local not yet supported with qv"
        assert page_table is None, "page table not yet supported with qv"

        assert not is_split_kv, "split kv not supported with qv"
        assert learnable_sink is None
        assert softcap is None
        assert score_mod is None
        assert mask_mod is None
        
        qv = maybe_contiguous(qv)

        gather_kv_length = 2048
        sparse_kv = gather_kv_indices is not None
        disable_sparse_kv_bitmask = False
        if sparse_kv:
            assert gather_kv_indices.shape[:-1] == q.shape[:-2]
            gather_kv_length = gather_kv_indices.shape[-1]
            assert gather_kv_length % 256 == 0
            if min_seqlen_k is None or causal:
                disable_sparse_kv_bitmask = False
            else:
                # seqlen_k_boundary = min_seqlen_k - max_seqlen_q + 1 if causal else min_seqlen_k
                seqlen_k_boundary = min_seqlen_k
                disable_sparse_kv_bitmask = seqlen_k_boundary >= gather_kv_length
    else:
        gather_kv_length = None
        sparse_kv = None
        disable_sparse_kv_bitmask = None
    
    reuse_scheduler_metadata = scheduler_metadata is not None
    if is_split_kv and scheduler_metadata is None and is_varlen and not disable_scheduler_metadata:
        scheduler_metadata = get_scheduler_metadata(
            num_batch=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            nheads=num_head,
            nheads_k=num_head_kv,
            headdim=head_dim,
            headdim_v=head_dim_v,
            num_splits=num_splits,
            tile_m=tile_m,
            tile_n=tile_n,
            pack_gqa=pack_gqa,
            causal=causal,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            seqlen_k_per_split=seqlen_k_per_split,
            zfill_padded_output=zfill_padded_output,
        )

    has_scheduler_metadata=scheduler_metadata is not None and not disable_scheduler_metadata
    if has_scheduler_metadata:
        (
            num_m_blocks,
            num_splits_dynamic,
            varlen_batch_idx,
            num_nheads_in_l2,
            tile_count_semaphore,
        ) = scheduler_metadata
        assert all(
            t is None or t.is_cuda
            for t in (
                num_m_blocks,
                num_splits_dynamic,
                varlen_batch_idx,
                num_nheads_in_l2,
                tile_count_semaphore,
            )
        ), "scheduler metadata must be on CUDA device"
        assert all(
            t is None or t.shape == (batch_size, )
            for t in (
                num_m_blocks,
                num_splits_dynamic,
                varlen_batch_idx,
                num_nheads_in_l2,
            )
        ), "these scheduler metadata tensors must have shape (batch_size, )"
        if tile_count_semaphore is not None:
            assert tile_count_semaphore.shape == (1, ), "semaphore has size 1"
        # print("In interface: num_splits_dynamic = ", num_splits_dynamic)
    else:
        num_m_blocks = None
        num_splits_dynamic = None
        varlen_batch_idx = None
        num_nheads_in_l2 = None
        tile_count_semaphore = None

    paged_kv_non_tma = page_size not in [None, tile_n]

    is_persistent = (
        not causal
        and not local
        and cu_seqlens_q is None
        and seqused_q is None
        and not is_split_kv
    )
    # override
    if max_m_blocks_leq_one and not is_split_kv:
        is_persistent = True
    is_dynamic_persistent_varlen = (
        tile_count_semaphore is not None and
        (cu_seqlens_q is not None or seqused_q is not None)
    )

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        aux_tensor_metadata,
        lse is None,
        logits_max is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        max_seqlen_q is not None,  # technically needed but we always set this
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        block_sparse_tensors is None or block_sparse_tensors.cu_total_m_blocks is None,
        block_sparse_tensors is None or block_sparse_tensors.cu_block_idx_offsets is None,
        tile_m,
        tile_n,
        q_stage,
        num_threads,
        is_split_kv,
        pack_gqa,
        arch,
        paged_kv_non_tma,  # paged KV non-TMA
        use_2cta_instrs,
        q_subtile_factor,
        mma_pv_is_rs,
        intra_wg_overlap,
        use_clc_scheduler,
        qv is not None,
        gather_kv_length,
        sparse_kv,
        disable_sparse_kv_bitmask,
        bias is not None,
        tile_bias,
        rel_extent,
        has_scheduler_metadata,
        tile_count_semaphore is not None,
        seqlen_k_per_split,
        blockscaled,
        qk_sf_vec_size,
        v_blockscaled,
        v_sf_vec_size,
        q_sf_interleaved,
        kv_sf_interleaved,
        sfq.ndim if sfq is not None else None,
        sfk.ndim if sfk is not None else None,
        sfv.ndim if sfv is not None else None,
        is_persistent,
        fa_logging.get_fa_log_level(),
    )

    if compile_key not in _flash_attn_fwd.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        page_table_tensor = (
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            if page_table is not None
            else None
        )
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out if not is_split_kv else out_partial)
        ]
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        elif lse is not None:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)
        else:
            lse_tensor = None
        
        if return_logits_max:
            if is_split_kv:
                logits_max_tensor = to_cute_tensor(logits_max_partial, assumed_align=4)
            else:
                logits_max_tensor = to_cute_tensor(logits_max, assumed_align=4)
        else:
            logits_max_tensor = None

        if has_scheduler_metadata:
            num_splits_dynamic_cute = to_cute_tensor(num_splits_dynamic, assumed_align=4)
            tile_count_semaphore_cute = to_cute_tensor(tile_count_semaphore, assumed_align=4)
        else:
            num_splits_dynamic_cute = None
            tile_count_semaphore_cute = None

        sparse_tensors = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        num_aux_tensors = len(aux_tensors) if aux_tensors is not None else 0
        if num_aux_tensors > 0:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]
        else:
            aux_tensors = None
            cute_aux_tensors = None

        qv_tensor = to_cute_tensor(qv) if qv is not None else None
        gather_kv_indices_tensor = to_cute_tensor(gather_kv_indices) if gather_kv_indices is not None else None

        bias_tensor = to_cute_tensor(bias) if bias is not None else None

        if blockscaled:
            # print("sfq torch shape = ", sfq.shape)
            # print("sfk torch shape = ", sfk.shape)
            sfq_tensor = to_cute_tensor(sfq)
            sfk_tensor = to_cute_tensor(sfk)
        else:
            sfq_tensor = None
            sfk_tensor = None
        # if v_blockscaled:
        #     print("sfv torch shape = ", sfv.shape)
        sfv_tensor = to_cute_tensor(sfv) if v_blockscaled else None

        if arch // 10 == 8:
            assert page_table is None, "paged KV not supported on SM 8.0"
            assert not is_split_kv, "SplitKV not supported on SM 8.0"
            fa_fwd = FlashAttentionForwardSm80(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=1,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        elif arch // 10 == 9:
            assert not is_split_kv, "SplitKV not supported on SM 9.0"
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                intra_wg_overlap=intra_wg_overlap,
                mma_pv_is_rs=mma_pv_is_rs,
                mask_mod=mask_mod,
                score_mod=score_mod,
                has_aux_tensors=aux_tensors is not None,
                q_subtile_factor=q_subtile_factor,
                paged_kv_non_tma=paged_kv_non_tma,
            )
        elif arch // 10 in [10, 11]:
            if qv is not None:
                fa_fwd = FlashAttentionMLAForwardSm100(
                    is_causal=causal,
                    use_cpasync_load_KV=sparse_kv,
                    topk_length=gather_kv_length,
                    is_topk_gather=sparse_kv,
                    pack_gqa=pack_gqa,
                    qhead_per_kvhead=qhead_per_kvhead,
                    nheads_kv=num_head_kv,
                    is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
                    disable_bitmask=disable_sparse_kv_bitmask,
                )
            else:
                if use_dedicated_hd256_kernel:
                    # hd=256 2CTA forward: check for currently unsupported features
                    assert softcap is None, "SM100 forward with head_dim=256 does not support softcap"
                    assert not use_block_sparsity, \
                        "SM100 forward with head_dim=256 does not support block sparsity"
                    assert learnable_sink is None, \
                        "SM100 forward with head_dim=256 does not support learnable_sink"
                    assert seqused_q is None and seqused_k is None, \
                        "SM100 forward with head_dim=256 does not support seqused_q/seqused_k"
                    if page_table is not None:
                        assert max_seqlen_k % page_size == 0, (
                            f"SM100 hd256 2CTA paged KV requires max_seqlen_k divisible by "
                            f"page_size ({page_size}), got max_seqlen_k={max_seqlen_k}"
                        )
                        assert page_table.shape[1] == max_seqlen_k // page_size, (
                            f"SM100 hd256 2CTA paged KV requires page_table.shape[1] == "
                            f"max_seqlen_k // page_size ({max_seqlen_k} // {page_size} = "
                            f"{max_seqlen_k // page_size}), got {page_table.shape[1]}; "
                            f"pass page_table[:, :{max_seqlen_k // page_size}] to slice to "
                            f"the actual sequence length"
                        )
                        assert page_table.stride(0) == page_table.shape[1], (
                            f"SM100 hd256 2CTA paged KV requires a fully contiguous page_table "
                            f"(stride(0)={page_table.stride(0)} must equal "
                            f"shape[1]={page_table.shape[1]})"
                        )
                    # pack_gqa is an auto-selected optimization; disable it for hd256 kernel
                    pack_gqa = False

                flash_fwd_obj_cls = (
                    BlackwellFusedMultiHeadAttentionForward
                    if use_dedicated_hd256_kernel
                    else FlashAttentionForwardSm100
                )

                fa_fwd = flash_fwd_obj_cls(
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_causal=causal,
                    is_local=local,
                    is_split_kv=is_split_kv,
                    pack_gqa=pack_gqa,
                    m_block_size=tile_m,
                    n_block_size=tile_n,
                    bias_block_size=tile_bias,
                    q_stage=q_stage,
                    is_persistent=is_persistent,
                    is_dynamic_persistent_varlen=is_dynamic_persistent_varlen,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    paged_kv_non_tma=paged_kv_non_tma,
                    is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
                    q_subtile_factor=q_subtile_factor,
                    use_2cta_instrs=use_2cta_instrs,
                    use_clc_scheduler=use_clc_scheduler,
                    has_bias=bias is not None,
                    rel_extent_padded=rel_extent_padded,
                    has_scheduler_metadata=has_scheduler_metadata,
                    seqlen_k_per_split=seqlen_k_per_split,
                    qk_blockscaled=blockscaled,
                    v_dequant=v_blockscaled,
                    q_sf_interleaved=q_sf_interleaved,
                    kv_sf_interleaved=kv_sf_interleaved,
                )
        elif arch // 10 == 12:
            # SM120 (Blackwell GeForce / DGX Spark): uses SM80 MMA with SM120 SMEM capacity
            assert not use_block_sparsity, "Block sparsity not supported on SM 12.0"
            assert page_table is None, "Paged KV not supported on SM 12.0 in this PR"
            assert not is_split_kv, "SplitKV not supported on SM 12.0 in this PR"
            fa_fwd = FlashAttentionForwardSm120(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=1,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {arch}. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
            )
        # TODO: check @can_implement
        if qv is not None:
            _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
                fa_fwd,
                q_tensor,
                qv_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                gather_kv_indices_tensor,
                page_table_tensor,
                window_size_left,
                window_size_right,
                current_stream,
                options="--enable-tvm-ffi",
            )
        elif arch // 10 in [10, 11]:
            _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
                fa_fwd,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                logits_max_tensor,
                softmax_scale,
                sfq_tensor,  # mSFQ
                sfk_tensor,  # mSFK
                sfv_tensor,  # mSFV
                qk_sf_vec_size,  # qk_sf_vec_size
                v_sf_vec_size,  # v_sf_vec_size
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                page_table_tensor,
                window_size_left,
                window_size_right,
                learnable_sink_tensor,
                sparse_tensors,
                cute_aux_tensors,
                bias_tensor,
                num_splits_dynamic_cute,
                tile_count_semaphore_cute,
                max_seqlen_q,
                current_stream,
                options="--enable-tvm-ffi",
            )
        else:
            _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
                fa_fwd,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                page_table_tensor,
                window_size_left,
                window_size_right,
                learnable_sink_tensor,
                sparse_tensors,
                cute_aux_tensors,
                num_splits_dynamic_cute,
                current_stream,
                options="--enable-tvm-ffi",
            )

    if not is_fake_mode():
        if qv is not None:
            _flash_attn_fwd.compile_cache[compile_key](
                q.detach(),
                qv.detach(),
                k.detach(),
                v.detach(),
                out.detach(),
                lse,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                gather_kv_indices,
                page_table,
                window_size_left,
                window_size_right,
            )
        elif arch // 10 in [10, 11]:
            exec_args = [
                q.detach(),
                k.detach(),
                v.detach(),
                out.detach() if not is_split_kv else out_partial,
                lse_partial if is_split_kv else lse,
                logits_max_partial if is_split_kv else logits_max,
                softmax_scale,
            ]
            exec_args.extend([
                sfq,  # mSFQ (None when not blockscaled)
                sfk,  # mSFK (None when not blockscaled)
                sfv,  # mSFV (None when not v_blockscaled)
            ])
            exec_args.extend([
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                window_size_left,
                window_size_right,
                learnable_sink,
                normalized_block_sparse_tensors[:4]
                if normalized_block_sparse_tensors is not None
                else None,
                aux_tensors,
                bias,
                num_splits_dynamic,
                tile_count_semaphore,
                max_seqlen_q,
            ])
            _flash_attn_fwd.compile_cache[compile_key](*exec_args)
        else:
            _flash_attn_fwd.compile_cache[compile_key](
                q.detach(),
                k.detach(),
                v.detach(),
                out.detach(),
                lse,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                window_size_left,
                window_size_right,
                learnable_sink,
                normalized_block_sparse_tensors[:4]
                if normalized_block_sparse_tensors is not None
                else None,
                aux_tensors,
                num_splits_dynamic,
            )
    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
            logits_max_partial.transpose(-1, -2) if logits_max_partial is not None else None,
            logits_max.transpose(-1, -2) if logits_max is not None else None,
            cu_seqlens=cu_seqlens_q,
            seqused=seqused_q,
            num_splits_dynamic=num_splits_dynamic if has_scheduler_metadata else None,
            semaphore_to_reset=tile_count_semaphore if has_scheduler_metadata else None,
            max_seqlen_q=max_seqlen_q,
        )
    elif reuse_scheduler_metadata and tile_count_semaphore is not None:
        tile_count_semaphore.zero_()
    return out, lse, logits_max, bias, cu_total_m_blocks_bias, blocks_to_batch_idx


_flash_attn_fwd.compile_cache = get_jit_cache("fwd")
_flash_attn_fwd.compile_cache_shear_bias = get_jit_cache("fwd_shear_bias")
_flash_attn_fwd.compile_cache_prepare_shear_bias = get_jit_cache("fwd_prepare_shear_bias")
_flash_attn_fwd.compile_cache_blocks_to_batch = get_jit_cache("blocks_to_batch")


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_bias: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    block_sparse_tensors_bwd: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    return_logits_max: bool = False,
    sfq: Optional[torch.Tensor] = None,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
):
    del deterministic, score_mod_bwd, block_sparse_tensors_bwd
    out, lse, logits_max, *_ = _flash_attn_fwd(
        q,
        k,
        v,
        qv=qv,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        learnable_sink=learnable_sink,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        score_mod=score_mod,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        block_sparse_tensors=block_sparse_tensors,
        return_lse=return_lse,
        gather_kv_indices=gather_kv_indices,
        return_logits_max=return_logits_max,
        rel_bias=rel_bias,
        sfq=sfq,
        sfk=sfk,
        sfv=sfv,
        qk_sf_vec_size=32 if sfq is not None else None,
        v_sf_vec_size=32 if sfv is not None else None,
    )
    return (out, lse, logits_max) if return_logits_max else (out, lse)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_bias: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    aux_tensors: Optional[list] = None,
    return_lse: bool = False,
    return_logits_max: bool = False,
    scheduler_metadata: Optional[SchedulerMetadataTensorsTorch] = None,
    seqlen_k_per_split: Optional[int] = None,
    disable_scheduler_metadata: bool = False,
    zfill_padded_output: bool = True,
    sfq: Optional[torch.Tensor] = None,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    del deterministic, score_mod_bwd
    qk_sf_vec_size = None if sfq is None else 32
    v_sf_vec_size = None if sfv is None else 32
    out, lse, logits_max, *_ = _flash_attn_fwd(
        q,
        k,
        v,
        qv=qv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        min_seqlen_k=min_seqlen_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        learnable_sink=learnable_sink,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        score_mod=score_mod,
        mask_mod=mask_mod,
        block_sparse_tensors=block_sparse_tensors,
        aux_tensors=aux_tensors,
        return_lse=return_lse,
        gather_kv_indices=gather_kv_indices,
        return_logits_max=return_logits_max,
        rel_bias=rel_bias,
        scheduler_metadata=scheduler_metadata,
        seqlen_k_per_split=seqlen_k_per_split,
        disable_scheduler_metadata=disable_scheduler_metadata,
        zfill_padded_output=zfill_padded_output,
        sfq=sfq,
        sfk=sfk,
        sfv=sfv,
        qk_sf_vec_size=qk_sf_vec_size,
        v_sf_vec_size=v_sf_vec_size,
        out=out,
    )
    return (out, lse, logits_max) if return_logits_max else (out, lse)


def _compile_fwd_combine(
    dtype, dtype_partial, head_dim, num_head, tile_m, k_block_size, log_max_splits,
    has_cu_seqlens, has_seqused, has_lse, has_varlen_batch_idx,
    has_num_splits_dynamic, has_semaphore_to_reset, max_seqlen_q, has_combine_semaphore,
    has_logits_max,
):
    """Compile fwd combine kernel using cute fake tensors (no real GPU tensors needed)."""
    sym = cute.sym_int
    div = 128 // dtype_partial.width  # 16-byte alignment in elements

    fa_combine = FlashAttentionForwardCombine(
        dtype=dtype,
        dtype_partial=dtype_partial,
        head_dim=head_dim,
        num_head=num_head,
        tile_m=tile_m,
        k_block_size=k_block_size,
        log_max_splits=log_max_splits,
    )
    if not fa_combine.can_implement(
        dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
        num_threads=256,
    ):
        raise RuntimeError(
            "FlashAttention combine kernel cannot be implemented with given parameters"
        )

    if has_cu_seqlens:
        # Varlen: (num_splits, total_q, nheads, headdim)
        num_splits, total_q, nheads = sym(), sym(), sym()
        mO_partial = fake_tensor(dtype_partial, (num_splits, total_q, nheads, head_dim), divisibility=div)
        mLSE_partial = fake_tensor(Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=1)
        mO = fake_tensor(dtype, (total_q, nheads, head_dim), divisibility=div)
        mLSE = fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=0) if has_lse else None
        mLogitsMax_partial = fake_tensor(Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=1) if has_logits_max else None
        mLogitsMax = fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=0) if has_logits_max else None
    else:
        # Batched: (num_splits, batch, seqlen, nheads, headdim)
        num_splits, batch, seqlen, nheads = sym(), sym(), sym(), sym()
        mO_partial = fake_tensor(dtype_partial, (num_splits, batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE_partial = fake_tensor(Float32, (num_splits, batch, seqlen, nheads), divisibility=1, leading_dim=2)
        mO = fake_tensor(dtype, (batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE = fake_tensor(Float32, (batch, seqlen, nheads), divisibility=1, leading_dim=1) if has_lse else None
        mLogitsMax_partial = fake_tensor(Float32, (num_splits, batch, seqlen, nheads), divisibility=1, leading_dim=2) if has_logits_max else None
        mLogitsMax = fake_tensor(Float32, (batch, seqlen, nheads), divisibility=1, leading_dim=1) if has_logits_max else None
        batch = mO_partial.shape[1]

    batch_for_1d = batch if not has_cu_seqlens else sym()
    batchp1 = sym()
    semaphore_num = sym()
    mCuSeqlens = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cu_seqlens else None
    mSeqused = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_seqused else None
    mNumSplitsDynamic = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_num_splits_dynamic else None
    mVarlenBatchIdx = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_varlen_batch_idx else None
    mSemaphore = fake_tensor(Int32, (semaphore_num,), divisibility=1) if has_semaphore_to_reset else None
    mCombineSemaphore = fake_tensor(Int32, (semaphore_num,), divisibility=1) if has_combine_semaphore else None

    return cute.compile(
        fa_combine,
        mO_partial, mLSE_partial, mO, mLSE, mLogitsMax_partial, mLogitsMax,
        mCuSeqlens, mSeqused, mNumSplitsDynamic, mVarlenBatchIdx, mSemaphore,
        max_seqlen_q, mCombineSemaphore,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    logits_max_partial: Optional[torch.Tensor] = None,
    logits_max: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic: Optional[torch.Tensor] = None,
    varlen_batch_idx: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    use_combine_semaphore: bool = False,
) -> None:
    """Forward combine kernel for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs.

    Args:
        out_partial: Partial outputs tensor (num_splits, batch, seqlen, nheads, headdim) or
                                            (num_splits, total_q, nheads, headdim) if there's cu_seqlens
        lse_partial: Partial LSE tensor (num_splits, batch, seqlen, nheads) or
                                        (num_splits, total_q, nheads) if there's cu_seqlens
        logits_max_partial: Partial row max tensor (num_splits, batch, seqlen, nheads) or
                                                (num_splits, total_q, nheads) if there's cu_seqlens
        out: Output tensor (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim) if there's cu_seqlens
        lse: Output LSE tensor (batch, seqlen, nheads) or (total_q, nheads) if there's cu_seqlens.
        logits_max: Output row max tensor (batch, seqlen, nheads) or (total_q, nheads) if there's cu_seqlens.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        num_splits_dynamic: Dynamic number of splits per batch
        semaphore_to_reset: Semaphore for synchronization
        max_seqlen_q: Maximum seqlen_q for any batch, used if there's cu_seqlens.

    Returns:
        None
    """
    assert out_partial.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    if not is_fake_mode():
        assert out_partial.is_cuda and lse_partial.is_cuda, "tensors must be on CUDA device"
    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4

    # Validate output tensor shapes and types
    assert out.shape == out_partial.shape[1:], "out shape mismatch"
    if lse is not None:
        assert lse.shape == lse_partial.shape[1:], "lse shape mismatch"
        assert lse.dtype == torch.float32, "lse must be fp32"

    if logits_max_partial is not None:
        assert logits_max is not None, "logits_max not provided"
        assert logits_max_partial.is_cuda, "logits_max_partial must be on CUDA device"
        assert logits_max_partial.dtype == torch.float32, "logits_max_partial must be fp32"
        assert logits_max_partial.shape == out_partial.shape[:-1], "logits_max_partial shape mismatch"
        assert logits_max_partial.stride(-2) == 1, "logits_max_partial must be contiguous in the seqlen dimension"

    if logits_max is not None:
        assert logits_max_partial is not None, "logits_max_partial not provided"
        assert logits_max.is_cuda, "logits_max must be on CUDA device"
        assert logits_max.shape == logits_max_partial.shape[1:], "logits_max shape mismatch"
        assert logits_max.dtype == torch.float32, "logits_max must be fp32"

    # Validate optional tensors
    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic, "num_splits_dynamic"),
    ]:
        if t is not None:
            if not is_fake_mode():
                assert t.is_cuda, f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"
    head_dim = out_partial.shape[-1]
    num_head = out_partial.shape[-2]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    # If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    # so that kBlockM is smaller and we have more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    # We want kBlockM to be as small as possible to maximize parallelism.
    # E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    tile_m = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if tile_m == 8:
        # If kBlockM == 8 then the minimum number of splits is 32.
        # TODO: we can deal w this by using 128 threads instead
        log_max_splits = max(log_max_splits, 5)

    if use_combine_semaphore:
        combine_semaphore = torch.zeros(1, dtype=torch.int32, device="cuda")
    else:
        combine_semaphore = None

    # Create combine kernel configuration
    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]
    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        num_head,
        tile_m,
        k_block_size,
        log_max_splits,
        cu_seqlens is not None,
        seqused is not None,
        lse is not None,
        varlen_batch_idx is not None,
        num_splits_dynamic is not None,
        semaphore_to_reset is not None,
        max_seqlen_q,
        combine_semaphore is not None,
        logits_max is not None,
    )
    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        _flash_attn_fwd_combine.compile_cache[compile_key] = _compile_fwd_combine(
            *compile_key
        )
    if not is_fake_mode():
        _flash_attn_fwd_combine.compile_cache[compile_key](
            out_partial, lse_partial, out, lse, logits_max_partial, logits_max,
            cu_seqlens, seqused, num_splits_dynamic, varlen_batch_idx,
            semaphore_to_reset, max_seqlen_q, combine_semaphore,
        )


_flash_attn_fwd_combine.compile_cache = get_jit_cache("fwd_combine")


def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    varlen_batch_idx: Optional[torch.Tensor] = None,
    return_lse: bool = True,
    max_seqlen_q: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Flash Attention combine function for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs. This is the main user-facing
    interface for the combine kernel.

    Args:
        out_partial: Partial outputs tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads, head_size) for regular batched input
            - (num_splits, total_q, num_heads, head_size) for variable length input
        lse_partial: Partial LSE tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads) for regular batched input
            - (num_splits, total_q, num_heads) for variable length input
        out: Optional output tensor. If None, will be created automatically.
        out_dtype: Optional output dtype. If None, will use fp16/bf16 based on input.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        varlen_batch_idx: Optional mapping from virtual batch index to real batch index
            (int32 tensor of shape (batch_size,)). Used by persistent tile schedulers
            that reorder batch processing for load balancing.
        return_lse: Whether to return the combined LSE tensor. Default is True.
        max_seqlen_q: Maximum seqlen_q for any batch, used if there's cu_seqlens.

    Returns:
        Tuple of (out, lse) where:
        - out: Combined output tensor with shape (batch_size, seqlen, num_heads, head_size)
              or (total_q, num_heads, head_size) for varlen
        - lse: Combined log-sum-exp tensor with shape (batch_size, seqlen, num_heads)
              or (total_q, num_heads) for varlen. None if return_lse=False

    Note:
        This function expects the input tensors to be in the format produced by
        split attention computation, where the first dimension is num_splits.
        The permuting from user format to kernel format is now done inside the kernel.
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4
    if is_varlen:
        # Variable length: (num_splits, total_q, num_heads, head_size)
        num_splits, total_q, num_heads, head_size = out_partial.shape
        batch_size = 1  # Treat as single batch for varlen
        seqlen = total_q
    else:
        # Regular batched: (num_splits, batch_size, seqlen, num_heads, head_size)
        num_splits, batch_size, seqlen, num_heads, head_size = out_partial.shape
    # Determine output dtype
    if out_dtype is None:
        out_dtype = out_partial.dtype
    # Create output if not provided
    device = out_partial.device
    if out is None:
        if is_varlen:
            out = torch.empty(total_q, num_heads, head_size, dtype=out_dtype, device=device)
        else:
            out = torch.empty(
                batch_size, seqlen, num_heads, head_size, dtype=out_dtype, device=device
            )
    # Create lse output only if requested
    if return_lse:
        if is_varlen:
            lse = torch.empty(num_heads, total_q, dtype=torch.float32, device=device)
        else:
            lse = torch.empty(batch_size, num_heads, seqlen, dtype=torch.float32, device=device)
        lse = lse.transpose(-1, -2)
    else:
        lse = None
    _flash_attn_fwd_combine(
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        varlen_batch_idx=varlen_batch_idx,
        max_seqlen_q=max_seqlen_q,
    )
    return out, lse

def get_scheduler_metadata(
    num_batch: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    nheads: int,
    nheads_k: int,
    headdim: int,
    num_splits: int,
    tile_m: int,
    tile_n: int,
    headdim_v: Optional[int] = None,
    pack_gqa: Optional[bool] = False,
    causal: bool = False,
    enable_pdl: bool = False,
    sort: bool = False,
    seqlen_k_new: int = 0,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqlen_k_per_split: Optional[int] = None,
    zfill_padded_output: bool = True,
) -> SchedulerMetadataTensorsTorch:
    """
    Helper method to get scheduler metadata for varlen sequences.
    """
    # Determine device from input tensors
    device = None
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            device = t.device
            break
    if device is None:
        raise ValueError(
            "At least one of cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be provided to determine device"
        )
    if headdim_v is None:
        headdim_v = headdim

    # Override enable_pdl (not supported yet)
    enable_pdl = False

    # Override sort (not supported yet)
    sort = False
    
    if seqlen_k_per_split is not None:
        assert seqlen_k_per_split % tile_n == 0, "seqlen per split must be divisible by tile_n"
        n_blocks_per_split = seqlen_k_per_split // tile_n
    else:
        n_blocks_per_split = None

    # Allocate metadata tensors (torch tensors)
    num_m_blocks = None
    num_splits_dynamic = torch.empty(num_batch, dtype=torch.int32, device=device)
    varlen_batch_idx = None
    num_nheads_in_l2 = None
    tile_count_semaphore = torch.empty(1, dtype=torch.int32, device=device)
    # Will enable more metadata preparation in future commit
    # num_m_blocks = torch.empty(num_batch, dtype=torch.int32, device=device)
    # varlen_batch_idx = torch.empty(num_batch, dtype=torch.int32, device=device) if sort else None
    # num_nheads_in_l2 = torch.empty(num_batch, dtype=torch.int32, device=device) if causal else None

    # Compute num_warps based on num_batch (capped at 32)
    num_warps = min((num_batch + 30) // 31, 32)
    # Round up to nearest power of 2
    num_warps = 1 << (num_warps - 1).bit_length()

    cache_key = (
        num_warps,
        tile_m,
        tile_n,
        nheads,
        nheads_k,
        headdim,
        headdim_v,
        causal,
        pack_gqa,
        sort,
        cu_seqlens_q is not None,
        cu_seqlens_k is not None,
        cu_seqlens_k_new is not None,
        seqused_q is not None,
        seqused_k is not None,
        leftpad_k is not None,
        num_m_blocks is not None,
        num_splits_dynamic is not None,
        varlen_batch_idx is not None,
        num_nheads_in_l2 is not None,
        tile_count_semaphore is not None,
        n_blocks_per_split is not None,
        zfill_padded_output,
    )

    if cache_key not in get_scheduler_metadata.compile_cache:
        (
            num_m_blocks_cute,
            num_splits_dynamic_cute,
            varlen_batch_idx_cute,
            num_nheads_in_l2_cute,
            tile_count_semaphore_cute,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            cu_seqlens_k_new_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            leftpad_k_tensor,
         ) = [
            to_cute_tensor(t, assumed_align=4)
            if t is not None
            else None
            for t in (
                num_m_blocks,
                num_splits_dynamic,
                varlen_batch_idx,
                num_nheads_in_l2,
                tile_count_semaphore,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_seqlens_k_new,
                seqused_q,
                seqused_k,
                leftpad_k,
            )
        ]
        scheduler = FlashPrepareScheduler(
            num_warps,
            tile_m,
            tile_n,
            nheads,
            nheads_k,
            headdim,
            headdim_v,
            causal,
            packgqa=pack_gqa,
            sort=sort,
            zfill_padded_output=zfill_padded_output,
        )
        get_scheduler_metadata.compile_cache[cache_key] = cute.compile(
            scheduler,
            max_seqlen_q,
            max_seqlen_k,
            seqlen_k_new,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            cu_seqlens_k_new_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            leftpad_k_tensor,
            num_batch,
            num_splits,
            tile_count_semaphore_cute,
            num_m_blocks_cute,
            num_splits_dynamic_cute,
            varlen_batch_idx_cute,
            num_nheads_in_l2_cute,
            n_blocks_per_split,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        get_scheduler_metadata.compile_cache[cache_key](
            max_seqlen_q,
            max_seqlen_k,
            seqlen_k_new,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_seqlens_k_new,
            seqused_q,
            seqused_k,
            leftpad_k,
            num_batch,
            num_splits,
            tile_count_semaphore,
            num_m_blocks,
            num_splits_dynamic,
            varlen_batch_idx,
            num_nheads_in_l2,
            n_blocks_per_split,
        )

    return SchedulerMetadataTensorsTorch(
        num_m_blocks_ptr=num_m_blocks,
        num_splits_dynamic_ptr=num_splits_dynamic,
        varlen_batch_idx_ptr=varlen_batch_idx,
        num_nheads_in_l2_ptr=num_nheads_in_l2,
        tile_count_semaphore=tile_count_semaphore,
    )

get_scheduler_metadata.compile_cache = get_jit_cache("scheduler_metadata")


def _compile_mixed_dtype_gemm(
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    mma_dtype: torch.dtype,
    acc_dtype: torch.dtype,
    c_dtype: torch.dtype,
    a_major: Literal["k", "m"],
    b_major: Literal["k", "n"],
    c_major: Literal["m", "n"],
    epilogue_op: Optional[Callable | EpilogueFunction] = None,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    use_2cta_instrs: bool = True,
    keep_ptx: bool = True,
):
    a_dtype = torch2cute_dtype_map[a_dtype]
    b_dtype = torch2cute_dtype_map[b_dtype]
    c_dtype = torch2cute_dtype_map[c_dtype]
    acc_dtype = torch2cute_dtype_map[acc_dtype]

    if epilogue_op is None:
        epilogue_op = lambda x: x

    # Interpret mma_dtype = torch.float32 as tf32
    torch2cute_dtype_map_mma = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.TFloat32,
    }
    mma_dtype = torch2cute_dtype_map_mma[mma_dtype]

    gemm = MixedDtypeGemmKernel(
        a_dtype,
        b_dtype,
        mma_dtype,
        acc_dtype,
        c_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
    )
    # Check if configuration can be implemented
    gemm.check_can_implement()

    m_fake, n_fake, k_fake, l_fake = [cute.sym_int() for _ in range(4)]
    a_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype,
        (l_fake, m_fake, k_fake),
        stride_order=(2, 1, 0) if a_major == "k" else (2, 0, 1),
        assumed_align=16,
    )
    b_fake = cute.runtime.make_fake_compact_tensor(
        b_dtype,
        (l_fake, n_fake, k_fake),
        stride_order=(2, 1, 0) if b_major == "k" else (2, 0, 1),
        assumed_align=16,
    )
    c_fake = cute.runtime.make_fake_compact_tensor(
        c_dtype,
        (l_fake, m_fake, n_fake),
        stride_order=(2, 1, 0) if c_major == "n" else (2, 0, 1),
        assumed_align=16,
    )

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compile_options = "--enable-tvm-ffi"
    if keep_ptx:
        compile_options += " --keep-ptx --generate-line-info"
    return cute.compile(
        gemm,
        a_fake,
        b_fake,
        c_fake,
        stream,
        epilogue_op=epilogue_op,
        options=compile_options,
    )


def mixed_dtype_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor],
    mma_dtype: torch.dtype,
    acc_dtype: torch.dtype = torch.float32,
    c_dtype: Optional[torch.dtype] = None,
    epilogue_op: Optional[Callable | EpilogueFunction] = None,
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: tuple[int, int] = (2, 1),
    use_2cta_instrs: bool = True,
    unfused: Optional[bool] = None,
) -> Optional[torch.Tensor]:
    """Mixed dtype GEMM.

    Args:
        a: Shape (L, M, K), either M- or K-major, bf16, fp16, or fp32.
        b: Shape (L, N, K), either N- or K-major, bf16, fp16, or fp32.
        c: Shape (L, M, N), either M- or N-major, bf16, fp16, or fp32.
            If c is not provided then it is allocated here based on the provided c_dtype.
        mma_dtype: Internal dtype for MMA: torch.bfloat16, torch.float16, or torch.float32
            (interpreted as tf32).
        acc_dtype: dtype for MMA accumulation: torch.float16 or torch.float32.
        c_dtype: Optional dtype for c if a tensor was not provided: bf16, fp16, or fp32.
        epilogue_op: Operation to fuse in epilogue, should act on cute.TensorSSA.
        mma_tiler_mn: MMA tile size (tuning parameter). M should be 64, 128, or 256.
            N should be a multiple of 32 in [32, 256].
        cluster_shape_mn: Cluster size (tuning parameter).
        use_2cta_instrs: Whether to use 2CTA MMA (tuning parameter). Requires MMA tile
            size M to be 128 or 256, and cluster size M to be divisible by 2.
        unfused: Whether to do dtype conversion in GMEM first (tuning parameter). If
            None, we set this with a heuristic: only use unfused path for compute-bound
            cases with bf16 MMA.

    Returns:
        c, if not running in fake mode.
    """

    if c is None:
        assert c_dtype is not None
        l = a.shape[0]
        m = a.shape[1]
        n = b.shape[1]
        c = torch.empty(l, m, n, dtype=c_dtype, device=a.device)
    assert a.ndim == 3, f"Expected a in shape (L, M, K), got {a.shape}"
    assert b.ndim == 3, f"Expected b in shape (L, N, K), got {b.shape}"
    assert c.ndim == 3, f"Expected c in shape (L, M, N), got {c.shape}"

    def get_leading_dim(t):
        for i, stride in enumerate(t.stride()):
            if stride == 1:
                return i

    a_leading_dim = get_leading_dim(a)
    b_leading_dim = get_leading_dim(b)
    c_leading_dim = get_leading_dim(c)
    a_major = [None, "m", "k"][a_leading_dim]
    b_major = [None, "n", "k"][b_leading_dim]
    c_major = [None, "m", "n"][c_leading_dim]

    assert a_major is not None, (
        f"Expected a to be M- or K-major, got shape {a.shape}, stride {a.stride()}"
    )
    assert b_major is not None, (
        f"Expected b to be N- or K-major, got shape {b.shape}, stride {b.stride()}"
    )
    assert c_major is not None, (
        f"Expected c to be M- or N-major, got shape {c.shape}, stride {c.stride()}"
    )

    # Heuristic: unfused path for compute-bound bf16 to avoid SMEM bottleneck
    if unfused is None:
        m = a.shape[1]
        k = a.shape[2]
        n = b.shape[1]
        unfused = (mma_dtype == torch.bfloat16 and m >= 512 and n >= 512 and k >= 512)
    if unfused and a.dtype != mma_dtype:
        a_ = a.to(mma_dtype)
    else:
        a_ = a
    if unfused and b.dtype != mma_dtype:
        b_ = b.to(mma_dtype)
    else:
        b_ = b

    assert a_.shape[a_leading_dim] % (16 // a_.element_size()) == 0, f"Expected 16B alignment for a"
    assert b_.shape[b_leading_dim] % (16 // b_.element_size()) == 0, f"Expected 16B alignment for b"
    assert c.shape[c_leading_dim] % (16 // c.element_size()) == 0, f"Expected 16B alignment for c"

    if epilogue_op is None:
        epilogue_op_hash = False
    elif isinstance(epilogue_op, EpilogueFunction):
        epilogue_op_hash = epilogue_op
    else:
        epilogue_op_hash = utils.hash_callable(epilogue_op)

    cache_key = (
        a_.dtype,
        b_.dtype,
        mma_dtype,
        acc_dtype,
        c.dtype,
        a_major,
        b_major,
        c_major,
        epilogue_op_hash,
        mma_tiler_mn,
        cluster_shape_mn,
        use_2cta_instrs,
    )

    if cache_key not in mixed_dtype_gemm.compile_cache:
        mixed_dtype_gemm.compile_cache[cache_key] = _compile_mixed_dtype_gemm(
            a_.dtype,
            b_.dtype,
            mma_dtype,
            acc_dtype,
            c.dtype,
            a_major,
            b_major,
            c_major,
            epilogue_op,
            mma_tiler_mn,
            cluster_shape_mn,
            use_2cta_instrs,
        )

    if not is_fake_mode():
        mixed_dtype_gemm.compile_cache[cache_key](a_, b_, c)
        return c

mixed_dtype_gemm.compile_cache = get_jit_cache("mixed_dtype_gemm")
