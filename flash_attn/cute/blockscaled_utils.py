import torch
import cutlass
import cutlass.cute as cute


def _ceil_div(a, b):
    return (a + b - 1) // b


def quantize_to_fp8_with_sf(x_bf16, sf_vec_size=32):
    """Quantize a BF16 tensor to FP8 E4M3FN with per-block E8M0FNU scale factors.

    Args:
        x_bf16: (batch, seqlen, nheads, hdim) in BF16
        sf_vec_size: block size for scale factors (default 32)

    Returns:
        x_fp8: (batch, seqlen, nheads, hdim) in FP8 E4M3FN
        sf_e8m0: (batch, seqlen, nheads, sf_k) in E8M0FNU
        scale_float: (batch, seqlen, nheads, sf_k) float scale factors for dequantization
    """
    batch, seqlen, nheads, hdim = x_bf16.shape
    sf_k = _ceil_div(hdim, sf_vec_size)
    hdim_padded = sf_k * sf_vec_size

    if hdim_padded > hdim:
        x_padded = torch.zeros(
            batch, seqlen, nheads, hdim_padded, dtype=x_bf16.dtype, device=x_bf16.device
        )
        x_padded[..., :hdim] = x_bf16
    else:
        x_padded = x_bf16

    x_blocks = x_padded.reshape(batch, seqlen, nheads, sf_k, sf_vec_size)
    absmax = x_blocks.abs().amax(dim=-1)

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    absmax_clamped = absmax.clamp(min=1e-12)
    log2_scale = torch.ceil(torch.log2(absmax_clamped / fp8_max)).clamp(min=-127, max=127)
    scale_float = torch.pow(2.0, log2_scale)

    scale_expanded = scale_float.unsqueeze(-1).expand_as(x_blocks)
    x_scaled = (x_blocks / scale_expanded).clamp(-fp8_max, fp8_max)
    x_fp8_padded = x_scaled.reshape(batch, seqlen, nheads, hdim_padded).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8_padded[..., :hdim].contiguous()

    sf_biased = (log2_scale + 127).to(torch.uint8)
    sf_e8m0 = sf_biased.contiguous().view(torch.float8_e8m0fnu).reshape(batch, seqlen, nheads, sf_k)

    return x_fp8, sf_e8m0, scale_float


def dequantize_fp8_with_sf(x_fp8, scale_float, sf_vec_size=32):
    """Dequantize FP8 tensor back to float using per-block scale factors.

    Args:
        x_fp8: (batch, seqlen, nheads, hdim) in FP8 E4M3FN
        scale_float: (batch, seqlen, nheads, sf_k) float scale factors (2^exponent)
        sf_vec_size: block size for scale factors

    Returns:
        x_float: (batch, seqlen, nheads, hdim) in float32
    """
    batch, seqlen, nheads, hdim = x_fp8.shape
    sf_k = scale_float.shape[-1]
    hdim_padded = sf_k * sf_vec_size

    x_float = x_fp8.float()
    if hdim_padded > hdim:
        x_padded = torch.zeros(
            batch, seqlen, nheads, hdim_padded, dtype=torch.float32, device=x_fp8.device
        )
        x_padded[..., :hdim] = x_float
    else:
        x_padded = x_float

    x_blocks = x_padded.reshape(batch, seqlen, nheads, sf_k, sf_vec_size)
    scale_expanded = scale_float.unsqueeze(-1).expand_as(x_blocks)
    x_dequant = (x_blocks * scale_expanded).reshape(batch, seqlen, nheads, hdim_padded)
    return x_dequant[..., :hdim]


def interleave_sf(sf, sf_vec_size):
    """Interleave a K-major scale factor tensor into the BlockScaledBasicChunk atom layout.

    Input:  sf with shape (batch, seqlen, nheads, sf_k) where sf_k = ceil(hdim / sf_vec_size)
    Output: physically contiguous as (batch, nheads, REST_M, REST_K, 32, 4, 4),
            reshaped to (total_sf_elements,) for passing as a raw buffer.
            The kernel endows this with the appropriate cute layout via tile_atom_to_shape_SF.
    """
    batch, seqlen, nheads, sf_k = sf.shape

    seqlen_padded = ((seqlen + 127) // 128) * 128
    sf_k_padded = ((sf_k + 3) // 4) * 4
    rest_m = seqlen_padded // 128
    rest_k = sf_k_padded // 4

    sf_work = sf.permute(0, 2, 1, 3)  # (batch, nheads, seqlen, sf_k)

    if seqlen_padded != seqlen or sf_k_padded != sf_k:
        sf_padded = torch.zeros(
            batch,
            nheads,
            seqlen_padded,
            sf_k_padded,
            dtype=sf.dtype,
            device=sf.device,
        )
        sf_padded[:, :, :seqlen, :sf_k] = sf_work
    else:
        sf_padded = sf_work.contiguous()

    # Decompose M -> (REST_M, 4, 32), SF_K -> (REST_K, 4)
    sf_decomp = sf_padded.reshape(batch, nheads, rest_m, 4, 32, rest_k, 4)
    # Permute to (batch, nheads, REST_M, REST_K, 32, 4, 4) and make contiguous
    return sf_decomp.permute(0, 1, 2, 5, 4, 3, 6).contiguous()


@cute.jit
def dequant_rowwise(
    sX_q: cute.Tensor,
    sSFX: cute.Tensor,
    sX_dequant: cute.Tensor,
    tidx: int,
    num_threads: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int] = 32,
):
    """
    Dequantize blockscaled FP8 tensor in SMEM.
    """
    nrows = cute.size(sX_q.shape[0])
    hdim = cute.size(sX_q.shape[1])
    num_sf = hdim // sf_vec_size
    # Read E8M0 as raw bytes: value = 2^(byte - 127)
    sSFX_u8 = cute.make_tensor(cute.recast_ptr(sSFX.iterator, dtype=cutlass.Uint8), sSFX.layout)
    # Tile hdim into (sf_vec_size, num_sf) blocks:
    # (nrows, hdim):(hdim, 1) -> ((nrows, 1), (sf_vec_size, num_sf)):((hdim, 0), (1, sf_vec_size))
    sX_q_tiled = cute.logical_divide(sX_q, (nrows, sf_vec_size))
    sX_dequant_tiled = cute.logical_divide(sX_dequant, (nrows, sf_vec_size))

    for row in cutlass.range(tidx, nrows, num_threads):
        for sf_idx in cutlass.range(num_sf, unroll_full=True):
            sf_val = cute.exp2(cutlass.Float32(sSFX_u8[row, sf_idx]) - 127.0)
            src = sX_q_tiled[row, (None, sf_idx)]
            dst = sX_dequant_tiled[row, (None, sf_idx)]
            frg = cute.make_fragment_like(src)
            cute.autovec_copy(src, frg)
            frg_out = cute.make_fragment_like(dst)
            frg_out.store((frg.load().to(cutlass.Float32) * sf_val).to(cutlass.BFloat16))
            cute.autovec_copy(frg_out, dst)
