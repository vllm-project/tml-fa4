# FlashAttention-4 Forward

Forward-only FlashAttention-4 CuTeDSL kernels from `thinking-machines-lab/colfax_collab`.

This repo keeps the FA4 forward attention paths, including relative bias shearing, split-KV combine, scheduler metadata, mask/score modifiers, block-sparse support, and the SM100 hd256 forward kernel.

## Install

```sh
pip install -e ".[dev,cu13]"
```

## Use

```python
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

out, lse = flash_attn_func(q, k, v, causal=True)
```

The public attention functions return forward outputs only. Inputs may require gradients, but returned tensors are not wired to FA4 backward kernels.

## Ragged Relative Bias

Thread one extra tensor through the attention backend SPI and pass it as `rel_bias`:

```python
# r: [total_q, n_q_heads, d_rel], flattened in the same order as q.
rel_bias = torch.einsum("thd,de->the", r, rel_proj).contiguous()

from flash_attn.cute import flash_attn_varlen_func

out, lse = flash_attn_varlen_func(
    q,  # [total_q, n_q_heads, head_dim]
    k,  # [total_k, n_kv_heads, head_dim]
    v,  # [total_k, n_kv_heads, head_dim_v]
    rel_bias=rel_bias,
    cu_seqlens_q=cu_seqlens_q,  # int32 CUDA tensor, [batch + 1], last == total_q
    cu_seqlens_k=cu_seqlens_k,  # int32 CUDA tensor, [batch + 1], last == total_k
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    causal=True,
    return_lse=True,
    out=preallocated_out,  # optional, same shape/dtype/device as the output
)
```

Argument contract:

- `q`: `[total_q, n_q_heads, head_dim]`
- `k`: `[total_k, n_kv_heads, head_dim]`
- `v`: `[total_k, n_kv_heads, head_dim_v]`
- `cu_seqlens_q`, `cu_seqlens_k`: contiguous int32 CUDA tensors of shape `[batch + 1]`
- `max_seqlen_q`, `max_seqlen_k`: max per-sequence lengths from the corresponding `cu_seqlens`
- `rel_bias`: `[total_q, n_q_heads, rel_extent]`; no batch dimension; same flattened order, CUDA device, and dtype as `q`; `rel_extent` is inferred from this shape and must be a multiple of 128

```python
q_len_b = cu_seqlens_q[b + 1] - cu_seqlens_q[b]
k_len_b = cu_seqlens_k[b + 1] - cu_seqlens_k[b]
d = i + (k_len_b - q_len_b) - j
score[i, j, h] += rel_bias[cu_seqlens_q[b] + i, h, d]  # when 0 <= d < rel_extent
```

In causal self-attention, `d == 0` is the query token's own key position, `d == 1` is the previous key, and keys farther back than `rel_extent - 1` receive no relative-bias contribution.
