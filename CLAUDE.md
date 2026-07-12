# CLAUDE.md

This repository is the forward-only FA4 split from `thinking-machines-lab/colfax_collab`.

## Project Overview

FlashAttention-4 forward kernels are implemented in Python using CuTeDSL. Runtime compilation targets Hopper, Blackwell, and related architectures through the same forward modules as the Colfax source:

- `flash_attn/cute/flash_fwd.py`
- `flash_attn/cute/flash_fwd_sm90.py`
- `flash_attn/cute/flash_fwd_sm100.py`
- `flash_attn/cute/flash_fwd_sm120.py`
- `flash_attn/cute/flash_fwd_combine.py`
- `flash_attn/cute/sm100_hd256_2cta_fmha_forward.py`

Attention backward kernels and autograd backward wiring are intentionally absent.

## Build And Install

```bash
pip install -e ".[dev,cu13]"
```

Dependencies: `nvidia-cutlass-dsl>=4.4.2`, `torch`, `einops`, `apache-tvm-ffi`, `torch-c-dlpack-ext`, and `quack-kernels>=0.4.0`.

## Tests

```bash
pytest tests
python tools/compare_forward_with_monorepo.py --reference /work/horace/monorepo/third_party --cuda-visible-devices 0
```

The parity script runs candidate and monorepo reference implementations in separate subprocesses and compares dense, split-KV, relative-bias, and varlen relative-bias forward outputs exactly.

## Public API

`flash_attn.cute` exports:

- `flash_attn_func`
- `flash_attn_varlen_func`

Both functions return forward outputs only. Backward-only keyword arguments retained for source compatibility are ignored.
