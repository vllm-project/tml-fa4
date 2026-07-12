#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


RUNNER = r"""
import sys
import torch

from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

torch.manual_seed(12345)
device = "cuda"
dtype = torch.bfloat16
results = {}


def keep(name, tensors):
    for idx, tensor in enumerate(tensors):
        if tensor is not None:
            results[f"{name}.{idx}"] = tensor.detach().cpu()


with torch.no_grad():
    q = torch.randn(1, 64, 4, 64, device=device, dtype=dtype)
    k = torch.randn(1, 96, 2, 64, device=device, dtype=dtype)
    v = torch.randn(1, 96, 2, 64, device=device, dtype=dtype)
    keep("dense_gqa", flash_attn_func(q, k, v, causal=False, return_lse=True))

    q = torch.randn(1, 128, 4, 128, device=device, dtype=dtype)
    k = torch.randn(1, 128, 4, 128, device=device, dtype=dtype)
    v = torch.randn(1, 128, 4, 128, device=device, dtype=dtype)
    rel_bias = torch.randn(1, 128, 4, 128, device=device, dtype=dtype) * 0.01
    keep("causal_rel_bias", flash_attn_func(q, k, v, rel_bias=rel_bias, causal=True, return_lse=True))

    q = torch.randn(1, 64, 4, 64, device=device, dtype=dtype)
    k = torch.randn(1, 384, 4, 64, device=device, dtype=dtype)
    v = torch.randn(1, 384, 4, 64, device=device, dtype=dtype)
    keep("split_kv", flash_attn_func(q, k, v, causal=False, num_splits=3, return_lse=True))

    cu_q = torch.tensor([0, 64, 128], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 80, 160], device=device, dtype=torch.int32)
    q = torch.randn(128, 4, 128, device=device, dtype=dtype)
    k = torch.randn(160, 4, 128, device=device, dtype=dtype)
    v = torch.randn(160, 4, 128, device=device, dtype=dtype)
    rel_bias = torch.randn(128, 4, 128, device=device, dtype=dtype) * 0.01
    keep(
        "varlen_rel_bias",
        flash_attn_varlen_func(
            q,
            k,
            v,
            rel_bias=rel_bias,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=64,
            max_seqlen_k=80,
            causal=True,
            return_lse=True,
        ),
    )

torch.cuda.synchronize()
torch.save(results, sys.argv[1])
"""


def run_one(source: Path, output: Path, cuda_visible_devices: str | None) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(source) if not existing else f"{source}:{existing}"
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    subprocess.run(
        [sys.executable, "-c", RUNNER, str(output)],
        cwd="/",
        env=env,
        check=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reference", type=Path, default=Path("/work/horace/monorepo/third_party"))
    parser.add_argument("--cuda-visible-devices", default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="fa4_forward_parity_") as tmpdir:
        tmp = Path(tmpdir)
        candidate_out = tmp / "candidate.pt"
        reference_out = tmp / "reference.pt"
        run_one(args.candidate.resolve(), candidate_out, args.cuda_visible_devices)
        run_one(args.reference.resolve(), reference_out, args.cuda_visible_devices)

        candidate = torch.load(candidate_out, map_location="cpu")
        reference = torch.load(reference_out, map_location="cpu")

    failures = []
    for key in sorted(reference):
        if key not in candidate:
            failures.append(f"{key}: missing from candidate")
            continue
        lhs = candidate[key]
        rhs = reference[key]
        equal = torch.equal(lhs, rhs)
        max_abs = (lhs.float() - rhs.float()).abs().max().item()
        mean_abs = (lhs.float() - rhs.float()).abs().mean().item()
        print(f"{key}: equal={equal} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")
        if not equal:
            failures.append(f"{key}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")

    extra = sorted(set(candidate) - set(reference))
    failures.extend(f"{key}: extra in candidate" for key in extra)
    if failures:
        raise SystemExit("Forward parity failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
