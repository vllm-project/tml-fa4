from pathlib import Path


def test_attention_backward_kernel_files_are_removed():
    repo_root = Path(__file__).resolve().parents[1]
    cute_root = repo_root / "flash_attn" / "cute"
    removed = [
        *cute_root.glob("flash_bwd*.py"),
        *cute_root.glob("sm100_hd256_2cta_fmha_backward*.py"),
        cute_root / "unshearing_bias.py",
    ]
    assert [path for path in removed if path.exists()] == []
