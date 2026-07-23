"""Verify that explicit final-layer extraction matches the original encoder path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import torch

from mqids.data import infer_channel_metadata, load_sensor_names
from mqids.moirai_tokenizer import FrozenMoiraiTokenizer
from mqids.paths import WADI_SENSOR_NAMES, WADI_TRAIN_X, moirai_model_path, require_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-length", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    require_files(WADI_TRAIN_X, WADI_SENSOR_NAMES, moirai_model_path("base") / "config.json")
    train_x = np.load(WADI_TRAIN_X, mmap_mode="r")
    names = load_sensor_names(WADI_SENSOR_NAMES, train_x.shape[1])
    metadata = infer_channel_metadata(train_x, names)
    endpoint = max(127, args.window_length - 1)
    window = np.asarray(
        train_x[endpoint - args.window_length + 1 : endpoint + 1, metadata.active_indices],
        dtype=np.float32,
    )
    batch = torch.from_numpy(np.ascontiguousarray(window)).unsqueeze(0).to(device)
    original = FrozenMoiraiTokenizer.from_pretrained(
        moirai_model_path("base"),
        window_length=args.window_length,
        patch_size=args.window_length,
        target_dim=len(metadata.active_indices),
        encoder_layer=None,
        device=device,
    )
    explicit = FrozenMoiraiTokenizer.from_pretrained(
        moirai_model_path("base"),
        window_length=args.window_length,
        patch_size=args.window_length,
        target_dim=len(metadata.active_indices),
        encoder_layer=12,
        device=device,
    )
    original_tokens = original(batch)
    explicit_tokens = explicit(batch)
    max_abs_diff = (original_tokens - explicit_tokens).abs().max().item()
    if original_tokens.shape != (1, len(metadata.active_indices), 768):
        raise AssertionError(f"Unexpected token shape: {tuple(original_tokens.shape)}")
    if not torch.allclose(original_tokens, explicit_tokens, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"Explicit layer 12 differs from original final output: {max_abs_diff}")
    if any(parameter.requires_grad for parameter in explicit.parameters()):
        raise AssertionError("Explicit-layer MOIRAI is not fully frozen")
    print(
        f"MOIRAI layer equivalence OK: L={args.window_length}, "
        f"shape={tuple(explicit_tokens.shape)}, max_abs_diff={max_abs_diff:.3e}"
    )


if __name__ == "__main__":
    main()
