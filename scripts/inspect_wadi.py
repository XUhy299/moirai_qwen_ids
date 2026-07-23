"""Audit WADI channels and the single-event support/query split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np

from mqids.config import load_config
from mqids.data import (
    infer_channel_metadata,
    load_sensor_names,
    split_single_attack_support_query,
)
from mqids.paths import WADI_SENSOR_NAMES, WADI_TRAIN_X, WADI_VAL_X, WADI_VAL_Y, require_files


def parse_args() -> argparse.Namespace:
    default_config = PACKAGE_ROOT / "configs" / "wadi_qwen3_06b.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_files(WADI_TRAIN_X, WADI_VAL_X, WADI_VAL_Y, WADI_SENSOR_NAMES)
    train_x = np.load(WADI_TRAIN_X, mmap_mode="r")
    val_x = np.load(WADI_VAL_X, mmap_mode="r")
    val_y = np.load(WADI_VAL_Y, mmap_mode="r")
    if len(val_x) != len(val_y):
        raise ValueError("WADI validation X/Y lengths differ")
    names = load_sensor_names(WADI_SENSOR_NAMES, train_x.shape[1])
    metadata = infer_channel_metadata(train_x, names)
    split = split_single_attack_support_query(
        val_y,
        window_length=config.window_length,
        support_fraction=config.support_fraction,
        guard=config.split_guard,
    )
    type_counts = {
        type_name: metadata.type_ids.count(type_id)
        for type_id, type_name in enumerate(metadata.type_names)
    }
    print(f"train={train_x.shape} val={val_x.shape} labels={val_y.shape}")
    print(
        f"channels: all={train_x.shape[1]} active={len(metadata.active_indices)} "
        f"constant={len(metadata.constant_indices)} types={type_counts}"
    )
    print(
        "validation attack: "
        f"[{split.attack_start}, {split.attack_end_exclusive}); "
        f"support={len(split.support_endpoints)} query={len(split.query_endpoints)}; "
        f"guard=[{split.excluded_start}, {split.excluded_end_exclusive})"
    )
    print("warning: support/query are portions of the same attack event, not independent events")
    if args.output is not None:
        metadata.save(args.output)
        print(f"saved channel metadata to {args.output}")


if __name__ == "__main__":
    main()
