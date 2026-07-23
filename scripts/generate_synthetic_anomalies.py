"""Generate a fixed offline Full-mixture synthetic anomaly dataset.

Only the provided normal training X array is opened.  Validation/test arrays
and labels are neither imported nor required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np

from mqids.data import infer_channel_metadata, load_sensor_names
from mqids.paths import WADI_SENSOR_NAMES, WADI_TRAIN_X, require_files
from mqids.synthetic_anomalies import (
    OPERATOR_NAMES,
    FullSyntheticAnomalyGenerator,
    SyntheticAnomalyConfig,
    fit_synthetic_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate five-type mixed anomalies from normal multivariate time series"
    )
    parser.add_argument("--input-x", type=Path, default=WADI_TRAIN_X)
    parser.add_argument("--sensor-names", type=Path, default=WADI_SENSOR_NAMES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--window-length", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stats-sample-size", type=int, default=100_000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in an existing synthetic-data directory",
    )
    return parser.parse_args()


def save_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        entries = list(path.iterdir()) if path.is_dir() else [path]
        if entries and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Use --overwrite explicitly."
            )
        if not path.is_dir():
            raise NotADirectoryError(path)
    path.mkdir(parents=True, exist_ok=True)


def write_dataset_readme(path: Path) -> None:
    text = """# Full five-type synthetic anomaly dataset

This package was generated from the normal training X array only. It contains
endpoint-aligned synthetic anomalies and does not contain validation/test data.

## Arrays

- `synthetic_windows.npy`: float32 `[samples, window, active_channels]` windows.
- `point_masks.npy`: bool `[samples, window]` anomaly locations; the final point is true.
- `channel_masks.npy`: bool `[samples, active_channels]` channels changed per sample.
- `endpoint_labels.npy`: uint8 endpoint labels (all one for this anomaly-only package).
- `source_endpoints.npy`: endpoints used to recover the corresponding clean windows.
- `operator_codes.npy`: integer operator IDs defined in `dataset_summary.json`.

`generation_manifest.jsonl` records the source, operator, affected channels,
segment, severity, optional donor, and retry count for every sample. Original
channel indices and active-channel types are stored in `channel_metadata.json`.
All arrays can be opened with `numpy.load(path, mmap_mode="r")`.
"""
    (path / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    require_files(args.input_x, args.sensor_names)
    output_dir = args.output_dir or (
        PACKAGE_ROOT
        / "synthetic_data"
        / f"{args.input_x.stem}_full_l{args.window_length}_seed{args.seed}"
    )
    output_dir = output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)

    train_x = np.load(args.input_x, mmap_mode="r")
    if train_x.ndim != 2:
        raise ValueError(f"Expected [time, channels], got {train_x.shape}")
    names = load_sensor_names(args.sensor_names, train_x.shape[1])
    metadata = infer_channel_metadata(train_x, names)
    config = SyntheticAnomalyConfig()
    statistics = fit_synthetic_statistics(
        train_x,
        metadata.active_indices,
        metadata.type_ids,
        sample_size=args.stats_sample_size,
        max_absolute_robust_z=config.max_absolute_robust_z,
    )
    generator = FullSyntheticAnomalyGenerator(
        train_x,
        statistics,
        window_length=args.window_length,
        config=config,
    )

    candidates = np.arange(args.window_length - 1, len(train_x), dtype=np.int64)
    if args.num_samples > len(candidates):
        raise ValueError("Requested more unique source windows than are available")
    endpoint_rng, mutation_rng = (
        np.random.default_rng(seed) for seed in np.random.SeedSequence(args.seed).spawn(2)
    )
    endpoints = np.sort(endpoint_rng.choice(candidates, size=args.num_samples, replace=False))

    channel_count = len(metadata.active_indices)
    windows = np.lib.format.open_memmap(
        output_dir / "synthetic_windows.npy",
        mode="w+",
        dtype=np.float32,
        shape=(args.num_samples, args.window_length, channel_count),
    )
    point_masks = np.lib.format.open_memmap(
        output_dir / "point_masks.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(args.num_samples, args.window_length),
    )
    channel_masks = np.lib.format.open_memmap(
        output_dir / "channel_masks.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(args.num_samples, channel_count),
    )
    labels = np.lib.format.open_memmap(
        output_dir / "endpoint_labels.npy", mode="w+", dtype=np.uint8, shape=(args.num_samples,)
    )
    source_endpoints = np.lib.format.open_memmap(
        output_dir / "source_endpoints.npy", mode="w+", dtype=np.int64, shape=(args.num_samples,)
    )
    operator_codes = np.lib.format.open_memmap(
        output_dir / "operator_codes.npy", mode="w+", dtype=np.uint8, shape=(args.num_samples,)
    )

    operator_to_code = {name: index for index, name in enumerate(OPERATOR_NAMES)}
    counts: Counter[str] = Counter()
    manifest_path = output_dir / "generation_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for sample_id, endpoint in enumerate(endpoints):
            sample = generator.generate(int(endpoint), mutation_rng)
            operator = str(sample.metadata["operator"])
            windows[sample_id] = sample.window
            point_masks[sample_id] = sample.point_mask
            channel_masks[sample_id] = sample.channel_mask
            labels[sample_id] = 1
            source_endpoints[sample_id] = endpoint
            operator_codes[sample_id] = operator_to_code[operator]
            counts[operator] += 1
            record = {"sample_id": sample_id, **sample.metadata}
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (sample_id + 1) % 500 == 0 or sample_id + 1 == args.num_samples:
                print(f"generated {sample_id + 1}/{args.num_samples}", flush=True)

    for array in (windows, point_masks, channel_masks, labels, source_endpoints, operator_codes):
        array.flush()

    metadata.save(output_dir / "channel_metadata.json")
    save_json(output_dir / "training_statistics.json", statistics.as_dict())
    save_json(output_dir / "generator_config.json", config.as_dict())
    save_json(
        output_dir / "dataset_summary.json",
        {
            "format_version": 1,
            "method": "full_five_type_mixture",
            "input_x": str(args.input_x.resolve()),
            "input_x_sha256": sha256_file(args.input_x),
            "sensor_names": str(args.sensor_names.resolve()),
            "sensor_names_sha256": sha256_file(args.sensor_names),
            "normal_train_shape": list(train_x.shape),
            "normal_train_dtype": str(train_x.dtype),
            "synthetic_windows_shape": [
                args.num_samples,
                args.window_length,
                channel_count,
            ],
            "synthetic_windows_dtype": "float32",
            "label_rule": "endpoint; point_masks[:, -1] is always true",
            "active_channels": channel_count,
            "continuous_channels": len(metadata.continuous_indices),
            "discrete_channels": len(metadata.discrete_indices),
            "operator_codes": operator_to_code,
            "operator_counts": dict(sorted(counts.items())),
            "seed": args.seed,
            "stats_sample_size": min(args.stats_sample_size, len(train_x)),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "validation_or_test_arrays_opened": False,
        },
    )
    write_dataset_readme(output_dir)
    print(f"saved synthetic dataset to {output_dir}")


if __name__ == "__main__":
    main()
