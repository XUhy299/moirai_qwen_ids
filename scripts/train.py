"""Train on WADI normal data plus a support portion of one validation attack.

This script deliberately never opens the official WADI test arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import platform
import re
import sys
from collections import Counter
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from mqids.config import PROMPT_VARIANTS, load_config
from mqids.data import (
    WadiWindowDataset,
    SyntheticWindowDataset,
    infer_channel_metadata,
    infer_discrete_state_vocabulary,
    load_sensor_names,
    sample_endpoints,
    split_single_attack_support_query,
)
from mqids.factory import build_model
from mqids.losses import DualObjectiveLoss
from mqids.model import count_trainable_parameters
from mqids.semantics import VariableSemanticMap
from mqids.paths import (
    OUTPUT_ROOT,
    WADI_SENSOR_NAMES,
    WADI_SCALER,
    WADI_TRAIN_X,
    WADI_VAL_X,
    WADI_VAL_Y,
    guard_development_paths,
    package_relative_path,
    require_files,
)
from mqids.training import (
    evaluate_classifier,
    save_json,
    save_trainable_checkpoint,
    set_seed,
    train_one_epoch,
    trainable_parameters,
    write_run_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "wadi_qwen3_06b.json",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--use-synthetic-anomalies",
        action="store_true",
        help="Add a validated Full five-type synthetic anomaly package to training.",
    )
    parser.add_argument(
        "--synthetic-data-dir",
        type=Path,
        default=None,
        help=(
            "Synthetic package directory. By default, select the local seed-2026 package "
            "matching the resolved window length."
        ),
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=None,
        help=(
            "Number of synthetic windows to use. Default: match the real support-anomaly "
            "count; pass 5000 explicitly to use the full generated package."
        ),
    )
    parser.add_argument("--projector", choices=("linear", "direct", "reprogramming"), default=None)
    parser.add_argument("--vocab-loss-weight", type=float, default=None)
    parser.add_argument("--classifier-loss-weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--baseline-hidden-dim", type=int, default=None)
    parser.add_argument("--baseline-layers", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the physical training batch size after a measured capacity probe.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default=None)
    parser.add_argument(
        "--moirai-layer",
        type=int,
        default=None,
        help="Use a 1-based MOIRAI Transformer Encoder layer (Base supports 1..12).",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        choices=(8, 16, 32, 64, 128),
        default=None,
        help="Set window_length and MOIRAI patch_size together.",
    )
    parser.add_argument(
        "--discrete-to-text",
        action="store_true",
        help="Convert binary/low-cardinality variables to natural-language summaries in the prompt prefix.",
    )
    parser.add_argument(
        "--dtt-semantic-style",
        choices=("compact", "full"),
        default=None,
        help="Use compact or full natural-language variable descriptions in DTT mode.",
    )
    parser.add_argument(
        "--full-run-authorized",
        action="store_true",
        help=(
            "Required for every non-smoke, non-prepare training run. Pass only after the user "
            "has explicitly authorized the full experiment."
        ),
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and save the data protocol without loading MOIRAI or Qwen.",
    )
    execution.add_argument(
        "--smoke",
        action="store_true",
        help="Run one epoch on four train and four development windows.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run_name(run_name: str) -> str:
    """Keep every run artifact inside outputs/ and make names shell-portable."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError(
            "--run-name must start with an ASCII letter or digit and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    return run_name


def main() -> None:
    args = parse_args()
    validate_run_name(args.run_name)
    config = load_config(args.config)
    replacements = {}
    if args.projector is not None:
        replacements["projector"] = args.projector
    if args.vocab_loss_weight is not None:
        replacements["vocab_loss_weight"] = args.vocab_loss_weight
    if args.classifier_loss_weight is not None:
        replacements["classifier_loss_weight"] = args.classifier_loss_weight
    if args.seed is not None:
        replacements["seed"] = args.seed
    if args.baseline_hidden_dim is not None:
        replacements["baseline_hidden_dim"] = args.baseline_hidden_dim
    if args.baseline_layers is not None:
        replacements["baseline_layers"] = args.baseline_layers
    if args.batch_size is not None:
        replacements["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        replacements["eval_batch_size"] = args.eval_batch_size
    if args.moirai_layer is not None:
        replacements["moirai_encoder_layer"] = args.moirai_layer
    if args.discrete_to_text:
        replacements["discrete_to_text"] = True
    if args.dtt_semantic_style is not None:
        replacements["dtt_semantic_style"] = args.dtt_semantic_style
    if args.prompt_variant is not None:
        replacements["prompt_variant"] = args.prompt_variant
    if args.window_length is not None:
        replacements["window_length"] = args.window_length
        replacements["patch_size"] = args.window_length
    if replacements:
        config = replace(config, **replacements)
    if args.dtt_semantic_style is not None and not config.discrete_to_text:
        raise ValueError("--dtt-semantic-style requires --discrete-to-text")
    if not args.use_synthetic_anomalies and (
        args.synthetic_data_dir is not None or args.synthetic_samples is not None
    ):
        raise ValueError(
            "--synthetic-data-dir/--synthetic-samples require --use-synthetic-anomalies"
        )
    if args.synthetic_samples is not None and args.synthetic_samples < 1:
        raise ValueError("--synthetic-samples must be positive")
    if not args.smoke and not args.prepare_only and not args.full_run_authorized:
        raise PermissionError(
            "Full training is locked. Obtain explicit user authorization, then rerun with "
            "--full-run-authorized. Smoke and prepare-only runs remain available without it."
        )
    set_seed(config.seed)
    device = resolve_device(args.device)
    unguarded_np_load = np.load

    def guarded_development_np_load(file, *load_args, **load_kwargs):
        if isinstance(file, (str, os.PathLike)):
            guard_development_paths(Path(file))
        return unguarded_np_load(file, *load_args, **load_kwargs)

    # Fail closed even if a future code path accidentally calls np.load on the
    # locked WADI test arrays without adding that path to development_paths.
    np.load = guarded_development_np_load
    development_paths = (WADI_TRAIN_X, WADI_VAL_X, WADI_VAL_Y, WADI_SENSOR_NAMES, WADI_SCALER)
    guard_development_paths(*development_paths)
    require_files(*development_paths)
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x = np.load(WADI_TRAIN_X, mmap_mode="r")
    val_x = np.load(WADI_VAL_X, mmap_mode="r")
    val_y = np.load(WADI_VAL_Y, mmap_mode="r")
    if len(val_x) != len(val_y):
        raise ValueError("WADI validation X/Y lengths differ")
    names = load_sensor_names(WADI_SENSOR_NAMES, train_x.shape[1])
    metadata = infer_channel_metadata(train_x, names)
    if not config.drop_train_constant_channels:
        raise NotImplementedError(
            "The first protocol intentionally removes train-constant channels; "
            "all-channel ablation will be added as a separate explicit config."
        )
    metadata.save(output_dir / "channel_metadata.json")
    discrete_vocabulary = None
    semantic_map = None
    if config.discrete_to_text:
        with WADI_SCALER.open("rb") as handle:
            scaler = pickle.load(handle)
        discrete_vocabulary = infer_discrete_state_vocabulary(
            train_x,
            metadata,
            source=(
                f"normal_train={package_relative_path(WADI_TRAIN_X)}; "
                f"scaler={package_relative_path(WADI_SCALER)}"
            ),
            scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
            scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        )
        discrete_vocabulary.save(output_dir / "discrete_state_vocabulary.json")
        semantic_map = VariableSemanticMap.from_names(
            metadata.active_names,
            style=config.dtt_semantic_style,
        )
        semantic_map.save(output_dir / "variable_semantics.json")
    split = split_single_attack_support_query(
        val_y,
        window_length=config.window_length,
        support_fraction=config.support_fraction,
        guard=config.split_guard,
    )

    anomaly_endpoints = split.support_endpoints[:: config.anomaly_train_stride]
    normal_candidates = np.arange(
        config.common_min_endpoint,
        len(train_x),
        config.normal_train_stride,
        dtype=np.int64,
    )
    normal_count = min(
        config.max_normal_train_windows,
        config.normal_to_anomaly_ratio * len(anomaly_endpoints),
    )
    normal_endpoints = sample_endpoints(normal_candidates, normal_count, seed=config.seed)
    if args.smoke:
        normal_endpoints = normal_endpoints[:2]
        anomaly_endpoints = anomaly_endpoints[:2]
    normal_dataset = WadiWindowDataset(
        train_x,
        normal_endpoints,
        config.window_length,
        metadata.active_indices,
        fixed_label=0,
    )
    anomaly_dataset = WadiWindowDataset(
        val_x,
        anomaly_endpoints,
        config.window_length,
        metadata.active_indices,
        labels=val_y,
    )
    synthetic_package = None
    synthetic_dataset = None
    synthetic_indices = np.empty(0, dtype=np.int64)
    synthetic_selected_operator_counts: dict[str, int] = {}
    if args.use_synthetic_anomalies:
        synthetic_dir = args.synthetic_data_dir or (
            PACKAGE_ROOT
            / "synthetic_data"
            / f"WADI-CLEAN_X_train_full_l{config.window_length}_seed2026"
        )
        train_x_sha256 = sha256_file(WADI_TRAIN_X)
        synthetic_package = SyntheticWindowDataset(
            synthetic_dir,
            expected_window_length=config.window_length,
            expected_active_indices=metadata.active_indices,
            expected_type_ids=metadata.type_ids,
            expected_train_x_sha256=train_x_sha256,
            expected_train_length=len(train_x),
        )
        requested_synthetic = (
            len(anomaly_dataset) if args.synthetic_samples is None else args.synthetic_samples
        )
        if requested_synthetic > len(synthetic_package):
            raise ValueError(
                f"Requested {requested_synthetic} synthetic windows, but package has "
                f"{len(synthetic_package)}"
            )
        selected_synthetic = min(requested_synthetic, 2) if args.smoke else requested_synthetic
        synthetic_indices = sample_endpoints(
            np.arange(len(synthetic_package), dtype=np.int64),
            selected_synthetic,
            seed=config.seed,
        )
        synthetic_dataset = Subset(synthetic_package, synthetic_indices.tolist())
        code_to_operator = {
            int(code): name
            for name, code in synthetic_package.summary["operator_codes"].items()
        }
        synthetic_selected_operator_counts = dict(
            sorted(
                Counter(
                    code_to_operator[int(code)]
                    for code in synthetic_package.operator_codes[synthetic_indices]
                ).items()
            )
        )
    train_parts = [normal_dataset, anomaly_dataset]
    if synthetic_dataset is not None:
        train_parts.append(synthetic_dataset)
    train_dataset = ConcatDataset(tuple(train_parts))

    before_attack = np.arange(
        config.common_min_endpoint,
        split.attack_start,
        config.dev_stride,
        dtype=np.int64,
    )
    query_and_after = np.arange(
        split.query_endpoints[0],
        len(val_y),
        config.dev_stride,
        dtype=np.int64,
    )
    dev_endpoints = np.concatenate((before_attack, query_and_after))
    if args.smoke:
        dev_endpoints = np.array(
            [
                before_attack[0],
                before_attack[-1],
                split.query_endpoints[0],
                split.query_endpoints[1],
            ],
            dtype=np.int64,
        )
    dev_dataset = WadiWindowDataset(
        val_x,
        dev_endpoints,
        config.window_length,
        metadata.active_indices,
        labels=val_y,
    )
    effective_train_batch_size = 1 if args.smoke else config.batch_size
    effective_eval_batch_size = 1 if args.smoke else config.eval_batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=effective_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    protocol = {
        "description": (
            "single-attack-event supervised transfer with train-only synthetic anomalies"
            if synthetic_dataset is not None
            else "single-attack-event supervised transfer"
        ),
        "smoke": bool(args.smoke),
        "opened_data_files": {
            "train_x": package_relative_path(WADI_TRAIN_X),
            "validation_x": package_relative_path(WADI_VAL_X),
            "validation_y": package_relative_path(WADI_VAL_Y),
            "sensor_names": package_relative_path(WADI_SENSOR_NAMES),
            "train_fitted_scaler": package_relative_path(WADI_SCALER),
            "synthetic_anomaly_package": (
                package_relative_path(synthetic_package.package_dir)
                if synthetic_package is not None
                else None
            ),
            "test_x": None,
            "test_y": None,
        },
        "locked_test_files_opened": [],
        "train_normal_windows": len(normal_dataset),
        "train_anomaly_windows": len(anomaly_dataset)
        + (len(synthetic_dataset) if synthetic_dataset is not None else 0),
        "train_real_anomaly_windows": len(anomaly_dataset),
        "train_synthetic_anomaly_windows": (
            len(synthetic_dataset) if synthetic_dataset is not None else 0
        ),
        "train_total_windows": len(train_dataset),
        "dev_windows": len(dev_dataset),
        "effective_train_batch_size": effective_train_batch_size,
        "effective_eval_batch_size": effective_eval_batch_size,
        "normal_endpoint_sha256": hashlib.sha256(normal_endpoints.tobytes()).hexdigest(),
        "anomaly_endpoint_sha256": hashlib.sha256(anomaly_endpoints.tobytes()).hexdigest(),
        "synthetic_selection_sha256": (
            hashlib.sha256(synthetic_indices.tobytes()).hexdigest()
            if synthetic_dataset is not None
            else None
        ),
        "dev_endpoint_sha256": hashlib.sha256(dev_endpoints.tobytes()).hexdigest(),
        "dev_stride": config.dev_stride,
        "formal_endpoint_stride": 1,
        "common_min_endpoint": config.common_min_endpoint,
        "attack_run": [split.attack_start, split.attack_end_exclusive],
        "support_endpoint_range": [int(anomaly_endpoints.min()), int(anomaly_endpoints.max())],
        "query_endpoint_range": [int(split.query_endpoints.min()), int(split.query_endpoints.max())],
        "excluded_guard": [split.excluded_start, split.excluded_end_exclusive],
        "endpoint_label_rule": "Y[window_end]",
        "classifier_loss_weight": config.classifier_loss_weight,
        "vocab_loss_weight": config.vocab_loss_weight,
        "moirai_encoder_layer": config.moirai_encoder_layer,
        "prompt_variant": config.prompt_variant,
        "discrete_to_text": config.discrete_to_text,
        "prompt_layout": (
            "variable_aligned_semantic_chat_dtt_v3"
            if config.discrete_to_text
            else "legacy_token_block"
        ),
        "qwen_chat_template_applied": bool(config.discrete_to_text),
        "qwen_thinking_enabled": False if config.discrete_to_text else None,
        "dtt_semantic_style": config.dtt_semantic_style if config.discrete_to_text else None,
        "full_run_authorized": bool(args.full_run_authorized),
        "synthetic_anomalies": (
            {
                "enabled": True,
                "package_available_windows": len(synthetic_package),
                "requested_windows": (
                    len(anomaly_dataset)
                    if args.synthetic_samples is None
                    else args.synthetic_samples
                ),
                "selected_windows": len(synthetic_dataset),
                "smoke_cap_applied": bool(
                    args.smoke
                    and (
                        len(anomaly_dataset)
                        if args.synthetic_samples is None
                        else args.synthetic_samples
                    )
                    > len(synthetic_dataset)
                ),
                "package_input_x_sha256": synthetic_package.summary["input_x_sha256"],
                "package_operator_counts": synthetic_package.summary["operator_counts"],
                "selected_operator_counts": synthetic_selected_operator_counts,
                "endpoint_label_rule": "point_masks[:, -1] == true",
                "selection_index_sha256": hashlib.sha256(
                    synthetic_indices.tobytes()
                ).hexdigest(),
            }
            if synthetic_dataset is not None
            else {"enabled": False}
        ),
        "variable_semantics_sha256": semantic_map.sha256() if semantic_map is not None else None,
        "discrete_state_vocabulary_sha256": (
            discrete_vocabulary.sha256() if discrete_vocabulary is not None else None
        ),
        "limitation": "Support and query are chronological portions of the same attack event.",
    }
    save_json(protocol, output_dir / "protocol.json")
    save_json(config.as_dict(), output_dir / "config.json")
    try:
        transformers_version = version("transformers")
    except PackageNotFoundError:
        transformers_version = None
    environment = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    save_json(environment, output_dir / "environment.json")
    if args.prepare_only:
        write_run_markdown(
            output_dir,
            config=config.as_dict(),
            protocol=protocol,
            environment=environment,
            history=[],
            status="prepare_only",
        )
        print(f"prepare-only OK: train={len(train_dataset)} dev={len(dev_dataset)}")
        print(f"protocol={output_dir / 'protocol.json'}")
        return

    model = build_model(
        config,
        active_variable_count=len(metadata.active_indices),
        variable_type_ids=torch.tensor(metadata.type_ids, dtype=torch.long),
        device=device,
        continuous_indices=metadata.continuous_indices,
        discrete_indices=metadata.discrete_indices,
        active_names=metadata.active_names,
        active_descriptions=semantic_map.descriptions if semantic_map is not None else (),
        discrete_vocabulary=discrete_vocabulary,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    objective = DualObjectiveLoss(
        vocab_weight=config.vocab_loss_weight,
        classifier_weight=config.classifier_loss_weight,
    )
    print(f"device={device} active_variables={len(metadata.active_indices)}")
    print(f"trainable_parameters={count_trainable_parameters(model):,}")
    print(f"train={len(train_dataset)} dev={len(dev_dataset)}")

    history: list[dict[str, object]] = []
    selection_metric = "vocab_pr_auc" if config.classifier_loss_weight == 0 else "pr_auc"
    best_selection_score = -float("inf")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epochs = 1 if args.smoke else config.epochs
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, objective, device)
        dev_metrics = evaluate_classifier(model, dev_loader, device)
        row: dict[str, object] = {"epoch": epoch, "train": train_metrics, "dev": dev_metrics}
        if device.type == "cuda":
            row["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
        selection_score = float(dev_metrics[selection_metric])
        row["selection_metric"] = selection_metric
        row["selection_score"] = selection_score
        history.append(row)
        save_json({"epochs": history}, output_dir / "history.json")
        write_run_markdown(
            output_dir,
            config=config.as_dict(),
            protocol=protocol,
            environment=environment,
            history=history,
            status="completed" if epoch == epochs else "in_progress",
        )
        print(
            f"epoch={epoch} loss={train_metrics['loss']:.4f} "
            f"dev_{selection_metric}={selection_score:.4f}"
        )
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            checkpoint_metrics = dict(dev_metrics)
            checkpoint_metrics["selection_metric"] = selection_metric
            checkpoint_metrics["selection_score"] = selection_score
            save_trainable_checkpoint(
                model,
                output_dir / "best_trainable.pt",
                config=config.as_dict(),
                metrics=checkpoint_metrics,
            )
    print(f"best_dev_{selection_metric}={best_selection_score:.4f}; outputs={output_dir}")


if __name__ == "__main__":
    main()
