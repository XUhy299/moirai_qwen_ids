"""Verify a Linux environment and probe safe batch sizes for the real full DTT prompt.

This is an engineering smoke/capacity check, not an experiment: it uses one
normal-training window repeated within a batch and never opens WADI test data.
It first runs the official four-window ``train.py --smoke`` path, which is
deliberately fixed at physical batch size one, then probes full forward,
backward, and optimizer steps at explicit candidate physical batch sizes.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from packaging.version import Version


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from mqids.config import load_config
from mqids.data import infer_channel_metadata, infer_discrete_state_vocabulary, load_sensor_names
from mqids.factory import build_model
from mqids.losses import DualObjectiveLoss
from mqids.paths import WADI_SCALER, WADI_SENSOR_NAMES, WADI_TRAIN_X, require_files
from mqids.semantics import VariableSemanticMap
from mqids.training import trainable_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-sizes", default="1,2,4,8", help="Ascending physical batch candidates.")
    parser.add_argument("--window-length", type=int, choices=(8, 16, 32, 64, 128), default=64)
    parser.add_argument("--moirai-layer", type=int, default=12)
    parser.add_argument(
        "--headroom-fraction",
        type=float,
        default=0.15,
        help="Reserve this fraction of total GPU memory when recommending a batch size.",
    )
    parser.add_argument("--skip-official-smoke", action="store_true")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda", torch.cuda.current_device())
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def parse_batch_sizes(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("--batch-sizes must contain positive integers")
    if values != sorted(set(values)):
        raise ValueError("--batch-sizes must be unique and in ascending order")
    return values


def environment_report(device: torch.device) -> dict[str, object]:
    report: dict[str, object] = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": package_version("transformers"),
        "uni2ts": package_version("uni2ts"),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_mb": round(props.total_memory / 1024**2, 1),
                "gpu_capability": f"{props.major}.{props.minor}",
            }
        )
    return report


def run_official_smoke(args: argparse.Namespace, device: torch.device, run_name: str) -> int:
    command = [
        sys.executable,
        str(PACKAGE_ROOT / "scripts" / "train.py"),
        "--run-name",
        run_name,
        "--smoke",
        "--device",
        device.type,
        "--window-length",
        str(args.window_length),
        "--moirai-layer",
        str(args.moirai_layer),
        "--projector",
        "direct",
        "--vocab-loss-weight",
        "0",
        "--discrete-to-text",
        "--dtt-semantic-style",
        "full",
    ]
    print("Running official full-prompt smoke:\n  " + " ".join(command))
    return subprocess.run(command, cwd=PACKAGE_ROOT, check=False).returncode


def build_full_prompt_model(args: argparse.Namespace, device: torch.device):
    require_files(WADI_TRAIN_X, WADI_SENSOR_NAMES, WADI_SCALER)
    train_x = np.load(WADI_TRAIN_X, mmap_mode="r")
    names = load_sensor_names(WADI_SENSOR_NAMES, train_x.shape[1])
    metadata = infer_channel_metadata(train_x, names)
    with WADI_SCALER.open("rb") as handle:
        scaler = pickle.load(handle)
    vocabulary = infer_discrete_state_vocabulary(
        train_x,
        metadata,
        source="normal_train=data/wadi/WADI-CLEAN_X_train.npy; scaler=data/wadi/WADI-CLEAN_scaler.pkl",
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
    )
    semantic_map = VariableSemanticMap.from_names(metadata.active_names, style="full")
    config = replace(
        load_config(PACKAGE_ROOT / "configs" / "wadi_qwen3_06b.json"),
        window_length=args.window_length,
        patch_size=args.window_length,
        moirai_encoder_layer=args.moirai_layer,
        projector="direct",
        vocab_loss_weight=0.0,
        discrete_to_text=True,
        dtt_semantic_style="full",
    )
    endpoint = max(config.common_min_endpoint, config.window_length - 1)
    window = np.asarray(
        train_x[endpoint - config.window_length + 1 : endpoint + 1, metadata.active_indices],
        dtype=np.float32,
    )
    if window.shape != (config.window_length, len(metadata.active_indices)):
        raise RuntimeError(f"Unexpected probe-window shape: {window.shape}")
    model = build_model(
        config,
        active_variable_count=len(metadata.active_indices),
        variable_type_ids=torch.tensor(metadata.type_ids, dtype=torch.long),
        device=device,
        continuous_indices=metadata.continuous_indices,
        discrete_indices=metadata.discrete_indices,
        active_names=metadata.active_names,
        active_descriptions=semantic_map.descriptions,
        discrete_vocabulary=vocabulary,
    )
    return config, model, torch.from_numpy(window)


def probe_batch(
    model: torch.nn.Module,
    window: torch.Tensor,
    batch_size: int,
    device: torch.device,
    objective: DualObjectiveLoss,
) -> dict[str, object]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before_free, before_total = torch.cuda.mem_get_info(device)
    else:
        before_free = before_total = None
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=3e-4)
    labels = torch.arange(batch_size, dtype=torch.long, device=device) % 2
    windows = window.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    started = time.perf_counter()
    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(windows)
            losses = objective(output.classifier_logits, output.verbalizer_logits, labels)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(list(trainable_parameters(model)), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result: dict[str, object] = {
            "batch_size": batch_size,
            "status": "ok",
            "loss": round(float(losses["loss"].detach().cpu()), 6),
            "seconds": round(time.perf_counter() - started, 3),
        }
        if device.type == "cuda":
            result.update(
                {
                    "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
                    "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
                    "free_before_mb": round(before_free / 1024**2, 1),
                    "total_memory_mb": round(before_total / 1024**2, 1),
                }
            )
        return result
    except torch.cuda.OutOfMemoryError as error:
        return {"batch_size": batch_size, "status": "oom", "error": str(error).splitlines()[0]}
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            return {"batch_size": batch_size, "status": "oom", "error": str(error).splitlines()[0]}
        raise
    finally:
        del optimizer, labels, windows
        if device.type == "cuda":
            torch.cuda.empty_cache()


def recommend_batch_size(
    successes: list[dict[str, object]],
    *,
    device: torch.device,
    total_memory_mb: float | None,
    headroom_fraction: float,
) -> int | None:
    if not successes:
        return None
    if device.type != "cuda":
        return int(successes[-1]["batch_size"])
    if total_memory_mb is None:
        raise ValueError("CUDA recommendation requires total GPU memory")
    limit_mb = total_memory_mb * (1 - headroom_fraction)
    with_headroom = [
        item for item in successes if float(item["peak_reserved_mb"]) <= limit_mb
    ]
    return int(with_headroom[-1]["batch_size"]) if with_headroom else None


def main() -> None:
    args = parse_args()
    if not 0 <= args.headroom_fraction < 1:
        raise ValueError("--headroom-fraction must be in [0, 1)")
    device = resolve_device(args.device)
    batches = parse_batch_sizes(args.batch_sizes)
    report = {"environment": environment_report(device)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    installed_transformers = package_version("transformers")
    if installed_transformers is None or Version(installed_transformers) < Version("4.51"):
        raise RuntimeError("Transformers >= 4.51 is required for Qwen3; run pip install -r requirements.txt")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not args.skip_official_smoke:
        smoke_code = run_official_smoke(args, device, f"env_full_prompt_smoke_{stamp}")
        if smoke_code != 0:
            raise RuntimeError(f"Official full-prompt smoke failed with exit code {smoke_code}")
    config, model, window = build_full_prompt_model(args, device)
    objective = DualObjectiveLoss(vocab_weight=config.vocab_loss_weight, classifier_weight=config.classifier_loss_weight)
    results: list[dict[str, object]] = []
    for batch_size in batches:
        result = probe_batch(model, window, batch_size, device, objective)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
        if result["status"] == "oom":
            break
    successes = [item for item in results if item["status"] == "ok"]
    recommendation = recommend_batch_size(
        successes,
        device=device,
        total_memory_mb=(
            float(report["environment"]["gpu_total_memory_mb"])
            if device.type == "cuda"
            else None
        ),
        headroom_fraction=args.headroom_fraction,
    )
    report.update(
        {
            "purpose": "engineering smoke and physical-batch capacity probe; not a detection experiment",
            "configuration": {
                "window_length": config.window_length,
                "moirai_layer": config.moirai_encoder_layer,
                "projector": config.projector,
                "discrete_to_text": config.discrete_to_text,
                "dtt_semantic_style": config.dtt_semantic_style,
            },
            "results": results,
            "recommended_physical_batch_size": recommendation,
            "headroom_fraction": args.headroom_fraction,
        }
    )
    output = PACKAGE_ROOT / "outputs" / f"environment_full_prompt_probe_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={output}")
    if recommendation is None:
        raise RuntimeError("No candidate batch completed; use a shorter prompt or a larger GPU")


if __name__ == "__main__":
    main()
