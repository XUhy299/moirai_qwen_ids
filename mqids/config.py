"""Typed experiment configuration with research-protocol validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


NATIVE_PATCH_SIZES = (8, 16, 32, 64, 128)
PROMPT_VARIANTS = ("process", "generic", "minimal", "wrong_process")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 2026
    window_length: int = 32
    patch_size: int = 32
    drop_train_constant_channels: bool = True
    backbone: str = "qwen"
    projector: str = "reprogramming"
    projector_hidden_dim: int = 1536
    attention_dim: int = 256
    attention_heads: int = 4
    baseline_hidden_dim: int = 256
    baseline_layers: int = 2
    prototype_words: tuple[str, ...] = ()
    vocab_loss_weight: float = 0.1
    classifier_loss_weight: float = 1.0
    dropout: float = 0.1
    support_fraction: float = 0.5
    split_guard: int = 128
    common_min_endpoint: int = 127
    normal_to_anomaly_ratio: int = 3
    normal_train_stride: int = 8
    anomaly_train_stride: int = 4
    dev_stride: int = 8
    max_normal_train_windows: int = 20_000
    batch_size: int = 4
    eval_batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    moirai_size: str = "base"
    moirai_encoder_layer: int | None = None
    prompt_variant: str = "process"
    discrete_to_text: bool = False
    dtt_semantic_style: str = "compact"
    dtt_numeric_mode: str = "continuous_only"
    qwen_subdir: str = "Qwen3-0.6B"
    labels: tuple[str, str] = ("正常", "异常")

    def __post_init__(self) -> None:
        if self.window_length not in NATIVE_PATCH_SIZES:
            raise ValueError(f"window_length must be one of {NATIVE_PATCH_SIZES}")
        if self.patch_size != self.window_length:
            raise ValueError("One-token-per-variable mode requires patch_size == window_length")
        if self.projector not in {"linear", "direct", "reprogramming"}:
            raise ValueError("projector must be linear, direct, or reprogramming")
        if self.backbone not in {"qwen", "none"}:
            raise ValueError("backbone must be qwen or none")
        if self.backbone == "none" and self.discrete_to_text:
            raise ValueError("discrete_to_text requires the Qwen backbone")
        if not 0.0 < self.support_fraction < 1.0:
            raise ValueError("support_fraction must lie strictly between 0 and 1")
        if self.split_guard < self.window_length:
            raise ValueError("split_guard must be at least window_length")
        if self.common_min_endpoint < self.window_length - 1:
            raise ValueError("common_min_endpoint must provide enough history for the window")
        if self.vocab_loss_weight < 0:
            raise ValueError("vocab_loss_weight cannot be negative")
        if self.classifier_loss_weight < 0:
            raise ValueError("classifier_loss_weight cannot be negative")
        if self.classifier_loss_weight == 0 and self.vocab_loss_weight == 0:
            raise ValueError("At least one supervised loss weight must be positive")
        if min(self.normal_train_stride, self.anomaly_train_stride, self.dev_stride) < 1:
            raise ValueError("All sampling strides must be positive")
        if min(self.batch_size, self.eval_batch_size) < 1:
            raise ValueError("Batch sizes must be positive")
        if min(
            self.normal_to_anomaly_ratio,
            self.max_normal_train_windows,
            self.epochs,
            self.projector_hidden_dim,
            self.attention_dim,
            self.attention_heads,
            self.baseline_hidden_dim,
            self.baseline_layers,
        ) < 1:
            raise ValueError("Counts, dimensions, and epochs must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.moirai_size not in {"small", "base", "large"}:
            raise ValueError("moirai_size must be small, base, or large")
        if len(self.labels) != 2:
            raise ValueError("Exactly two verbalizer labels are required")
        if self.labels[0] == self.labels[1] or any(not label.strip() for label in self.labels):
            raise ValueError("Verbalizer labels must be distinct non-empty strings")
        if self.backbone == "none" and (
            self.vocab_loss_weight != 0 or self.classifier_loss_weight == 0
        ):
            raise ValueError("The no-Qwen baseline requires classifier-only supervision")
        if self.moirai_encoder_layer is not None and self.moirai_encoder_layer < 1:
            raise ValueError("moirai_encoder_layer uses 1-based indexing and must be positive")
        if self.prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(f"prompt_variant must be one of {PROMPT_VARIANTS}")
        if self.dtt_semantic_style not in {"compact", "full"}:
            raise ValueError("dtt_semantic_style must be compact or full")
        if self.dtt_numeric_mode not in {"continuous_only", "all_active"}:
            raise ValueError("dtt_numeric_mode must be continuous_only or all_active")
        if not self.discrete_to_text and self.dtt_numeric_mode != "continuous_only":
            raise ValueError("dtt_numeric_mode=all_active requires discrete_to_text")

    def as_dict(self) -> dict[str, Any]:
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["prototype_words"] = list(self.prototype_words)
        result["labels"] = list(self.labels)
        return result


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    raw["prototype_words"] = tuple(raw.get("prototype_words", ()))
    raw["labels"] = tuple(raw.get("labels", ("正常", "异常")))
    known = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return ExperimentConfig(**raw)
