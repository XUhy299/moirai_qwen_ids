"""Small, explicit training/evaluation utilities with no test-set access."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch import nn

from .losses import DualObjectiveLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    objective: DualObjectiveLoss,
    device: torch.device,
    *,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    started = time.perf_counter()
    model.train()
    totals = {"loss": 0.0, "classifier_loss": 0.0, "verbalizer_loss": 0.0}
    examples = 0
    use_amp = device.type == "cuda"
    for batch in loader:
        windows = batch["window"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model(windows)
            losses = objective(output.classifier_logits, output.verbalizer_logits, labels)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(list(trainable_parameters(model)), grad_clip)
        optimizer.step()
        batch_size = labels.numel()
        examples += batch_size
        for key in totals:
            totals[key] += float(losses[key]) * batch_size
    if examples == 0:
        raise RuntimeError("Training loader produced no examples")
    metrics = {key: value / examples for key, value in totals.items()}
    metrics["seconds"] = time.perf_counter() - started
    return metrics


@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader, device: torch.device) -> dict[str, object]:
    started = time.perf_counter()
    model.eval()
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    vocab_scores_all: list[np.ndarray] = []
    endpoints_all: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    for batch in loader:
        windows = batch["window"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model(windows)
        scores = torch.softmax(output.classifier_logits.float(), dim=-1)[:, 1]
        vocab_scores = None
        if output.verbalizer_logits is not None:
            vocab_scores = torch.softmax(output.verbalizer_logits.float(), dim=-1)[:, 1]
        labels_all.append(batch["label"].numpy())
        endpoints_all.append(batch["endpoint"].numpy())
        scores_all.append(scores.cpu().numpy())
        if vocab_scores is not None:
            vocab_scores_all.append(vocab_scores.cpu().numpy())
    labels = np.concatenate(labels_all)
    scores = np.concatenate(scores_all)
    vocab_scores = np.concatenate(vocab_scores_all) if vocab_scores_all else None
    endpoints = np.concatenate(endpoints_all)
    if np.unique(labels).size != 2:
        raise ValueError("Validation evaluation requires both normal and anomaly endpoints")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_index = int(np.nanargmax(f1))
    threshold = float(thresholds[min(best_index, len(thresholds) - 1)])
    predictions = (scores >= threshold).astype(np.int64)
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    metrics: dict[str, object] = {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "vocab_roc_auc": None if vocab_scores is None else float(roc_auc_score(labels, vocab_scores)),
        "vocab_pr_auc": None if vocab_scores is None else float(average_precision_score(labels, vocab_scores)),
        "threshold": threshold,
        "precision": float(true_positive / max(true_positive + false_positive, 1)),
        "recall": float(true_positive / max(true_positive + false_negative, 1)),
        "f1": float(f1[best_index]),
        "false_positive_rate": float(false_positive / max(false_positive + true_negative, 1)),
        "false_negative_rate": float(false_negative / max(false_negative + true_positive, 1)),
        "examples": int(len(labels)),
        "anomalies": int(labels.sum()),
        "endpoints_min": int(endpoints.min()),
        "endpoints_max": int(endpoints.max()),
        "seconds": time.perf_counter() - started,
    }
    if vocab_scores is not None:
        vocab_precision, vocab_recall, vocab_thresholds = precision_recall_curve(labels, vocab_scores)
        vocab_f1 = 2 * vocab_precision * vocab_recall / np.maximum(
            vocab_precision + vocab_recall,
            1e-12,
        )
        vocab_best_index = int(np.nanargmax(vocab_f1))
        vocab_threshold = float(
            vocab_thresholds[min(vocab_best_index, len(vocab_thresholds) - 1)]
        )
        vocab_predictions = (vocab_scores >= vocab_threshold).astype(np.int64)
        vocab_fp = int(((vocab_predictions == 1) & (labels == 0)).sum())
        vocab_tn = int(((vocab_predictions == 0) & (labels == 0)).sum())
        vocab_fn = int(((vocab_predictions == 0) & (labels == 1)).sum())
        vocab_tp = int(((vocab_predictions == 1) & (labels == 1)).sum())
        metrics.update(
            {
                "vocab_threshold": vocab_threshold,
                "vocab_precision": float(vocab_tp / max(vocab_tp + vocab_fp, 1)),
                "vocab_recall": float(vocab_tp / max(vocab_tp + vocab_fn, 1)),
                "vocab_f1": float(vocab_f1[vocab_best_index]),
                "vocab_false_positive_rate": float(vocab_fp / max(vocab_fp + vocab_tn, 1)),
                "vocab_false_negative_rate": float(vocab_fn / max(vocab_fn + vocab_tp, 1)),
            }
        )
    return metrics


def save_trainable_checkpoint(
    model: nn.Module,
    output_path: str | Path,
    *,
    config: dict[str, object],
    metrics: dict[str, object],
) -> None:
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trainable_state_dict": state,
            "config": config,
            "validation_metrics": metrics,
            "saved_at_unix": time.time(),
        },
        output_path,
    )


def save_json(payload: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _format_metric(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def write_run_markdown(
    output_dir: str | Path,
    *,
    config: dict[str, object],
    protocol: dict[str, object],
    environment: dict[str, object],
    history: list[dict[str, object]],
    status: str,
) -> None:
    """Write a human-readable companion to the machine-readable run artifacts."""
    output_dir = Path(output_dir)
    verbalizer_is_primary = config.get("classifier_loss_weight") == 0
    selection_metric = "vocab_pr_auc" if verbalizer_is_primary else "pr_auc"
    primary_output = "verbalizer" if verbalizer_is_primary else "classifier"
    metric_prefix = "vocab_" if verbalizer_is_primary else ""

    def primary_metric(dev: dict[str, object], name: str) -> object:
        return dev.get(f"{metric_prefix}{name}")

    selected = None
    if history:
        selected = max(
            history,
            key=lambda row: float(
                row.get("selection_score", row.get("dev", {}).get(selection_metric, -float("inf")))
            ),
        )
    layer = config.get("moirai_encoder_layer")
    layer_text = "最终层" if layer is None else str(layer)
    locked_test_files = protocol.get("locked_test_files_opened")
    if locked_test_files is None:
        test_opened = bool(protocol.get("test_arrays_opened", False))
    else:
        test_opened = bool(locked_test_files)
    lines = [
        f"# {output_dir.name}",
        "",
        "## 运行状态",
        "",
        f"- 状态：`{status}`",
        f"- 实验性质：{protocol.get('description', '未记录')}",
        f"- 正式测试数组已打开：`{str(test_opened).lower()}`",
        "- 结果边界：当前指标来自开发集；若测试数组未打开，不得表述为测试结果。",
        "",
        "## 核心配置",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| Seed | {config.get('seed')} |",
        f"| 窗口 / Patch | {config.get('window_length')} / {config.get('patch_size')} |",
        f"| MOIRAI | {config.get('moirai_size')}，Encoder层={layer_text} |",
        f"| Backbone | {config.get('backbone')} |",
        f"| Projector | {config.get('projector')} |",
        f"| 分类损失权重 | {config.get('classifier_loss_weight')} |",
        f"| 词表损失权重 | {config.get('vocab_loss_weight')} |",
        f"| 主评估输出 | {primary_output} |",
        f"| 配置 Epochs / 实际完成 | {config.get('epochs')} / {len(history)} |",
        f"| 学习率 | {config.get('learning_rate')} |",
        f"| DTT语义样式 | {protocol.get('dtt_semantic_style')} |",
        f"| DTT语义变体 | {protocol.get('dtt_semantic_variant')} |",
        f"| Prompt布局 | {protocol.get('prompt_layout')} |",
        f"| Qwen chat template | {protocol.get('qwen_chat_template_applied')} |",
        "",
        "## 数据协议",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| 正常训练窗口 | {protocol.get('train_normal_windows')} |",
        f"| 异常训练窗口（合计） | {protocol.get('train_anomaly_windows')} |",
        f"| 真实异常训练窗口 | {protocol.get('train_real_anomaly_windows', protocol.get('train_anomaly_windows'))} |",
        f"| 合成异常训练窗口 | {protocol.get('train_synthetic_anomaly_windows', 0)} |",
        f"| 训练窗口总数 | {protocol.get('train_total_windows')} |",
        f"| 开发窗口 | {protocol.get('dev_windows')} |",
        f"| Smoke | {protocol.get('smoke')} |",
        f"| 实际训练/评估batch | {protocol.get('effective_train_batch_size')} / {protocol.get('effective_eval_batch_size')} |",
        f"| 开发 stride | {protocol.get('dev_stride')} |",
        f"| 统一最小端点 | {protocol.get('common_min_endpoint')} |",
        f"| 攻击段 | {protocol.get('attack_run')} |",
        f"| Support端点范围 | {protocol.get('support_endpoint_range')} |",
        f"| Query端点范围 | {protocol.get('query_endpoint_range')} |",
        f"| 标签规则 | {protocol.get('endpoint_label_rule')} |",
        "",
        "## 逐 Epoch 指标",
        "",
        f"选模指标：`{selection_metric}`。阈值指标与选中的同一 epoch 绑定。",
        "",
        "| Epoch | Train loss | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Threshold | Train s | Dev s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history:
        train = row["train"]
        dev = row["dev"]
        lines.append(
            "| {epoch} | {loss} | {roc} | {pr} | {precision} | {recall} | {f1} | "
            "{fpr} | {threshold} | {train_s} | {dev_s} |".format(
                epoch=row["epoch"],
                loss=_format_metric(train.get("loss")),
                roc=_format_metric(primary_metric(dev, "roc_auc")),
                pr=_format_metric(primary_metric(dev, "pr_auc")),
                precision=_format_metric(primary_metric(dev, "precision")),
                recall=_format_metric(primary_metric(dev, "recall")),
                f1=_format_metric(primary_metric(dev, "f1")),
                fpr=_format_metric(primary_metric(dev, "false_positive_rate")),
                threshold=_format_metric(primary_metric(dev, "threshold")),
                train_s=_format_metric(train.get("seconds")),
                dev_s=_format_metric(dev.get("seconds")),
            )
        )
    if not history:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — |")
    lines.extend(["", "## 当前最佳", ""])
    if selected is None:
        lines.append("尚无训练 epoch；本目录只完成了数据协议准备。")
    else:
        dev = selected["dev"]
        lines.extend(
            [
                f"- Epoch：{selected['epoch']}",
                f"- 选模指标 `{selection_metric}`："
                f"{_format_metric(selected.get('selection_score', dev.get(selection_metric)))}",
                f"- 主评估输出：{primary_output}",
                f"- ROC-AUC：{_format_metric(primary_metric(dev, 'roc_auc'))}",
                f"- PR-AUC：{_format_metric(primary_metric(dev, 'pr_auc'))}",
                f"- Precision / Recall / F1：{_format_metric(primary_metric(dev, 'precision'))} / "
                f"{_format_metric(primary_metric(dev, 'recall'))} / "
                f"{_format_metric(primary_metric(dev, 'f1'))}",
                f"- FPR / FNR：{_format_metric(primary_metric(dev, 'false_positive_rate'))} / "
                f"{_format_metric(primary_metric(dev, 'false_negative_rate'))}",
                f"- 峰值CUDA显存：{_format_metric(selected.get('peak_cuda_memory_mb'))} MB",
            ]
        )
    lines.extend(
        [
            "",
            "## 环境与产物",
            "",
            f"- Python：{environment.get('python_version')}（`{environment.get('python_executable')}`）",
            f"- PyTorch / Transformers：{environment.get('torch_version')} / {environment.get('transformers_version')}",
            f"- 设备：{environment.get('gpu_name') or environment.get('device')}",
            "- 机器可读文件：`config.json`、`protocol.json`、`environment.json`、`history.json`。",
            "- 最佳可训练参数：`best_trainable.pt`（若训练已开始）。",
            "",
            "## 局限性",
            "",
            f"- {protocol.get('limitation', '未记录')} ",
            "- 当前F1阈值在同一开发集选择，只能用于诊断；正式测试前必须锁定阈值和后处理。",
            "",
        ]
    )
    (output_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")
