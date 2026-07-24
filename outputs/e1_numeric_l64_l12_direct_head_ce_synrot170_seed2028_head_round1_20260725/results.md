# e1_numeric_l64_l12_direct_head_ce_synrot170_seed2028_head_round1_20260725

## 运行状态

- 状态：`completed`
- 实验性质：single-attack-event supervised transfer with train-only synthetic anomalies
- 正式测试数组已打开：`false`
- 结果边界：当前指标来自开发集；若测试数组未打开，不得表述为测试结果。

## 核心配置

| 项目 | 值 |
|---|---|
| Seed | 2028 |
| 窗口 / Patch | 64 / 64 |
| MOIRAI | base，Encoder层=12 |
| Backbone | qwen |
| Projector | direct |
| 分类损失权重 | 1.0 |
| 词表损失权重 | 0.0 |
| 主评估输出 | classifier |
| 配置 Epochs / 实际完成 | 5 / 5 |
| 学习率 | 0.0003 |
| DTT语义样式 | None |
| Prompt布局 | legacy_token_block |
| Qwen chat template | False |

## 数据协议

| 项目 | 值 |
|---|---|
| 正常训练窗口 | 510 |
| 异常训练窗口（合计） | 340 |
| 真实异常训练窗口 | 170 |
| 合成异常训练窗口 | 170 |
| 训练窗口总数 | 850 |
| 开发窗口 | 6364 |
| Smoke | False |
| 实际训练/评估batch | 4 / 4 |
| 开发 stride | 8 |
| 统一最小端点 | 127 |
| 攻击段 | [5139, 6625] |
| Support端点范围 | [5139, 5815] |
| Query端点范围 | [5946, 6624] |
| 标签规则 | Y[window_end] |

## 逐 Epoch 指标

选模指标：`pr_auc`。阈值指标与选中的同一 epoch 绑定。

| Epoch | Train loss | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Threshold | Train s | Dev s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6993 | 0.9024 | 0.0602 | 0.0818 | 0.8000 | 0.1485 | 0.1215 | 0.4868 | 39.6929 | 77.1929 |
| 2 | 0.6627 | 0.9343 | 0.0922 | 0.1049 | 0.8235 | 0.1862 | 0.0951 | 0.2892 | 39.6014 | 77.5491 |
| 3 | 0.6410 | 0.9615 | 0.1478 | 0.1705 | 0.7059 | 0.2746 | 0.0465 | 0.6182 | 39.3553 | 77.3989 |
| 4 | 0.6112 | 0.9181 | 0.0955 | 0.1264 | 0.4118 | 0.1934 | 0.0385 | 0.8657 | 39.1341 | 76.4601 |
| 5 | 0.6396 | 0.9494 | 0.1112 | 0.1494 | 0.6941 | 0.2458 | 0.0535 | 0.9365 | 39.1083 | 77.2897 |

## 当前最佳

- Epoch：3
- 选模指标 `pr_auc`：0.1478
- 主评估输出：classifier
- ROC-AUC：0.9615
- PR-AUC：0.1478
- Precision / Recall / F1：0.1705 / 0.7059 / 0.2746
- FPR / FNR：0.0465 / 0.2941
- 峰值CUDA显存：2397.9131 MB

## 环境与产物

- Python：3.10.20（`/root/miniconda3/envs/mqids/bin/python`）
- PyTorch / Transformers：2.4.1+cu121 / 4.51.3
- 设备：NVIDIA GeForce RTX 4080 SUPER
- 机器可读文件：`config.json`、`protocol.json`、`environment.json`、`history.json`。
- 最佳可训练参数：`best_trainable.pt`（若训练已开始）。

## 局限性

- Support and query are chronological portions of the same attack event. 
- 当前F1阈值在同一开发集选择，只能用于诊断；正式测试前必须锁定阈值和后处理。
