# e1_numeric_l64_l12_direct_head_ce_synrot170_seed2026_head_round1_20260725

## 运行状态

- 状态：`completed`
- 实验性质：single-attack-event supervised transfer with train-only synthetic anomalies
- 正式测试数组已打开：`false`
- 结果边界：当前指标来自开发集；若测试数组未打开，不得表述为测试结果。

## 核心配置

| 项目 | 值 |
|---|---|
| Seed | 2026 |
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
| 1 | 0.6828 | 0.9325 | 0.0984 | 0.1364 | 0.3176 | 0.1908 | 0.0272 | 0.4683 | 40.0465 | 85.4814 |
| 2 | 0.6714 | 0.9224 | 0.0968 | 0.1012 | 0.6941 | 0.1766 | 0.0835 | 0.7527 | 39.6274 | 84.8683 |
| 3 | 0.6249 | 0.8636 | 0.0453 | 0.0571 | 0.7647 | 0.1063 | 0.1709 | 0.3427 | 39.4263 | 81.2401 |
| 4 | 0.6631 | 0.8071 | 0.0332 | 0.0373 | 0.9059 | 0.0716 | 0.3168 | 0.5373 | 39.5827 | 82.8164 |
| 5 | 0.6510 | 0.8580 | 0.0442 | 0.0520 | 0.7765 | 0.0974 | 0.1918 | 0.6427 | 39.3913 | 81.5558 |

## 当前最佳

- Epoch：1
- 选模指标 `pr_auc`：0.0984
- 主评估输出：classifier
- ROC-AUC：0.9325
- PR-AUC：0.0984
- Precision / Recall / F1：0.1364 / 0.3176 / 0.1908
- FPR / FNR：0.0272 / 0.6824
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
