# s1_numeric_l64_l12_direct_vonly_synrot170_seed2026_20260724_r2

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
| 分类损失权重 | 0.0 |
| 词表损失权重 | 1.0 |
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

选模指标：`vocab_pr_auc`。阈值指标与选中的同一 epoch 绑定。

| Epoch | Train loss | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Threshold | Train s | Dev s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7451 | 0.7623 | 0.0322 | 0.0372 | 0.7176 | 0.0708 | 0.2513 | 0.4204 | 43.5641 | 88.0594 |
| 2 | 0.7810 | 0.1802 | 0.0077 | 0.0134 | 1.0000 | 0.0264 | 0.9967 | 0.3063 | 43.7068 | 86.2600 |
| 3 | 0.6787 | 0.0704 | 0.0072 | 0.0134 | 1.0000 | 0.0264 | 0.9973 | 0.2636 | 43.5241 | 82.5181 |
| 4 | 0.6845 | 0.3525 | 0.0214 | 0.0140 | 1.0000 | 0.0276 | 0.9551 | 0.5233 | 42.7558 | 82.8256 |
| 5 | 0.6702 | 0.7909 | 0.0364 | 0.0439 | 0.5882 | 0.0816 | 0.1736 | 0.2422 | 43.2845 | 83.7242 |

## 当前最佳

- Epoch：4
- 选模指标 `vocab_pr_auc`：0.1119
- ROC-AUC：0.3525
- PR-AUC：0.0214
- Precision / Recall / F1：0.0140 / 1.0000 / 0.0276
- FPR / FNR：0.9551 / 0.0000
- 峰值CUDA显存：2397.8765 MB

## 环境与产物

- Python：3.10.20（`/root/miniconda3/envs/mqids/bin/python`）
- PyTorch / Transformers：2.4.1+cu121 / 4.51.3
- 设备：NVIDIA GeForce RTX 4080 SUPER
- 机器可读文件：`config.json`、`protocol.json`、`environment.json`、`history.json`。
- 最佳可训练参数：`best_trainable.pt`（若训练已开始）。

## 局限性

- Support and query are chronological portions of the same attack event. 
- 当前F1阈值在同一开发集选择，只能用于诊断；正式测试前必须锁定阈值和后处理。
