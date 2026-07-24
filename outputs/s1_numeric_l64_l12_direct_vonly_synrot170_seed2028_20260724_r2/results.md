# s1_numeric_l64_l12_direct_vonly_synrot170_seed2028_20260724_r2

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
| 1 | 0.7618 | 0.6076 | 0.0173 | 0.0209 | 0.7412 | 0.0406 | 0.4705 | 0.1833 | 43.5018 | 88.0613 |
| 2 | 0.6199 | 0.0813 | 0.0074 | 0.0136 | 1.0000 | 0.0268 | 0.9839 | 0.1743 | 43.6012 | 83.5982 |
| 3 | 0.5662 | 0.3593 | 0.0096 | 0.0143 | 0.9412 | 0.0281 | 0.8790 | 0.1917 | 43.0165 | 85.1110 |
| 4 | 0.5091 | 0.5858 | 0.0225 | 0.0375 | 0.2353 | 0.0647 | 0.0817 | 0.1992 | 41.9999 | 86.2874 |
| 5 | 0.5212 | 0.3737 | 0.0103 | 0.0145 | 1.0000 | 0.0285 | 0.9232 | 0.1814 | 44.2277 | 88.4487 |

## 当前最佳

- Epoch：3
- 选模指标 `vocab_pr_auc`：0.1326
- ROC-AUC：0.3593
- PR-AUC：0.0096
- Precision / Recall / F1：0.0143 / 0.9412 / 0.0281
- FPR / FNR：0.8790 / 0.0588
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
