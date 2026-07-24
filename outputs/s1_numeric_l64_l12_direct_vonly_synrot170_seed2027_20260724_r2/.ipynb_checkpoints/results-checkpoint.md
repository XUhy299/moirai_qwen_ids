# s1_numeric_l64_l12_direct_vonly_synrot170_seed2027_20260724_r2

## 运行状态

- 状态：`completed`
- 实验性质：single-attack-event supervised transfer with train-only synthetic anomalies
- 正式测试数组已打开：`false`
- 结果边界：当前指标来自开发集；若测试数组未打开，不得表述为测试结果。

## 核心配置

| 项目 | 值 |
|---|---|
| Seed | 2027 |
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
| 1 | 0.7177 | 0.8447 | 0.0465 | 0.0587 | 0.3647 | 0.1011 | 0.0792 | 0.3785 | 42.9091 | 75.4162 |
| 2 | 0.6543 | 0.6737 | 0.0205 | 0.0245 | 0.4118 | 0.0462 | 0.2220 | 0.4649 | 42.6157 | 75.8016 |
| 3 | 0.6069 | 0.7846 | 0.0304 | 0.0380 | 0.3882 | 0.0692 | 0.1331 | 0.4634 | 42.6400 | 75.1849 |
| 4 | 0.5785 | 0.6013 | 0.0157 | 0.0195 | 0.6118 | 0.0378 | 0.4166 | 0.4800 | 42.8104 | 77.0117 |
| 5 | 0.5726 | 0.2623 | 0.0086 | 0.0139 | 0.9882 | 0.0275 | 0.9460 | 0.4446 | 43.0968 | 76.5343 |

## 当前最佳

- Epoch：3
- 选模指标 `vocab_pr_auc`：0.1566
- ROC-AUC：0.7846
- PR-AUC：0.0304
- Precision / Recall / F1：0.0380 / 0.3882 / 0.0692
- FPR / FNR：0.1331 / 0.6118
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
