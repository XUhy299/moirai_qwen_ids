# migration_real_stack_compact_l64_20260724

## 运行状态

- 状态：`completed`
- 实验性质：single-attack-event supervised transfer
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
| 配置 Epochs / 实际完成 | 5 / 1 |
| 学习率 | 0.0003 |
| DTT语义样式 | compact |
| Prompt布局 | variable_aligned_semantic_chat_dtt_v3 |
| Qwen chat template | True |

## 数据协议

| 项目 | 值 |
|---|---|
| 正常训练窗口 | 2 |
| 异常训练窗口（合计） | 2 |
| 真实异常训练窗口 | 2 |
| 合成异常训练窗口 | 0 |
| 训练窗口总数 | 4 |
| 开发窗口 | 4 |
| Smoke | True |
| 实际训练/评估batch | 1 / 1 |
| 开发 stride | 8 |
| 统一最小端点 | 127 |
| 攻击段 | [5139, 6625] |
| Support端点范围 | [5139, 5143] |
| Query端点范围 | [5946, 6624] |
| 标签规则 | Y[window_end] |

## 逐 Epoch 指标

选模指标：`pr_auc`。阈值指标与选中的同一 epoch 绑定。

| Epoch | Train loss | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | Threshold | Train s | Dev s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8555 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.6658 | 4.0981 | 1.1397 |

## 当前最佳

- Epoch：1
- 选模指标 `pr_auc`：1.0000
- ROC-AUC：1.0000
- PR-AUC：1.0000
- Precision / Recall / F1：1.0000 / 1.0000 / 1.0000
- FPR / FNR：0.0000 / 0.0000
- 峰值CUDA显存：4630.7183 MB

## 环境与产物

- Python：3.11.9（`C:\Users\83509\Desktop\python project\.venv\Scripts\python.exe`）
- PyTorch / Transformers：2.4.1+cu121 / 4.51.3
- 设备：NVIDIA GeForce RTX 4060 Laptop GPU
- 机器可读文件：`config.json`、`protocol.json`、`environment.json`、`history.json`。
- 最佳可训练参数：`best_trainable.pt`（若训练已开始）。

## 局限性

- Support and query are chronological portions of the same attack event. 
- 当前F1阈值在同一开发集选择，只能用于诊断；正式测试前必须锁定阈值和后处理。
