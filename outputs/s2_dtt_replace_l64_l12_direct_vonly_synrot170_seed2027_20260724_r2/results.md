# s2_dtt_replace_l64_l12_direct_vonly_synrot170_seed2027_20260724_r2

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
| DTT语义样式 | compact |
| Prompt布局 | variable_aligned_semantic_chat_dtt_v3 |
| Qwen chat template | True |

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
| 1 | 0.7900 | 0.5114 | 0.0127 | 0.0160 | 0.8235 | 0.0314 | 0.6855 | 0.4778 | 146.9045 | 439.3685 |
| 2 | 0.7800 | 0.8978 | 0.0652 | 0.0833 | 0.4235 | 0.1393 | 0.0631 | 0.5770 | 150.8786 | 442.1782 |
| 3 | 0.6415 | 0.6286 | 0.0172 | 0.0216 | 0.6941 | 0.0419 | 0.4259 | 0.5795 | 145.8464 | 448.0440 |
| 4 | 0.6329 | 0.9295 | 0.1264 | 0.1464 | 0.4824 | 0.2247 | 0.0381 | 0.5804 | 155.4062 | 445.3789 |
| 5 | 0.5784 | 0.8199 | 0.0363 | 0.0430 | 0.4000 | 0.0776 | 0.1206 | 0.6116 | 157.3069 | 449.0521 |

## 当前最佳

- Epoch：2
- 选模指标 `vocab_pr_auc`：0.0958
- ROC-AUC：0.8978
- PR-AUC：0.0652
- Precision / Recall / F1：0.0833 / 0.4235 / 0.1393
- FPR / FNR：0.0631 / 0.5765
- 峰值CUDA显存：14787.4106 MB

## 环境与产物

- Python：3.10.20（`/root/miniconda3/envs/mqids/bin/python`）
- PyTorch / Transformers：2.4.1+cu121 / 4.51.3
- 设备：NVIDIA GeForce RTX 4080 SUPER
- 机器可读文件：`config.json`、`protocol.json`、`environment.json`、`history.json`。
- 最佳可训练参数：`best_trainable.pt`（若训练已开始）。

## 局限性

- Support and query are chronological portions of the same attack event. 
- 当前F1阈值在同一开发集选择，只能用于诊断；正式测试前必须锁定阈值和后处理。
