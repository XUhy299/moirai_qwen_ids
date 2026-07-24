# s2_dtt_replace_l64_l12_direct_vonly_synrot170_seed2028_20260724_r2

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
| 1 | 0.8129 | 0.3726 | 0.0102 | 0.0134 | 1.0000 | 0.0265 | 0.9955 | 0.4162 | 152.9598 | 445.0058 |
| 2 | 0.7399 | 0.8088 | 0.0349 | 0.0406 | 0.5765 | 0.0759 | 0.1844 | 0.4655 | 146.8868 | 446.1634 |
| 3 | 0.7023 | 0.5684 | 0.0197 | 0.0312 | 0.2353 | 0.0552 | 0.0987 | 0.4833 | 144.5845 | 445.3895 |
| 4 | 0.6211 | 0.7983 | 0.0419 | 0.0352 | 0.8588 | 0.0676 | 0.3187 | 0.4586 | 144.7317 | 446.1717 |
| 5 | 0.5801 | 0.3338 | 0.0091 | 0.0134 | 1.0000 | 0.0264 | 0.9973 | 0.4392 | 143.0862 | 447.4125 |

## 当前最佳

- Epoch：4
- 选模指标 `vocab_pr_auc`：0.1495
- ROC-AUC：0.7983
- PR-AUC：0.0419
- Precision / Recall / F1：0.0352 / 0.8588 / 0.0676
- FPR / FNR：0.3187 / 0.1412
- 峰值CUDA显存：14787.6606 MB

## 环境与产物

- Python：3.10.20（`/root/miniconda3/envs/mqids/bin/python`）
- PyTorch / Transformers：2.4.1+cu121 / 4.51.3
- 设备：NVIDIA GeForce RTX 4080 SUPER
- 机器可读文件：`config.json`、`protocol.json`、`environment.json`、`history.json`。
- 最佳可训练参数：`best_trainable.pt`（若训练已开始）。

## 局限性

- Support and query are chronological portions of the same attack event. 
- 当前F1阈值在同一开发集选择，只能用于诊断；正式测试前必须锁定阈值和后处理。
