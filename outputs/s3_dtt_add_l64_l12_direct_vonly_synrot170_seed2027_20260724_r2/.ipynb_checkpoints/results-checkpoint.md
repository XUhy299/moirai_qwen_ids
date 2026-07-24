# s3_dtt_add_l64_l12_direct_vonly_synrot170_seed2027_20260724_r2

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
| 1 | 0.7114 | 0.2989 | 0.0090 | 0.0135 | 0.9765 | 0.0267 | 0.9640 | 0.3907 | 179.9698 | 507.9647 |
| 2 | 0.6887 | 0.1844 | 0.0076 | 0.0134 | 1.0000 | 0.0264 | 0.9997 | 0.2088 | 180.6537 | 510.5065 |
| 3 | 0.6857 | 0.8469 | 0.0432 | 0.0489 | 0.8353 | 0.0923 | 0.2201 | 0.3935 | 172.1662 | 510.2974 |
| 4 | 0.6125 | 0.7845 | 0.0354 | 0.0492 | 0.7294 | 0.0921 | 0.1910 | 0.4076 | 174.3352 | 507.7494 |
| 5 | 0.5717 | 0.9099 | 0.0758 | 0.0841 | 0.6235 | 0.1483 | 0.0919 | 0.4366 | 180.5943 | 506.8712 |

## 当前最佳

- Epoch：5
- 选模指标 `vocab_pr_auc`：0.1462
- ROC-AUC：0.9099
- PR-AUC：0.0758
- Precision / Recall / F1：0.0841 / 0.6235 / 0.1483
- FPR / FNR：0.0919 / 0.3765
- 峰值CUDA显存：16438.0835 MB

## 环境与产物

- Python：3.10.20（`/root/miniconda3/envs/mqids/bin/python`）
- PyTorch / Transformers：2.4.1+cu121 / 4.51.3
- 设备：NVIDIA GeForce RTX 4080 SUPER
- 机器可读文件：`config.json`、`protocol.json`、`environment.json`、`history.json`。
- 最佳可训练参数：`best_trainable.pt`（若训练已开始）。

## 局限性

- Support and query are chronological portions of the same attack event. 
- 当前F1阈值在同一开发集选择，只能用于诊断；正式测试前必须锁定阈值和后处理。
