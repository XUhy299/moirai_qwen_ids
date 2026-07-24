# e2_dtt_add_l64_l12_direct_head_ce_nosynth_seed2027_head_round1_20260725

## 运行状态

- 状态：`completed`
- 实验性质：single-attack-event supervised transfer
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
| 分类损失权重 | 1.0 |
| 词表损失权重 | 0.0 |
| 主评估输出 | classifier |
| 配置 Epochs / 实际完成 | 5 / 5 |
| 学习率 | 0.0003 |
| DTT语义样式 | compact |
| Prompt布局 | variable_aligned_semantic_chat_dtt_v3 |
| Qwen chat template | True |

## 数据协议

| 项目 | 值 |
|---|---|
| 正常训练窗口 | 510 |
| 异常训练窗口（合计） | 170 |
| 真实异常训练窗口 | 170 |
| 合成异常训练窗口 | 0 |
| 训练窗口总数 | 680 |
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
| 1 | 0.4587 | 0.9638 | 0.1535 | 0.1964 | 0.5176 | 0.2848 | 0.0287 | 0.0637 | 139.6886 | 493.3737 |
| 2 | 0.3703 | 0.9265 | 0.1107 | 0.1313 | 0.4588 | 0.2042 | 0.0411 | 0.1082 | 139.2587 | 494.2323 |
| 3 | 0.4235 | 0.9085 | 0.0825 | 0.1017 | 0.6235 | 0.1749 | 0.0745 | 0.6588 | 137.1125 | 488.5014 |
| 4 | 0.2751 | 0.9702 | 0.2092 | 0.1885 | 0.9294 | 0.3135 | 0.0541 | 0.9931 | 133.2201 | 488.9822 |
| 5 | 0.2549 | 0.9705 | 0.2234 | 0.1905 | 0.9412 | 0.3168 | 0.0541 | 0.9917 | 133.1210 | 489.3326 |

## 当前最佳

- Epoch：5
- 选模指标 `pr_auc`：0.2234
- 主评估输出：classifier
- ROC-AUC：0.9705
- PR-AUC：0.2234
- Precision / Recall / F1：0.1905 / 0.9412 / 0.3168
- FPR / FNR：0.0541 / 0.0588
- 峰值CUDA显存：16438.1201 MB

## 环境与产物

- Python：3.10.20（`/root/miniconda3/envs/mqids/bin/python`）
- PyTorch / Transformers：2.4.1+cu121 / 4.51.3
- 设备：NVIDIA GeForce RTX 4080 SUPER
- 机器可读文件：`config.json`、`protocol.json`、`environment.json`、`history.json`。
- 最佳可训练参数：`best_trainable.pt`（若训练已开始）。

## 局限性

- Support and query are chronological portions of the same attack event. 
- 当前F1阈值在同一开发集选择，只能用于诊断；正式测试前必须锁定阈值和后处理。
