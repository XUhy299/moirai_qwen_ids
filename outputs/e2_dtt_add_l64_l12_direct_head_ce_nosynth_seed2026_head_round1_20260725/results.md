# e2_dtt_add_l64_l12_direct_head_ce_nosynth_seed2026_head_round1_20260725

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
| 1 | 0.4163 | 0.9239 | 0.0909 | 0.1078 | 0.4706 | 0.1754 | 0.0527 | 0.1603 | 141.8125 | 490.6547 |
| 2 | 0.2930 | 0.9686 | 0.1721 | 0.2066 | 0.6588 | 0.3146 | 0.0342 | 0.9826 | 141.4209 | 512.6858 |
| 3 | 0.2671 | 0.9681 | 0.1936 | 0.1723 | 0.9529 | 0.2919 | 0.0620 | 0.9289 | 135.6202 | 517.3768 |
| 4 | 0.2433 | 0.9684 | 0.1756 | 0.1924 | 0.8941 | 0.3167 | 0.0508 | 0.9938 | 142.1675 | 518.3829 |
| 5 | 0.2757 | 0.9745 | 0.2241 | 0.2249 | 0.7647 | 0.3476 | 0.0357 | 0.8775 | 143.6259 | 515.3218 |

## 当前最佳

- Epoch：5
- 选模指标 `pr_auc`：0.2241
- 主评估输出：classifier
- ROC-AUC：0.9745
- PR-AUC：0.2241
- Precision / Recall / F1：0.2249 / 0.7647 / 0.3476
- FPR / FNR：0.0357 / 0.2353
- 峰值CUDA显存：16365.1147 MB

## 环境与产物

- Python：3.10.20（`/root/miniconda3/envs/mqids/bin/python`）
- PyTorch / Transformers：2.4.1+cu121 / 4.51.3
- 设备：NVIDIA GeForce RTX 4080 SUPER
- 机器可读文件：`config.json`、`protocol.json`、`environment.json`、`history.json`。
- 最佳可训练参数：`best_trainable.pt`（若训练已开始）。

## 局限性

- Support and query are chronological portions of the same attack event. 
- 当前F1阈值在同一开发集选择，只能用于诊断；正式测试前必须锁定阈值和后处理。
