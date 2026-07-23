# 角色与作用域

你是一名研究执行助手（Research Assistant），自主性很低。本文件适用于 `ids/moirai_qwen_ids/` 及其全部子目录。

你的职责是：

- 严格按照 CLAUDE.md 和已有实验计划执行命令行实验；
- 运行完成后报告结果（PR-AUC、F1、FPR 等核心指标），不做主观解读；
- 不撰写正式 record/README.md，不提出下一步方向或建议；
- 不自行决定实验策略、不评判路线是否有效、不主动停止实验；
- 对代码的任何修改必须记录到 `实验记录/修改记录.md`，包含日期、涉及文件、变更摘要和变更原因。

本文件是根目录 `AGENTS.md` 的局部补充，不替代根目录规范。实验方向和优先级由已有的 CLAUDE.md 计划和用户指令决定。

# 接手顺序

新对话或新 Agent 接手本目录时，必须按顺序阅读：

1. 项目根目录 `final_goal.md`；
2. 项目根目录 `AGENTS.md`；
3. 本文件；
4. `README.md`；
5. `record/moirai_base_layer_scan/README.md`；
6. `record/direct_window_grid/README.md` 和 `record/l32_validation_diagnostic/README.md`；
7. 若涉及基线比较，再读项目根目录 `record/wadi_baseline_comparison.md`；
8. 若涉及旧 MOIRAI Core-set 路线，再读项目根目录 `record/wadi_moirai_small_formal.md` 和 `record/wadi_moirai_realtime_coreset_validation_ablation.md`。

不要只根据 `outputs/` 中某个最高数字形成结论。必须结合解析后的配置、数据协议、逐 epoch 历史、checkpoint 选择规则和正式记录解释。

# 子项目目标与真实状态

本子项目研究以下假设：

> 将 MOIRAI 作为多变量时序 Tokenizer，每个有效工控变量产生一个连续 Token；通过轻量映射层投影到 Qwen3-0.6B 词向量空间，利用冻结 Qwen 完成异常分类或正常/异常 verbalizer 判断。

当前已经实现：

- WADI 数据审计、恒定变量剔除和因果滑动窗口；
- 验证集唯一攻击事件的 support/query 时间隔离；
- 冻结 MOIRAI Base 的变量级 context Token 提取；
- Linear、Direct、Reprogramming 三种映射层；
- 冻结 Qwen3-0.6B 的连续 Token 输入；
- 轻量二分类头与“正常/异常”双 Token verbalizer；
- 分类头损失、词表辅助损失和纯 verbalizer 消融；
- 无 Qwen 小 Transformer 基线与参数匹配基线；
- 配置、环境、协议、逐 epoch 指标、资源成本和仅训练参数 checkpoint 保存；
- L=32 的结构消融和 Direct/无 Qwen 三种子验证诊断；
- 统一端点协议下 `L=8/16/32/64/128` 的 Direct 单种子窗口筛选；
- L=64 下 MOIRAI Base 第1至12层的统一端点筛选；
- 第12层 Direct 与无 Qwen 对照的 seed 2026/2027/2028 配对复验；
- 每个运行目录自动生成可阅读的 `results.md`，并支持旧运行批量回填。

当前尚未完成：

- 第二名第10层的补充种子复验；
- Prompt 正确/通用/错误或打乱反事实消融；
- stride=1 的锁定验证评分与阈值复核；
- WADI 锁定测试集评估；
- 事件级指标、检测延迟和攻击段级误报/漏报分析；
- SWaT 或其他数据集迁移；
- Student、Teacher–Student 蒸馏、端侧导出或真正轻量部署。

因此，不得把当前实现描述为完整知识蒸馏系统、正式测试通过的 IDS，或已经证明 LLM/Prompt 有效的最终方法。

# 当前方法选择

当前主候选是：

```text
WADI 窗口 [B,L,C]
  → 删除仅由正常训练集判定的恒定变量
  → 冻结 MOIRAI Base
  → 每个变量一个 context Token [B,C_active,768]
  → DirectProjector: LN → 768→1536→1024 MLP → RMSNorm
  → 变量身份嵌入 + 变量类型嵌入
  → 固定任务 Prompt 前缀 + 连续 Token + 判定后缀
  → 冻结 Qwen3-0.6B
  → 最后判定位置隐藏状态
  → 轻量二分类头
```

主异常分数暂定为二分类头的 `P(异常)`。

必须保留的对照包括：

- 冻结 MOIRAI Token + 小型跨变量 Transformer，无 Qwen；
- LinearProjector + 冻结 Qwen + 分类头；
- DirectProjector + 冻结 Qwen + 分类头；
- Reprogramming + 分类头；
- Reprogramming + 分类头 + verbalizer 辅助；
- 纯 verbalizer 输出。

当前证据不支持把 Reprogramming 作为主方法。除非新窗口或 Prompt 反事实实验显示稳定优势，否则不要继续扩大文本原型、注意力头数或原型词数量网格。

# 已验证的核心结果

完整结果见 `record/moirai_base_layer_scan/README.md`、`record/direct_window_grid/README.md`
和 `record/l32_validation_diagnostic/README.md`。必须保留以下边界：

- 实验是 WADI 验证集单攻击事件 support→query 诊断，不是测试结果，也不是无监督结果。
- DirectProjector + Qwen 三种子开发 PR-AUC 为 `0.1646±0.0227`，F1 为 `0.2827±0.0377`。
- 无 Qwen hidden=256 三种子开发 PR-AUC 为 `0.1028±0.0222`，F1 为 `0.2058±0.0347`。
- Direct 的逐 seed PR-AUC 配对提升均值为 `+0.0617±0.0437`，但平均 FPR 增加约 `0.0192`。
- Direct 峰值显存约 2401 MB，无 Qwen 基线约 495 MB；五轮训练+开发耗时约 564 s 对 184 s。
- 参数匹配无 Qwen hidden=384 在 seed 2026 的 PR-AUC 只有 0.0392，增加容量没有解释 Direct 的提升。
- Seed 2026 下 Reprogramming-head PR-AUC 为 0.1444，低于 Direct 的 0.1579。
- `0.1 × verbalizer CE` 能显著改变词表方向，但主分类 PR-AUC 只有 0.1313，没有改善主输出。
- 纯 verbalizer seed 2026 PR-AUC 为 0.1472，存在信号但训练波动大，应作为消融而非当前主方法。
- 统一 `common_min_endpoint=127` 后，Direct seed 2026 的 L=8/16/32/64/128 开发
  PR-AUC 分别为 `0.0817/0.0662/0.1376/0.2139/0.1212`；当前窗口候选为 L=64。
- L=64 同一最佳 PR epoch 的 ROC-AUC/F1/FPR 为 `0.9729/0.3626/0.0325`，但最佳阈值
  `0.9955` 暴露出概率校准风险，不能直接迁移到测试集。
- L=64、seed 2026 的 MOIRAI Base 第1至12层 PR-AUC 依次为
  `0.0706/0.1033/0.0855/0.0535/0.0403/0.0860/0.0952/0.1013/0.1039/0.1447/0.1112/0.2139`；
  第12层是当前单种子全层筛选胜者，第10层是第二名。
- 第12层 + Qwen 三种子开发 PR-AUC/F1 为 `0.1857±0.0288/0.3134±0.0446`；相同
  第12层 Token 的无 Qwen 对照为 `0.0835±0.0595/0.1750±0.1044`。逐 seed Qwen 的 PR-AUC
  均更高，配对增益 `+0.1022±0.0371`，但资源成本约为 2401 MB 对 496 MB、672 s 对 204 s。
- 第12层 Qwen 三个 seed 的阈值为 `0.9955/0.9358/0.1520`，概率校准明显不稳定；且只有
  第12层做了多种子，尚未证明其均值一定高于第10层均值。

这些数字只允许表述为“值得继续验证的开发集信号”。不能据此声称超过既有 AE 正式测试基线、实现跨攻击泛化，或证明工控知识 Prompt 有贡献。

# 数据与切分约束

当前只围绕 WADI 工作。所有路径必须从 `mqids/paths.py` 解析，不要在源码中硬编码当前机器绝对路径。

## WADI 文件

- 正常训练：`WADI-CLEAN_X_train.npy`；
- 验证：`WADI-CLEAN_X_test_val.npy` 与 `WADI-CLEAN_Y_test_val.npy`；
- 锁定测试：`WADI-CLEAN_X_test_new.npy` 与 `WADI-CLEAN_Y_test_new.npy`；
- 变量名：`WADI-CLEAN_sensor_cols.txt`。

X/Y 必须成对使用且长度一致。训练和诊断脚本不得为了方便导入测试文件。

## 通道审计

当前正常训练集审计结果：

- 原始变量 110 个；
- 训练恒定变量 30 个；
- 有效变量 80 个；
- 其中连续变量 54 个、二值变量 16 个、低基数变量 10 个。

恒定变量只能根据正常训练集判断，不能使用验证或测试数据决定通道。代码应按数据自动推断并保存 `channel_metadata.json`，不要无条件硬编码 80。

## 当前单事件监督协议

WADI 验证集唯一攻击段为 `[5139,6625)`：

- support 使用攻击前半段，异常端点 stride=4；
- support/query 隔离区为 `[5818,5946)`，宽度 128；
- query 使用攻击后半段；
- 每轮训练采样 510 个正常训练窗口和 170 个异常 support 窗口；
- 窗口筛选统一 `common_min_endpoint=127`，五种 L 的正常、异常和开发端点 SHA256 必须一致；
- 当前窗口筛选开发诊断 stride=8，共 6364 个端点，其中异常 85 个；旧 L=32 诊断从端点 31
  开始、共 6376 个端点，只能作为旧协议结构消融，不能与新窗口网格直接比较；
- 每个窗口标签严格使用窗口末端 `Y[s + L - 1]`。

该协议只能称为“单攻击事件监督迁移”或“support→query 诊断”。support/query 来自同一攻击事件，即使有隔离区，也不能称为独立攻击泛化。

如果未来更改 support 比例、隔离区、正常/异常采样比、stride 或端点标签，必须使用新运行名并单独记录，不能与当前结果直接混合。

## 窗口与 MOIRAI Token

- 只允许用户指定的原生窗口/patch：`8,16,32,64,128`；
- 一变量一 Token 模式必须满足 `window_length == patch_size`；
- 只选择 `prediction_mask == false` 的历史 context Token；
- 不能把未来预测占位 Token 混入表示；
- 不能重新对全部变量 Token 做全局单向量池化后再声称是当前方法；
- Token 必须按 `variate_id` 排序，并验证每个有效变量恰好一个 context Token。

真实 smoke 已验证 L=32 和 L=128 均能保持一变量一 Token；L=32 时输入 `[1,32,80]`，
MOIRAI 输出 `[1,80,768]`，映射后 `[1,80,1024]`。

# 模型与梯度约束

## MOIRAI

- 默认使用本地原始 MOIRAI Base：`ids/model/moirai-1.1-R-base/`；
- 当前不加载 LoRA；
- MOIRAI 参数全部冻结并保持 eval 模式；
- MOIRAI 前向可以使用 `torch.no_grad()`，因为 Teacher 和窗口输入都不训练；
- 不修改 `uni2ts/` 上游源码来适配本实验，优先在 `mqids/moirai_tokenizer.py` 封装。

## Qwen

- 使用本地 `ids/model/Qwen3-0.6B/`；
- Qwen 参数全部冻结并保持 eval 模式；
- **不能**用 `torch.no_grad()` 包住 Qwen 前向，因为梯度必须穿过冻结 Qwen 回到映射层；
- 使用 `inputs_embeds`，关闭 `use_cache`，不进行文本生成；
- 当前输入顺序为 Prompt 前缀 → 连续时序 Token → 判定后缀；分类使用最后一个后缀位置隐藏状态；
- “正常”“异常”必须分别验证为单 Token；不要改成存在问句极性歧义的“是/否”而不做独立消融；
- verbalizer 概率只在“正常/异常”两个 label logit 之间归一化，不把它误称为完整词表校准概率。

真实 smoke 必须满足：

- 映射层存在有限非零梯度；
- Qwen 梯度张量数量为 0；
- MOIRAI 梯度张量数量为 0；
- 所有输出有限且形状正确。

## 映射层

- `LinearProjector` 是最严格维度对齐基线；
- `DirectProjector` 是当前主候选，Base 配置为 `768→1536→1024`；
- `ReprogrammingProjector` 使用 Direct 残差和固定 Qwen 原型交叉注意力；
- 不得复用 Time-LLM 的 `vocab_size→num_tokens` 巨大映射，Qwen 词表规模下会产生不合理参数量；
- Reprogramming 代码已复制并改写到本目录，不运行时依赖克隆的 `Time-LLM/`；
- 修改来自 Time-LLM 的代码时保留 Apache-2.0 归属，参考 `third_party/TIME_LLM_NOTICE.md`。

# 损失、选模与阈值规范

主双目标形式为：

```text
L = classifier_loss_weight × CE(classifier_logits, y)
  + vocab_loss_weight × CE([normal_logit, anomaly_logit], y)
```

约束：

- 主分类方法默认 `classifier_loss_weight=1`；
- Head-only 消融设置 `vocab_loss_weight=0`；
- 当前双目标消融设置 `vocab_loss_weight=0.1`；
- 纯 verbalizer 设置 `classifier_loss_weight=0, vocab_loss_weight=1`，并冻结分类头；
- 只有正常样本时不能训练该二分类目标；若改为正常数据训练，必须设计并命名为独立的一类/自监督目标，不能伪装成正常/异常概率监督。

模型选择规则：

- 分类头为主输出时，按开发 `pr_auc` 最高 epoch 选 checkpoint；
- 纯 verbalizer 时，按开发 `vocab_pr_auc` 最高 epoch 选 checkpoint；
- F1、Precision、Recall、FPR 必须来自被选中的同一个 epoch；
- 不得按 PR-AUC 选一个 epoch，再从另一个 epoch 抄最高 F1；
- 阈值只允许在训练/验证协议中确定；测试集阈值必须锁定，禁止测试标签调参；
- 类别极不平衡时以 PR-AUC、F1、FPR、FNR 为主，不能只看 ROC-AUC。

当前开发异常比例只有约 1.33%，随机 PR 基线约为 0.0133。任何 PR-AUC 都必须结合这个先验、FPR 和检测覆盖解释。

# Prompt 研究约束

当前固定 Prompt 只是工程占位和初始方案，不代表工控知识已经证明有效。

声称 Prompt 或工控知识有贡献前，至少比较：

1. 正确的工控任务/过程知识 Prompt；
2. 只有任务描述的通用 Prompt；
3. 删除领域知识的最简 Prompt；
4. 错误、打乱或不相关的过程规则 Prompt；
5. 必要时打乱变量身份/类型嵌入作为反事实控制。

所有 Prompt 对照必须使用相同数据、seed、窗口、模型、训练步数和选模规则。若正确 Prompt 不优于错误/通用 Prompt，不得声称 LLM 在利用工控知识推理。

# 目录职责

| 路径 | 职责 |
|---|---|
| `configs/` | 解析后可复现的 JSON 配置；窗口或协议变化使用新配置或完整记录的 CLI override。 |
| `record/<实验名>/README.md` | 本子项目的多运行正式实验记录；不再写入项目根 `record/`。 |
| `outputs/<run-name>/results.md` | 单次运行自动生成的可阅读结果；与同目录 JSON/checkpoint 配套。 |
| `mqids/paths.py` | 项目相对路径；禁止在模型/数据代码中硬编码机器路径。 |
| `mqids/data.py` | 通道审计、support/query 切分和因果窗口数据集。 |
| `mqids/moirai_tokenizer.py` | 冻结 MOIRAI 变量级 Token 封装。 |
| `mqids/projectors.py` | Linear、Direct、Reprogramming 映射层。 |
| `mqids/prompting.py` | Prompt 拼接、label/prototype Token 处理。 |
| `mqids/model.py` | Qwen 主模型和无 Qwen 基线。 |
| `mqids/losses.py` | 分类头和 verbalizer 监督。 |
| `mqids/training.py` | 训练、开发指标、trainable-only checkpoint 和单运行 Markdown。 |
| `scripts/smoke_test.py` | 合成/真实 Qwen 形状和梯度 smoke。 |
| `scripts/inspect_wadi.py` | WADI 通道及单事件切分审计。 |
| `scripts/train.py` | 当前唯一训练入口；不会主动打开 WADI 测试集。 |
| `outputs/` | 诊断运行产物，默认 Git 忽略；不能脱离配置和记录引用。 |
| `third_party/` | Time-LLM 改写代码的许可证和来源说明。 |

不要在本目录重新实现或复制大型模型权重。不要修改 `Time-LLM/`、`uni2ts/` 或 `ics-anomaly-detection/`，除非任务确实需要且已说明对数据流和可比性的影响。

# 运行环境与常用命令

当前 Windows 已验证环境：

```text
C:\Users\83509\Desktop\python project\.venv\Scripts\python.exe
Python 3.11.9
PyTorch 2.4.1+cu121
Transformers 4.51.3
RTX 4060 Laptop 8GB
```

`main.py` 的历史 MOIRAI 实验也使用这个外部虚拟环境。不要重新下载重复的 Torch/Uni2TS 环境；仓库内 `.venv` 是 macOS 环境，`.venv-win` 不是本子项目的首选正式环境。

从项目根目录运行：

```powershell
$mqPython = 'C:\Users\83509\Desktop\python project\.venv\Scripts\python.exe'

& $mqPython ids/moirai_qwen_ids/scripts/smoke_test.py
& $mqPython ids/moirai_qwen_ids/scripts/smoke_test.py --real-qwen --device cuda
& $mqPython ids/moirai_qwen_ids/scripts/inspect_wadi.py
& $mqPython ids/moirai_qwen_ids/scripts/train.py `
  --run-name prepare_check --prepare-only
& $mqPython ids/moirai_qwen_ids/scripts/train.py `
  --run-name real_stack_smoke --smoke --device cuda
```

当前 Direct L=32 训练示例：

```powershell
& $mqPython ids/moirai_qwen_ids/scripts/train.py `
  --run-name diag_l32_direct_seed2026 `
  --projector direct `
  --vocab-loss-weight 0 `
  --seed 2026 `
  --device cuda
```

运行名必须唯一。训练入口会拒绝覆盖非空运行目录，不要绕过该保护来覆盖可比较产物。

# 实验执行顺序

任何涉及特征提取、Prompt 拼接、标签、损失、评分或阈值的修改，必须按顺序执行：

1. `python -m compileall` 或等价静态语法检查；
2. 合成 smoke；
3. 真实 Qwen 梯度 smoke；
4. 真实 MOIRAI 一窗一 Token 形状检查；
5. `train.py --prepare-only` 数据协议检查；
6. `train.py --smoke` 最小训练闭环；
7. 单 seed 开发诊断；
8. 关键配置多 seed；
9. stride=1 锁定验证；
10. 协议全部锁定后，才允许一次正式测试。

运行长实验前必须估算窗口数、batch 数、显存和耗时。当前参考：

- 无 Qwen 五轮训练+开发约 3 分钟，峰值约 495 MB；
- Direct+Qwen 五轮训练+开发约 9--10 分钟，峰值约 2.4 GB；
- Reprogramming 与 verbalizer 配置约 9--10 分钟。
- 统一端点网格中 Direct 的五轮累计训练+开发耗时约为 L=8 的 607 s 到 L=128 的 748 s；
  L=64 为 714 s，峰值仍约 2401 MB。

这些耗时只适用于当前 RTX 4060、训练 680 窗口、开发 stride=8 的诊断协议。

# 下一步优先级

当前严格顺序为：

1. 补跑第二名第10层的 seed 2027/2028，与第12层做跨 seed 层选择确认；
2. 固定胜出层和 L=64，做正确/通用/删除/错误 Prompt 反事实消融；
3. 对关键 Prompt 对照做多种子复验，不能只报告单 seed；
4. 在 stride=1 验证端点上重新生成分数并锁定阈值，同时检查概率校准；
5. 在测试前冻结配置、checkpoint、阈值、标签规则和后处理；
6. 一次性运行 WADI 测试，并报告原始点级、事件级/point-adjustment、攻击段检出、误报、漏报、延迟和资源成本；
7. 只有 Teacher/LLM 信号在锁定测试中确实有效后，才讨论 Student 蒸馏与边缘部署。

不要同时大规模搜索窗口、Prompt、映射层、损失权重和阈值。每轮实验只回答一个清晰问题，防止在唯一验证攻击上过拟合。

# 测试集保护

当前 `scripts/train.py` 的重要安全性质是：训练和开发阶段不打开 `WADI_TEST_X/WADI_TEST_Y`。

在新增正式评估脚本前，必须先明确：

- checkpoint 按哪个验证指标选择；
- stride=1 验证阈值；
- 分类头还是 verbalizer 是正式分数；
- 是否存在报警确认或后处理；
- 点级和事件级指标定义；
- 检测延迟计算方式；
- 测试输出和记录路径。

正式测试脚本不得包含阈值搜索、epoch 选择、Prompt 选择、窗口选择或基于测试标签的后处理选择。若测试失败，应诚实记录，不能回到测试集继续调参。

# 实验记录规范

每次运行必须保存：

- 解析后的完整配置；
- Python/Torch/Transformers/CUDA/GPU 环境；
- support/query、guard、窗口、stride、样本数和标签规则；
- channel metadata；
- 每 epoch 训练损失和开发指标；
- 主选模指标与选中 epoch；
- 阈值、Precision、Recall、F1、ROC-AUC、PR-AUC、FPR、FNR；
- verbalizer 消融对应指标；
- 峰值显存和耗时；
- 仅包含可训练参数的最佳 checkpoint。
- 每个运行目录必须有由代码生成的可阅读 `results.md`；机器可读 JSON 不能作为唯一结果入口。

未完成的 smoke 或单次中间运行只保存在 `outputs/`，即使存在自动 `results.md` 也不得写成正式结论。

完成一个完整实验后，在本子项目 `record/<实验名>/README.md` 新增或更新汇总，包含目的、数据切分、配置、随机种子、运行环境、耗时、核心指标、局限性和产物路径；本子项目不再向项目根 `record/` 写记录。完成多配置消融、窗口筛选、多种子复验或正式测试等较大实验后，还应把可迁移的方法学结论同步到项目根 `AGENTS.md` 和本文件。

# 工程与协作约束

- 工作树包含用户未提交代码和实验产物；保留所有无关改动。
- 禁止使用 `git reset --hard`、`git checkout --` 或批量删除清理工作树。
- 本地源码编辑使用小范围、可审计修改；不要无理由重写整个训练框架。
- 配置变化必须保存解析后的结果和唯一运行名，不能悄然改变默认值后复用旧产物。
- checkpoint 默认只保存可训练参数，避免重复保存冻结 Qwen/MOIRAI 权重。
- 不要把 smoke 的 4 样本指标当作实验结果。
- 不要把某个 seed、epoch 或阈值下的最高 F1 单独宣传；关键结论报告均值和波动。
- 不要使用合成异常，除非后续将其作为明确、独立的增强实验，并保留无合成异常对照。
- 不要在 Teacher 信号尚未锁定前启动 Student 蒸馏。
- 如果新结果否定当前 Direct+Qwen 路线，应更新记录并调整研究方向，而不是继续扩大超参数搜索掩盖失败。

## 2026-07-20 层选择复验结论

- L=64、DirectProjector、冻结 Qwen 的三种子层比较已完成：显式第12层的 PR-AUC/F1 为 `0.1830±0.0250/0.3112±0.0409`，第10层为 `0.1264±0.0165/0.2440±0.0343`；每个对应 seed 的两项主指标均由第12层领先。后续 Prompt 反事实只使用显式第12层，不再继续扩展静态层扫描。
- 此选择仍只适用于同一验证攻击事件的 support→query 开发诊断。第12层平均 FPR 约高 `0.0064`，概率阈值仍不稳定；必须完成 Prompt 对照和 stride=1 锁定，才能打开一次性正式测试。
