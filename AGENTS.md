# 角色与作用域

你是一名资深研究工程师（Research Engineer）。本文件适用于仓库顶层 `moirai_qwen_ids/` 及其全部子目录，用于维护“MOIRAI 变量级时序 Token + 冻结 Qwen”研究路线。

你的职责不是盲目扩大实验，而是：

- 理解检测目标和当前证据边界；
- 优先排查数据泄漏、标签错位、过拟合和不公平比较；
- 通过基线、消融、反事实实验判断各模块是否真的有效；
- 在运行长实验前估算规模并说明实验目的；
- 诚实记录失败结果、资源代价和适用范围；
- 帮助项目形成可靠、可解释、可复现的硕士论文证据链。

本文件是独立仓库 `moirai_qwen_ids/` 后续工作的**主要接手与执行文档**。2026-07-24
迁移后，本目录不再位于旧项目的 `ids/` 下，云端也不应依赖旧仓库目录。若本仓库同级没有旧项目
`final_goal.md` 或上级 `AGENTS.md`，不要把文件缺失当作阻塞；本文件已保留其核心研究、安全和
实验隔离约束。若本地仍能访问旧项目根级文件，它们的最终研究目标仍高于单一技术路线。

如果当前 MOIRAI-Qwen 路线不能带来可靠收益，应主动调整或停止，不得预设 LLM、MOIRAI、
Prompt、Reprogramming 或知识蒸馏必然有效。

# 研究总策略：先构造强候选，再做解释性消融

本项目允许采用“先尽可能找到强方法/SOTA 候选，再逐步消融”的研究路线，但必须严格区分**开发候选**和**测试结论**：

1. 先提出可证伪假设，并根据论文、已有结果和机制设计少量有根据的强候选；
2. 只使用训练集与验证开发协议筛选架构、Prompt、损失、窗口、层和阈值；
3. 在验证阶段找到表现与稳定性较好的候选后冻结主方法，并预注册需要进入正式比较的基线和消融；
4. 消融用于回答各组件是否真的贡献，而不是从测试集反向寻找更高数字；
5. stride=1 验证完成、配置和阈值锁定后，才允许一次性打开 WADI 全量测试；
6. 测试后不得根据测试结果继续修改 Prompt、损失、窗口或阈值再反复测试；若失败，应记录失败并转向新数据集或新协议。

如果需要比较多个“可能的 SOTA 候选”，可以在测试前预注册不超过少量关键候选，并在同一次正式
测试阶段全部评估、全部报告；不得只公布其中最高者，也不得看完测试结果后再追加新候选。任何
“达到/超过 SOTA”的说法必须先核对公开论文的数据版本、切分、点级/事件级指标、point adjustment、
标签对齐和是否使用测试调参。协议不一致时只能称为仓库内统一协议的最佳结果，不能直接声称 SOTA。

“寻找 SOTA”不等于在测试集上挑最高配置。当前单攻击事件验证集容易过拟合，因此强候选必须同时考虑：

- PR-AUC、F1、FPR/FNR 和阈值稳定性；
- seed 间均值、标准差和逐 seed 配对差值；
- 参数量、显存、训练/推理耗时；
- 与无 Qwen、无语义、传统/神经基线的公平性；
- 是否存在数据泄漏、配置漂移或同一攻击事件的后验搜索。

# 当前待验证假设

以下均是研究假设，不是已经成立的事实：

- **H1 时序 Token 假设**：MOIRAI Base 第12层的变量级 Token 比浅层或普通统计表示更适合作为 WADI 连续变量表示。
- **H2 语义对齐假设**：变量自然语言语义与对应 soft Token 逐项相邻，比“文本块+Token块”或只有变量身份 embedding 更有效。
- **H3 离散状态假设**：开关量/低基数变量使用训练期固定的当前状态文本，比直接交给 MOIRAI 或完全删除更适合 Qwen 融合。
- **H4 Qwen 迁移假设**：预训练冻结 Qwen 的提升来自预训练语义/关系建模，而不只是更大容量或随机非线性变换。
- **H5 工艺 Prompt 假设**：正确且具体的 WADI 任务描述应优于通用、错误、打乱或无语义 Prompt；旧的宽泛工控 Prompt 结果不支持该假设，新 Prompt 必须重新验证。
- **H6 损失假设**：普通 CE 对极不平衡开发集的 PR 排序与校准不足，CE 加弱排序损失可能优于只增大 verbalizer CE。

每轮获准实验必须明确主要检验哪一个假设。不要一次同时搜索 Prompt、Projector、损失、窗口和阈值，使结果无法归因。

# 实验授权硬约束

- 未经用户在**当前对话**明确许可，只允许运行：代码阅读、静态检查、编译、`--prepare-only`、合成 smoke、真实模型 `--smoke`。
- 未获许可不得运行任何完整单 seed 开发训练、多 seed、stride=1、全量验证或正式测试，即使 `PLAN.md` 已列出该任务也不算授权。
- 获准执行完整训练时，`scripts/train.py` 还必须显式传入 `--full-run-authorized`；该参数只是防误触门禁，不能替代用户许可。
- 正式 WADI 测试、重新打开测试集或改变已锁定测试协议，必须单独再次获得许可。
- smoke 指标只验证工程闭环，永远不得用于选择方法或写成效果结论。

# 接手顺序

新对话或新 Agent 接手本目录时，必须按顺序阅读：

1. 本文件；
2. `README.md`；
3. `PLAN.md`，确认当前未完成项、阻塞项和最近已勾选任务；
4. `record/dtt_isolation_repair/README.md`，掌握当前严格隔离与 compact Prompt 实现；
5. `record/dtt_ablation/README.md` 和 `record/prompt_counterfactual/README.md`，但只把旧 DTT 当失效诊断；
6. `record/moirai_base_layer_scan/README.md`；
7. `record/direct_window_grid/README.md` 和 `record/l32_validation_diagnostic/README.md`；
8. 若本地存在迁移交接文档 `../moirai_qwen_ids_cloud_handover_2026-07-24.md`，再读该文档核对
   云端机器的实际路径、硬件与环境状态；云端执行不能依赖该仓库外文件；
9. 若工作确实涉及旧基线或 Core-set，才回旧仓库读取相应正式记录，不要把旧仓库当作当前代码依赖。

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
- 第12层与第10层的三种子层选择复验；
- 正确/通用/最简/错误过程 Prompt 的三种子反事实消融；
- 离散变量文本化（DTT）的窗口、层、Projector 和损失开发集消融；
- 当前严格隔离 DTT：80个训练期活跃变量、训练期固定离散状态词表和未知状态规则；
- WADI 标签的 compact/full 自然语言语义映射，同时保留原始变量 ID；
- 连续变量名/语义与对应 `1×1024` soft Token 的逐变量交错输入；
- Qwen 原生 chat template，`add_generation_prompt=True`、`enable_thinking=False`；
- 完整训练授权门禁，未传 `--full-run-authorized` 的非 smoke 训练会直接失败；
- compact DTT 的真实 MOIRAI Base→Direct→Qwen 全栈 smoke；
- 每个运行目录自动生成可阅读的 `results.md`，并支持旧运行批量回填。
- 仅基于正常训练集的 Full 五类混合异常离线生成器，以及可由命令行显式启用的训练集适配器；当前已有 WADI 的 L=16/32/64、每包5000窗、seed 2026 工程数据包。

当前尚未完成：

- 修复后 DTT 的 A/B/C/D 机制对照完整训练；
- compact/full 语义版及正确/打乱/原始ID语义反事实；
- 适配8GB显存的正式 DTT physical batch 与梯度累积方案锁定；
- CE、排序损失、弱 verbalizer 和 focal 控制的严格损失比较；
- stride=1 的锁定验证评分与阈值复核；
- WADI 锁定测试集评估；
- 事件级指标、检测延迟和攻击段级误报/漏报分析；
- SWaT 或其他数据集迁移；
- Student、Teacher–Student 蒸馏、端侧导出或真正轻量部署。

因此，不得把当前实现描述为完整知识蒸馏系统、正式测试通过的 IDS，或已经证明 LLM/Prompt 有效的最终方法。

# 当前方法选择

当前严格隔离的非 DTT 参考候选是（尚未锁定为最终方法）：

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

当前修复后的 DTT 强候选结构是：

```text
WADI 窗口 [B,L,80]
  → 54个连续变量：冻结 MOIRAI Base 第12层 → 每变量[768]
  → DirectProjector: 768→1536→1024
  → 26个离散变量：训练期 scaler 还原原始编码 → 固定的窗口末端当前状态文本
  → compact语义 + 原始变量ID，与每个连续soft Token/离散状态逐变量交错
  → Qwen原生 system/user/assistant chat template，关闭thinking
  → 冻结 Qwen3-0.6B
  → assistant判定位置隐藏状态 → 轻量二分类头
```

compact 示例：

```text
1_AIT_002_PV（第一段·分析仪·过程值），64步：[1×1024 soft token]
1_MV_001_STATUS（第一段·电动阀·状态）=开启
```

完整语义版约3267个 Qwen 位置；第一版 compact 约2670；当前进一步压缩的 compact 约1976。当前 compact 在 L=64、MOIRAI Base第12层、Direct、physical batch=1 的真实全栈 smoke 中峰值显存约4631 MB。该结果只证明可运行，不证明效果。

必须保留的对照包括：

- 冻结 MOIRAI Token + 小型跨变量 Transformer，无 Qwen；
- LinearProjector + 冻结 Qwen + 分类头；
- DirectProjector + 冻结 Qwen + 分类头；
- Reprogramming + 分类头；
- Reprogramming + 分类头 + verbalizer 辅助；
- 纯 verbalizer 输出。

非 DTT 证据不支持把 Reprogramming 作为主方法；旧 DTT 诊断中 Reprogramming 有开发集优势，
但该批结果存在测试 X 依赖和离散文本语义 bug。修复后的主候选先使用 Direct，除非最小机制
对照表明 Reprogramming 有稳定增益，否则不要继续扩大文本原型、注意力头数或原型词数量网格。

# 已验证的核心结果

完整结果见 `record/moirai_base_layer_scan/README.md`、`record/direct_window_grid/README.md`、
`record/l32_validation_diagnostic/README.md`、`record/prompt_counterfactual/README.md` 和
`record/dtt_ablation/README.md`。必须保留以下边界：

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
- 第12层 Qwen 三个 seed 的阈值为 `0.9955/0.9358/0.1520`，概率校准明显不稳定。
- 第12层与第10层三种子 PR-AUC 为 `0.1830±0.0250/0.1264±0.0165`，三个 seed 均由
  第12层领先；该层选择只适用于当前单攻击事件开发协议。
- Prompt 三种子反事实中，正确工控 Prompt PR-AUC 为 `0.1830±0.0250`，错误过程 Prompt为
  `0.2010±0.0138`，最简 Prompt 为 `0.2318±0.1198`。正确 Prompt 没有稳定领先，禁止声称
  Qwen 正在利用正确工控知识推理。
- 旧的85变量、带测试 X 依赖和状态语义 bug 的 DTT 诊断中，DTT ON/OFF PR-AUC 为 `0.1762±0.0078/0.2023±0.0693`；
  该结果不能判断修复后 DTT 是否有效，只能说明旧实现没有稳定提升，且资源成本明显增加。
- 旧 DTT Reprogramming 和等权词表辅助在开发集上有较高均值，但组合后退化；不得将同一单攻击
  事件上的多轮组件搜索当作可泛化增益。

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

2026-07-21 的旧 DTT 入口曾读取 `WADI_TEST_X` 并联合 train/val/test X 决定85个活跃通道；
旧产物因此仍是 transductive 诊断，对应 `test_arrays_opened=false` 字段也不真实。2026-07-23
当前入口已移除测试 X/Y 依赖，并恢复只由正常训练集决定的80个活跃变量。代码修复不会让旧结果
自动有效；只有修复后重新运行的结果才能进入比较。

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

## 合成异常数据（工程能力，尚无效果结论）

当前已实现一套与具体数据集弱耦合的 Full 五类混合异常生成逻辑：

- 实现位于 `mqids/synthetic_anomalies.py`，离线生成入口为 `scripts/generate_synthetic_anomalies.py`；
- 只读取正常训练 X 和变量名，不读取 validation/test X/Y，也不使用真实异常标签；
- 五类算子为 spike、shift/ramp、flatline、soft patch replacement、cross-channel dependency break；
- 连续量与离散量分开处理，离散变量只能取正常训练集中观察到的合法状态；
- 每个样本均为 `[L, C_active]` 的完整因果窗口，扰动区间强制延伸至窗口末端；只有实际改变至少一个通道且 `point_mask[-1] == True` 的样本才会被接受；
- 合成窗口的训练标签固定为1，并与真实窗口相同采用严格端点语义：标签只表示 `Y[s + L - 1]`。中间存在扰动但末端正常的窗口不属于当前合成协议；
- 数据包保存窗口、点/通道 mask、端点标签、源训练端点、算子编号、训练统计量、配置、文件哈希和逐样本 manifest，供复现与审计。

WADI 当前已有以下本地工程数据包，均为80个训练期活跃变量、5000个窗口、seed 2026：

- `synthetic_data/WADI-CLEAN_X_train_full_l16_seed2026`
- `synthetic_data/WADI-CLEAN_X_train_full_l32_seed2026`
- `synthetic_data/WADI-CLEAN_X_train_full_l64_seed2026`

默认生成 WADI 的 L=64 数据包：

```powershell
python scripts/generate_synthetic_anomalies.py
```

迁移到 SWaT 或其他数据集时，显式提供其正常训练数组、变量名、输出目录和窗口长度；生成器输入要求为正常训练二维数值数组 `[time, channels]` 及逐行变量名文件，输出窗口统一保存为 float32：

```powershell
python scripts/generate_synthetic_anomalies.py `
  --input-x path/to/normal_train_x.npy `
  --sensor-names path/to/channel_names.txt `
  --output-dir path/to/synthetic_output `
  --window-length 64 --num-samples 5000 --seed 2026
```

训练默认**不使用**合成异常。只有显式传入 `--use-synthetic-anomalies` 才会加载与当前 L 匹配的数据包：

```powershell
python scripts/train.py --run-name l32_with_synthetic `
  --window-length 32 --use-synthetic-anomalies --prepare-only
```

默认会确定性抽取与真实 support 异常窗相同数量的合成窗（当前非 smoke 协议为170），避免5000个合成窗压倒真实数据。若需改变数量或包路径，使用 `--synthetic-samples N` 和 `--synthetic-data-dir PATH`；例如使用完整 L=16 包：

```powershell
python scripts/train.py --run-name l16_with_all_synthetic `
  --window-length 16 --use-synthetic-anomalies --synthetic-samples 5000 `
  --synthetic-data-dir synthetic_data/WADI-CLEAN_X_train_full_l16_seed2026 `
  --prepare-only
```

`SyntheticWindowDataset` 在拼接训练集前会校验正常训练文件 SHA256、活跃通道顺序与类型、窗口长度、dtype/shape、有限值、末端异常 mask、标签、源端点和 train-only 标记；选中数量、算子分布、包路径与抽样哈希写入运行的 `protocol.json`。这批数据目前只证明生成与训练接口闭环，不能声称能提升检测性能。后续效果实验必须保留完全相同协议下的无合成异常对照，并将是否启用、样本数、窗口长度、seed 和五类算子分布写入实验记录。

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
- DTT 当前使用 Qwen 原生 chat template：system任务定义 → user变量输入 → assistant generation prompt；
- user变量输入按“语义+原始ID文本 → 对应soft Token”逐变量交错，离散变量只插入窗口末端状态；
- 分类使用 assistant 判定位置隐藏状态；verbalizer 在该位置预测下一个“正常/异常”Token；
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

## 损失函数修改思路

当前 CE 是必须保留的基线，但不能预设词表损失一定有效。获准后按少量、预注册的顺序比较：

1. `CE(classifier)`：主基线；
2. `CE + 0.1×verbalizer CE`：只测试弱语义辅助，不直接使用旧失效 DTT 的最佳权重；
3. `CE + λ_rank×pairwise ranking`：直接鼓励异常分数高于正常分数，候选 `λ_rank≈0.1~0.2`；
4. `CE + ranking + 0.1×verbalizer CE`：只有前两种组件分别有效后才测试组合；
5. focal loss 或 class-balanced CE：作为类别不平衡控制，不默认作为主方法。

Pairwise ranking 可写为 `softplus(-(s_pos-s_neg))`，但必须保证同一优化步存在正负样本。当前 DTT
在8GB显存下 physical batch=1 才通过 smoke，不能用 detached 历史分数伪造可微 pair；在实现排序损失前，
必须先设计可运行的正负配对、梯度累积/检查点方案并通过 smoke。不同损失必须使用相同端点、epoch、
checkpoint 规则和主异常分数比较。

# Prompt 研究约束

当前固定 Prompt 只是工程占位和初始方案，不代表工控知识已经证明有效。

声称 Prompt 或工控知识有贡献前，至少比较：

1. 正确的工控任务/过程知识 Prompt；
2. 只有任务描述的通用 Prompt；
3. 删除领域知识的最简 Prompt；
4. 错误、打乱或不相关的过程规则 Prompt；
5. 必要时打乱变量身份/类型嵌入作为反事实控制。

所有 Prompt 对照必须使用相同数据、seed、窗口、模型、训练步数和选模规则。若正确 Prompt 不优于错误/通用 Prompt，不得声称 LLM 在利用工控知识推理。

当前新 DTT Prompt 必须保留以下语义：

- 明确这是 WADI 供水/配水工业控制系统的窗口末端二分类异常检测；
- 连续 soft Token 表示过去到当前共 L 步，不表示未来；
- 离散变量文本只表示当前末端状态，不暗示包含历史；
- 每个变量同时保留自然语言描述和原始 ID；
- 每个连续变量描述必须紧邻它自己的 soft Token，禁止重新退回“所有文本+所有 Token”分块；
- 使用 Qwen 自身 chat template，关闭 thinking，不手工伪造 assistant 角色标记。

语义消融的优先顺序：

1. compact 正确语义 + 原始 ID（当前主候选）；
2. 只有原始 ID，无自然语言语义；
3. 打乱变量之间的自然语言描述，但保持原始 ID 和 Token 不变；
4. full 正确语义，用于检验更多文本是否值得其显存/耗时代价；
5. 通用或错误过程 Prompt 作为反事实，而不是最终候选。

只有正确语义稳定优于“原始ID-only”和“打乱语义”，才能把收益归因于变量语义。compact 若与 full
相当或更好，应优先 compact，因为它在当前 L=64 下约1976位置，而 full 约3267位置。

# 强候选与消融的建议顺序

在用户授权完整开发实验后，优先建立三个候选，而不是扩大所有组件网格：

- **候选A（主创新）**：54连续 MOIRAI Token + 26离散当前状态 + compact变量语义 + Qwen + Direct + CE；
- **候选B（稳健数值）**：全部80变量 MOIRAI数值 Token + Qwen + Direct + CE，不使用 DTT；
- **候选C（无Qwen）**：与A或B使用相同可用输入，但用小型跨变量 Transformer，作为容量/预训练因果对照。

先用单 seed 验证机制方向，再对少量胜者补 seed 2027/2028。主候选形成后，消融顺序建议为：

1. A/B/C，判断 DTT 与 Qwen 是否分别有贡献；
2. compact正确语义 / ID-only / shuffled，判断语义是否有贡献；
3. CE / 排序 / 弱 verbalizer，判断训练目标是否有贡献；
4. Direct / Reprogramming，只在前面机制成立时比较映射复杂度；
5. full Prompt 作为资源—效果权衡，不把更长文本默认视为更强知识。

不得把组件在同一验证攻击上的单项最佳值全部拼成“最终 SOTA”而不重新验证组合；旧实验已经显示
Reprogramming、L=16、词表权重1组合后反而退化，组件收益不具有可加性。

# 目录职责

| 路径 | 职责 |
|---|---|
| `configs/` | 解析后可复现的 JSON 配置；窗口或协议变化使用新配置或完整记录的 CLI override。 |
| `PLAN.md` | 后续实验计划和完成状态；使用 Markdown 复选框维护，完成且复核后才可勾选。 |
| `record/<实验名>/README.md` | 本子项目的多运行正式实验记录；不再写入项目根 `record/`。 |
| `outputs/<run-name>/results.md` | 单次运行自动生成的可阅读结果；与同目录 JSON/checkpoint 配套。 |
| `mqids/paths.py` | 项目相对路径；禁止在模型/数据代码中硬编码机器路径。 |
| `mqids/data.py` | 通道审计、support/query 切分和因果窗口数据集。 |
| `mqids/moirai_tokenizer.py` | 冻结 MOIRAI 变量级 Token 封装。 |
| `mqids/projectors.py` | Linear、Direct、Reprogramming 映射层。 |
| `mqids/prompting.py` | Prompt 拼接、label/prototype Token 处理。 |
| `mqids/semantics.py` | WADI 标签的 compact/full 自然语言解释与可审计语义映射。 |
| `mqids/model.py` | Qwen 主模型和无 Qwen 基线。 |
| `mqids/losses.py` | 分类头和 verbalizer 监督。 |
| `mqids/training.py` | 训练、开发指标、trainable-only checkpoint 和单运行 Markdown。 |
| `scripts/smoke_test.py` | 合成/真实 Qwen 形状和梯度 smoke。 |
| `scripts/inspect_wadi.py` | WADI 通道及单事件切分审计。 |
| `scripts/train.py` | 当前唯一训练入口；2026-07-23 后仅打开训练/验证数据并显式记录文件依赖，旧 DTT 产物仍保留测试 X 依赖边界。 |
| `tests/probe_full_prompt_environment.py` | 云端 full DTT 真实 smoke 与 physical batch 容量探测；只作工程诊断。 |
| `outputs/` | 诊断运行产物，默认 Git 忽略；不能脱离配置和记录引用。 |
| `third_party/` | Time-LLM 改写代码的许可证和来源说明。 |

不要在本目录重新实现或复制大型模型权重。不要修改 `Time-LLM/`、`uni2ts/` 或 `ics-anomaly-detection/`，除非任务确实需要且已说明对数据流和可比性的影响。

# 本地—云端固定工作模式

自 2026-07-24 起，默认采用以下协作模式：

1. Windows 本地负责代码阅读、修改、compile、单元测试、合成 smoke、真实模型 smoke 和
   `--prepare-only`；本地不再承担完整开发训练。
2. 每次改动必须先在本地完成允许范围内的验证，再由用户提交并推送代码。
3. 云端 `/root/moirai_qwen_ids` 只从 Git 更新代码；数据、模型和运行产物不进入 Git。
4. 云端先核对环境、数据与权重，再运行 full Prompt 容量 probe；完整实验使用 probe 推荐的
   physical batch，并通过 `--batch-size`/`--eval-batch-size` 显式记录。
5. Agent 不远程启动完整实验。用户在云端执行已审核命令，随后将整个
   `outputs/<run-name>/` 拉回本地；Agent 再审计配置、协议、逐 epoch 指标、checkpoint 选择和
   `results.md`，并更新 `record/`、`PLAN.md`，必要时更新本文件。
6. 云端运行名必须全局唯一，建议包含候选、语义样式、窗口、层、Projector、损失、seed 和日期。
7. 代码推送不等于实验授权；完整训练、stride=1 和正式测试仍分别遵守本文件的当前对话授权门禁。

当前 CLI 已支持机制对照 A（全部80变量数值 Token）和 C（54连续 Token + 26离散文本）。
计划中的 B（仅连续数值 Token、无文本）和 D（全部数值 Token + 额外离散文本）尚无独立开关；
在实现、smoke 和协议字段完成前，不得用现有参数冒充 B/D，也不得声称 A/B/C/D 已完整运行。

# 运行环境与常用命令

当前 Windows 已验证环境：

```text
C:\Users\83509\Desktop\python project\.venv\Scripts\python.exe
Python 3.11.9
PyTorch 2.4.1+cu121
Transformers 4.51.3
RTX 4060 Laptop 8GB
```

本地可通过 Git 忽略目录中的目录联接复用旧仓库数据和模型，但代码、配置与记录不得保存这些
机器相关绝对路径。所有命令都从当前独立仓库根目录运行，不再带 `ids/moirai_qwen_ids/` 前缀。

从项目根目录运行：

```powershell
$mqPython = 'C:\Users\83509\Desktop\python project\.venv\Scripts\python.exe'

& $mqPython scripts/smoke_test.py
& $mqPython scripts/smoke_test.py --real-qwen --device cuda
& $mqPython scripts/inspect_wadi.py
& $mqPython scripts/train.py `
  --run-name prepare_dtt_l64_l12_direct --prepare-only `
  --window-length 64 --moirai-layer 12 --projector direct `
  --vocab-loss-weight 0 --discrete-to-text --dtt-semantic-style compact
& $mqPython scripts/train.py `
  --run-name smoke_dtt_compact_l64_l12_direct --smoke --device cuda `
  --window-length 64 --moirai-layer 12 --projector direct `
  --vocab-loss-weight 0 --discrete-to-text --dtt-semantic-style compact
```

云端首次部署与工程验证：

```bash
cd /root/moirai_qwen_ids
git status --short
git pull --ff-only origin main
conda create -n mqids python=3.10 -y
conda activate mqids
python -m pip install --upgrade pip
python -m pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 120 --retries 5

mkdir -p data/wadi
cp -a /root/ics-anomaly-detection/exported_data/. data/wadi/

python -m compileall -q mqids scripts tests
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/smoke_test.py
python scripts/smoke_test.py --real-qwen --device cuda
python scripts/inspect_wadi.py
python scripts/train.py \
  --run-name cloud_prepare_dtt_l64_l12_direct_20260724 \
  --prepare-only --window-length 64 --moirai-layer 12 \
  --projector direct --vocab-loss-weight 0 \
  --discrete-to-text --dtt-semantic-style compact
python tests/probe_full_prompt_environment.py \
  --device cuda --batch-sizes 1,2,4,6,8
```

运行前还必须确认权重位于：

```text
/root/moirai_qwen_ids/models/moirai-1.1-R-base/
/root/moirai_qwen_ids/models/Qwen3-0.6B/
```

容量 probe 只有在 `recommended_physical_batch_size` 非空时才算通过。不得选择报告中标为 OOM
或超过15%保留余量的 batch。probe 会更新参数一次，但只使用重复的正常训练窗口，不产生效果结论。

只有用户在当前对话明确授权后，完整训练命令才允许加入门禁参数。例如：

```powershell
& $mqPython scripts/train.py `
  --run-name cloud_dtt_c_compact_l64_l12_direct_ce_seed2026_20260724 `
  --window-length 64 --moirai-layer 12 `
  --projector direct --vocab-loss-weight 0 `
  --discrete-to-text --dtt-semantic-style compact `
  --batch-size 4 --eval-batch-size 4 --seed 2026 `
  --full-run-authorized `
  --device cuda
```

未获授权时禁止复制执行上面的完整训练命令。`--full-run-authorized` 不能由 Agent 自行推断或预先添加。

Linux 云端使用等价命令时去掉 PowerShell 的 `& $mqPython` 和续行反引号，改用 `python` 与
反斜杠续行。运行名必须唯一。训练入口会拒绝覆盖非空运行目录，不要绕过该保护来覆盖可比较产物。

上述候选 C 在云端的等价完整命令为：

```bash
python scripts/train.py \
  --run-name cloud_dtt_c_compact_l64_l12_direct_ce_seed2026_20260724 \
  --window-length 64 --moirai-layer 12 \
  --projector direct --vocab-loss-weight 0 \
  --discrete-to-text --dtt-semantic-style compact \
  --batch-size 4 --eval-batch-size 4 --seed 2026 \
  --full-run-authorized --device cuda
```

完成后应从云端复制整个
`/root/moirai_qwen_ids/outputs/cloud_dtt_c_compact_l64_l12_direct_ce_seed2026_20260724/`
目录，而不是只复制 `results.md`；`outputs/` 被 Git 忽略，不能依靠 `git pull` 回收结果。

# 实验执行顺序

未经用户在当前对话中明确授权，只允许执行编译、静态检查、`--prepare-only` 和
`--smoke`。任何完整训练必须同时满足用户明确许可和命令行 `--full-run-authorized`；
任何 stride=1 锁定评估或正式测试也必须再次获得用户许可。不得把计划文件、历史许可
或代码开关本身解释为本次运行授权。

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

修复后的 DTT 不适用旧的2.4GB参考：当前 compact、L=64、第12层、Direct、约1976个Qwen位置，
physical batch=1 的4+4样本全栈 smoke 峰值约4631 MB；约2670位置、batch=4 已发生 OOM。
云端32GB RTX 4080 必须以 full Prompt probe 实测为准；交接时的 batch=4 只是保守起点，不是
已验证结论。当前代码支持显式 physical/eval batch，不支持梯度累积；在梯度累积实现并 smoke
前不得在记录中声称使用了梯度累积。

# 下一步优先级

当前严格顺序为：

1. 等待用户授权；未授权只允许继续代码审计、prepare-only 和 smoke；
2. 在不改变数据端点的前提下，先锁定 DTT 可运行的 physical batch/梯度累积与记录字段；
3. 完成 A/B/C/D 最小机制对照：全部数值、仅连续、连续+DTT、全部数值+DTT；
4. 根据机制结果形成候选A/B/C，不再大规模搜索旧 DTT 超参数；
5. 对胜者比较 compact正确语义、ID-only、shuffled语义，full仅作资源—效果对照；
6. 在唯一架构上比较 CE、弱 verbalizer、可正确实现的排序损失和 focal控制；
7. 单 seed 明确后只对关键胜者补 seed 2027/2028，并检查阈值和FPR稳定性；
8. 获得单独许可后，在唯一锁定配置上做 stride=1 验证、阈值校准、事件级指标和检测延迟；
9. 测试前冻结配置、checkpoint、阈值、标签规则和后处理；再次获许可后一次性运行 WADI 全量测试；
10. 只有 Teacher/LLM 信号在锁定测试中确实有效后，才讨论 SWaT迁移、Student蒸馏与边缘部署。

不要同时大规模搜索窗口、Prompt、映射层、损失权重和阈值。每轮实验只回答一个清晰问题，防止在唯一验证攻击上过拟合。

# 测试集保护

2026-07-23 已移除 `scripts/train.py` 对 `WADI_TEST_X/WADI_TEST_Y` 的导入和读取；当前
prepare-only 恢复为训练期80个活跃变量，并在协议中显式记录实际打开的文件。旧 DTT
运行仍有测试 X 依赖，不能作为严格结果。修复后的 compact 真实全栈 smoke 已完成，
但 A/B/C/D 机制对照尚未运行，因此仍禁止直接进入 stride=1 锁定或正式测试。

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

每次开始新实验前先检查 `PLAN.md`。实验状态变化时同步维护：

- 新增任务使用 `[ ]`；
- 只有完整运行、产物复核和必要记录都完成后才能改为 `[x]`；
- smoke、部分 seed、失败运行或存在泄漏/配置错误的实验不得勾选为完成；
- 发现阻塞时在对应任务下说明原因和解除条件；
- `PLAN.md` 的勾选状态不得替代 `record/<实验名>/README.md` 的正式结果记录。

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
- 合成异常默认关闭；只有把它作为明确、独立的数据增强变量时才可启用，并必须保留无合成异常对照及记录数据包、样本数、窗口长度、seed 和算子分布。
- 不要在 Teacher 信号尚未锁定前启动 Student 蒸馏。
- 如果新结果否定当前 Direct+Qwen 路线，应更新记录并调整研究方向，而不是继续扩大超参数搜索掩盖失败。

## 2026-07-20 层选择复验结论

- L=64、DirectProjector、冻结 Qwen 的三种子层比较已完成：显式第12层的 PR-AUC/F1 为 `0.1830±0.0250/0.3112±0.0409`，第10层为 `0.1264±0.0165/0.2440±0.0343`；每个对应 seed 的两项主指标均由第12层领先。后续 Prompt 反事实只使用显式第12层，不再继续扩展静态层扫描。
- 此选择仍只适用于同一验证攻击事件的 support→query 开发诊断。第12层平均 FPR 约高 `0.0064`，概率阈值仍不稳定；必须完成 Prompt 对照和 stride=1 锁定，才能打开一次性正式测试。

## 2026-07-22 用户新增实验审计

- Prompt 反事实已完成，结果明确不支持正确工控 Prompt 的稳定语义贡献。
- 旧 DTT 批次维度 bug 已在旧正式运行前修复，但旧离散文本仍存在单窗重编号语义错误；旧85通道结果只允许作为开发诊断。
- 2026-07-23 当前代码已恢复训练期80通道，固定离散状态词表，并实现“自然语言语义+原始ID+对应soft Token”的逐变量布局。
- 新 DTT 使用 Qwen 原生 chat template、关闭 thinking；完整语义版本在L=64下约3267个Qwen位置。
- `dtt_abl_loss_vocab0.1_seed2028` 与 `dtt_abl_loss_verbalizer_seed2027/2028` 不符合计划的 L=64、显式第12层配置，禁止合并成三种子结论。

## 2026-07-23 DTT 修复与 compact Prompt 更新

- 数据隔离、固定状态词表、变量语义映射和逐变量 Token 对齐的工程记录位于 `record/dtt_isolation_repair/README.md`。
- 当前 compact Prompt 示例为 `1_AIT_002_PV（第一段·分析仪·过程值），64步：[soft token]`；离散变量只描述末端当前状态。
- full/第一版compact/当前compact长度约为3267/2670/1976个Qwen位置，当前compact相对full减少约39.5%。
- 2670位置、batch=4的真实全栈 smoke 发生OOM；1976位置、batch=1的 L=64、第12层、Direct、纯分类CE smoke成功，峰值显存约4631 MB。
- smoke只有2正常+2异常训练窗和4开发窗，产生的PR-AUC/F1无效果意义，不得作为候选选择依据。
- 当前没有任何修复后 DTT 的完整开发结果；旧 DTT 数字不能转移到新实现。
- 完整训练已有双重授权门禁：当前对话用户明确许可 + `--full-run-authorized`。未经许可只允许prepare-only与smoke。

## 2026-07-24 独立仓库迁移与本地 smoke

- 当前仓库已从旧 `ids/moirai_qwen_ids/` 分离；运行时数据固定解析到 `data/wadi/`，模型固定解析到
  `models/moirai-1.1-R-base/` 和 `models/Qwen3-0.6B/`，命令不得再带旧目录前缀。
- 迁移后已通过 compileall、9项 unittest、三种 Projector 合成 smoke、真实 Qwen 梯度 smoke、
  WADI prepare-only，以及 L=64/第12层/Direct/compact DTT 的真实 MOIRAI→Qwen 全栈 smoke。
- 本地 smoke 使用旧资产的 Git 忽略目录联接，不改变代码的相对路径协议；smoke 指标无效果意义。
- 代码审计已移除 `infer_channel_metadata` 接收额外数组的接口，防止未来重新引入验证/测试通道选择；
  未知离散状态现在用训练期 scaler 正确还原原始值。
- `train.py` 现在拒绝路径型运行名、互斥 `--prepare-only/--smoke`、无 Qwen DTT 和无 DTT 的语义样式；
  并支持 `--batch-size/--eval-batch-size` 显式覆盖。
- full Prompt 容量 probe 只有满足预留显存比例的 batch 才会推荐；若所有成功 batch 均不满足余量，
  推荐值为 null 并使 probe 失败。
- 云端 PyTorch 2.4.1 的 `torch.cuda.mem_get_info` 要求显式设备号；容量脚本必须把 `cuda`
  解析为 `cuda:<current_device>`。2026-07-24 已修复并通过真实 forward/backward/AdamW probe，
  禁止恢复为无索引 `torch.device("cuda")`。
