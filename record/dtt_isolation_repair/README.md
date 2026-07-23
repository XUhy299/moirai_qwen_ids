# DTT 数据隔离、状态语义与 Prompt 输入修复

日期：2026-07-23

## 目的

修复旧 DTT 实现中测试 X 参与通道选择、协议字段失真、离散状态按单窗口重编号，以及变量名与连续 soft Token 未逐项对齐的问题。本记录是工程与协议验证，不是效果实验。

## 修复内容

- `scripts/train.py` 不再导入或打开 `WADI_TEST_X/WADI_TEST_Y`；正常训练、验证 X/Y、传感器名和训练期 scaler 是开发入口唯一数据依赖。
- 活跃通道和变量类型只由正常训练集确定，恢复为80个活跃变量：54个连续、16个二值、10个低基数。
- 使用训练期 `WADI-CLEAN_scaler.pkl` 将离散变量的标准化值还原为原始编码，并为每个变量建立固定词表；未知状态显式输出，禁止按窗口重新编号。
- 离散文本只读取窗口末端值，不包含切换次数、持续时间或历史状态。
- 连续变量按“变量语义+原始ID→对应 MOIRAI soft Token”逐项交错插入 Qwen；每个 Token 维度为1024。
- 使用 Qwen tokenizer 原生 chat template，`add_generation_prompt=True`、`enable_thinking=False`。
- 语义描述提供 `compact/full` 两种样式，默认 compact；完整映射和 SHA256 写入 `variable_semantics.json`。
- 非 smoke、非 prepare-only 训练必须同时获得用户明确授权并传入 `--full-run-authorized`。

## Prompt 形态

压缩语义示例：

```text
1_AIT_002_PV（第一段·分析仪·过程值），64步：[对应的1×1024 soft token]
1_MV_001_STATUS（第一段·电动阀·状态）=开启
```

L=64 时，完整语义版约3267个 Qwen 位置；第一版 compact 约2670；进一步压缩后的当前 compact 约1976，比完整语义版减少约39.5%。

## 验证

使用环境：Python 3.11.9、PyTorch 2.4.1+cu121、Transformers 4.51.3、RTX 4060 Laptop GPU。

已通过：

- `python -m compileall`；
- 三种 Projector 合成形状与梯度 smoke；
- 离散历史改变、末端状态不变时文本保持不变；
- 本地 Qwen 原生 chat template 渲染；
- 真实 Qwen 的 variable-aligned DTT 梯度 smoke；
- prepare-only：80个活跃变量，且协议中的 test X/Y 均为 `null`；
- 未带 `--full-run-authorized` 的完整训练被拒绝；
- 真实全栈 smoke：L=64、MOIRAI Base第12层、Direct、分类CE、batch=1，4训练窗+4开发窗完成。

真实全栈 smoke 峰值 CUDA 显存为4630.72 MB，训练/开发耗时约3.68/1.12秒（不含模型加载）。4样本产生的 PR-AUC/F1 不具有任何效果解释意义。

## 失败诊断

第一版 compact 约2670位置、batch=4 的真实全栈 smoke 发生 CUDA OOM。当前版本缩短为约1976位置，并仅在 smoke 中使用 batch=1 后通过。未来获准的正式 DTT 实验必须单独锁定并记录 batch，不得默认沿用旧 batch=4。

## 产物

- prepare-only：`outputs/dtt_compact_prepare_20260723/`
- 成功全栈 smoke：`outputs/dtt_compact_v2_l64_l12_direct_real_smoke_20260723/`
- OOM 诊断：`outputs/dtt_compact_l64_l12_direct_real_smoke_20260723/`

## 结果边界

- 旧 DTT 数值不会因代码修复而自动有效，仍只能作为带测试 X 依赖和错误状态语义的诊断。
- 本次没有运行完整开发实验、stride=1评估或正式测试。
- 修复只证明数据与梯度闭环可运行，不证明 compact/full Prompt、DTT、Qwen或词表损失能提高检测效果。
