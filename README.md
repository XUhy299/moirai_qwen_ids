# MOIRAI-Qwen IDS

Independent research implementation for using frozen MOIRAI as a variable-level
time-series tokenizer and frozen Qwen3-0.6B as a reasoning backbone for WADI
anomaly classification.

This directory does not import code from the cloned `Time-LLM` repository.
The reprogramming attention implementation in `mqids/projectors.py` is adapted
from Time-LLM under Apache-2.0; see `third_party/TIME_LLM_NOTICE.md`.

## Self-contained layout and portability

All non-environment runtime assets are stored below this directory and are
resolved from the location of `mqids/paths.py`, never from the working
directory or a machine-specific absolute path. Uni2TS is a pinned virtual-
environment dependency rather than vendored source code:

```text
data/wadi/                 WADI train, validation, locked-test arrays and metadata
models/moirai-1.1-R-base/  local MOIRAI Base checkpoint
models/Qwen3-0.6B/         local Qwen3-0.6B checkpoint
```

The repository may therefore be moved as one directory to Linux. On a Linux
x86_64 host with Python 3.11, a CUDA 12.1-compatible NVIDIA driver, and an
empty virtual environment, run `python -m pip install -r requirements.txt`.
This installs the pinned CUDA PyTorch, Uni2TS 2.0.0 and all Uni2TS transitive
dependencies. Run all commands from this directory with `python scripts/...`.

### Full-prompt cloud smoke and batch probe

After copying the WADI package into `data/wadi/` and the two local checkpoints
into `models/`, run the following engineering-only check before any experiment:

```bash
python tests/probe_full_prompt_environment.py --device cuda
```

It checks the installed GPU stack, runs the authorized four-window real-model
smoke with DTT `full` semantics, then performs one real forward/backward/update
step at physical batches `1,2,4,8`. The probe uses one repeated normal-training
window, never opens locked test data, and writes its resource report under
`outputs/`. It is a capacity diagnostic, not a detection result.

## Current scope

- WADI only.
- Window/patch sizes are restricted to `8, 16, 32, 64, 128`.
- One MOIRAI context token per retained WADI variable.
- Frozen MOIRAI and frozen Qwen3-0.6B.
- Trainable projector, variable/type embeddings and a two-class head.
- The `正常/异常` Qwen verbalizer objective is auxiliary; the classifier head is
  the primary anomaly score.
- Optional train-only synthetic anomalies; no LoRA, Student distillation or
  test-set tuning.

The first supervised experiment uses part of WADI's single validation attack as
a support set. It must be reported as **single-attack-event supervised transfer**,
not as unsupervised anomaly detection. Prompt 消融阶段没有访问正式测试划分。

2026-07-21 的旧 DTT 入口曾读取正式测试 X 来决定通道集合；这些旧产物仍只能作为
transductive 开发诊断。2026-07-23 已将当前入口改为只由正常训练集确定80个活跃通道，
并用训练期 scaler 恢复离散原始编码、建立固定状态词表；修复后的真实全栈训练尚未完成。

当前进展：L=64、第12层已通过三种子复验；四类 Prompt 三种子反事实已经完成，但不支持
正确工控 Prompt 的稳定语义优势；旧 DTT 多组消融存在测试 X 依赖和单窗状态重编号 bug。
新 DTT 已实现变量语义+原始ID、逐变量 soft Token 对齐和 Qwen 原生 chat template，但尚无
修复后的效果结果。完整旧结果审计见 `record/prompt_counterfactual/README.md` 和
`record/dtt_ablation/README.md`。

DTT 语义描述支持 `compact/full` 两种样式，默认 `compact`。完整训练受到授权门禁保护：
未经用户明确许可只能运行 `--prepare-only` 或 `--smoke`；获准后的非 smoke 训练还必须
显式传入 `--full-run-authorized`，避免误启动长实验。

## Environment

Use a Python environment that can load both local Uni2TS and Qwen3. Qwen3
requires `transformers>=4.51`; do not install Time-LLM's old Transformers pin.

From this directory (PowerShell, Windows):

```powershell
$python = "python"  # or the path to your virtual-environment interpreter
& $python scripts/smoke_test.py
& $python scripts/inspect_wadi.py
& $python scripts/train.py --run-name prepare_check --prepare-only
& $python scripts/train.py --run-name real_stack_smoke --smoke --device cuda
```

The synthetic smoke test checks tensor shapes, frozen-backbone gradients and all
three projectors without loading large weights. Add `--real-qwen` to also use the
local Qwen checkpoint. A real MOIRAI run requires the complete Uni2TS dependency
environment.

After the real-model smoke passes, a diagnostic training run may be started only
after explicit authorization. Every non-smoke run must include
`--full-run-authorized`:

```powershell
& $python scripts/train.py `
  --run-name wadi_l32_reprogramming_seed2026 `
  --full-run-authorized --device cuda
```

The first controlled L=32 ablation uses the same base config and only changes
the recorded command-line override:

```powershell
& $python scripts/train.py --run-name l32_linear_head `
  --projector linear --vocab-loss-weight 0 --full-run-authorized --device cuda
& $python scripts/train.py --run-name l32_direct_head `
  --projector direct --vocab-loss-weight 0 --full-run-authorized --device cuda
& $python scripts/train.py --run-name l32_reprogram_head `
  --projector reprogramming --vocab-loss-weight 0 --full-run-authorized --device cuda
& $python scripts/train.py --run-name l32_reprogram_dual `
  --projector reprogramming --vocab-loss-weight 0.1 --full-run-authorized --device cuda
& $python scripts/train.py --run-name l32_verbalizer_only `
  --projector reprogramming --classifier-loss-weight 0 `
  --vocab-loss-weight 1 --full-run-authorized --device cuda
& $python scripts/train.py --config configs/wadi_no_llm_baseline.json `
  --run-name l32_no_llm --full-run-authorized --device cuda
& $python scripts/train.py --config configs/wadi_no_llm_baseline.json `
  --baseline-hidden-dim 384 --run-name l32_no_llm_param_matched `
  --full-run-authorized --device cuda
```

The test-X dependency and DTT state-mapping bug have been removed from the
current code, but this does not authorize formal testing. Select and lock the
architecture and threshold protocol using training/development data only before
opening the formal test split.

For the fair window grid, all window lengths share endpoint candidates starting
at 127, so their sampled normal endpoints and development endpoints match for a
given seed:

```powershell
& $python scripts/train.py --run-name l8_direct `
  --window-length 8 --projector direct --vocab-loss-weight 0 `
  --full-run-authorized --device cuda
```

MOIRAI Base Encoder layers use 1-based numbering. For example, this selects the
10th of 12 Transformer Encoder layers while keeping the rest of the experiment
unchanged:

```powershell
& $python scripts/train.py --run-name l64_layer10_direct `
  --window-length 64 --moirai-layer 10 --projector direct `
  --vocab-loss-weight 0 --full-run-authorized --device cuda
```

Every prepared or trained run writes a human-readable `results.md` beside its
JSON files in `outputs/<run-name>/`. Backfill older outputs with:

```powershell
& $python scripts/generate_results_md.py
```

Multi-run research summaries live under `record/<experiment-name>/README.md`.
This subproject no longer writes new records to the project-root `record/`.

## Synthetic anomaly generation

`scripts/generate_synthetic_anomalies.py` creates an offline, reproducible Full
mixture of five anomaly operators from a normal training X array only: spike,
shift/ramp, flatline, soft patch replacement, and cross-channel dependency
break. Continuous and discrete channels are handled separately, and discrete
outputs are restricted to states observed in normal training data.

The WADI default generates 5,000 endpoint-aligned L=64 windows:

```powershell
& $python scripts/generate_synthetic_anomalies.py
```

For another dataset, pass its normal `[time, channels]` NumPy array and one
UTF-8 channel name per line:

```powershell
& $python scripts/generate_synthetic_anomalies.py `
  --input-x path/to/normal_train_x.npy `
  --sensor-names path/to/channel_names.txt `
  --output-dir path/to/synthetic_output `
  --window-length 64 --num-samples 5000 --seed 2026
```

The output package contains synthetic windows, point/channel masks, endpoint
labels, source endpoints, operator codes, train-derived statistics, hashes, and
a per-sample JSONL generation manifest. Generation never opens validation or
test arrays. These artifacts are engineering inputs; they do not establish that
synthetic anomalies improve detection performance.

Training ignores synthetic data by default. Enable the package matching the
resolved window length explicitly:

```powershell
& $python scripts/train.py `
  --run-name l32_with_synthetic --window-length 32 `
  --use-synthetic-anomalies --prepare-only
```

By default, the entry point deterministically selects as many synthetic windows
as real support-anomaly windows (170 in the current non-smoke protocol), rather
than allowing the 5,000-window package to dominate training. Override the count
or package path explicitly when needed:

```powershell
& $python scripts/train.py `
  --run-name l16_with_all_synthetic --window-length 16 `
  --use-synthetic-anomalies --synthetic-samples 5000 `
  --synthetic-data-dir synthetic_data/WADI-CLEAN_X_train_full_l16_seed2026 `
  --prepare-only
```

The default `fixed` sampling mode reuses one selected subset across epochs.
For the pre-registered strong candidates, use deterministic operator-stratified
rotation instead:

```powershell
& $python scripts/train.py `
  --config configs/wadi_s2_dtt_replacement_synth_verbalizer.json `
  --run-name s2_prepare `
  --use-synthetic-anomalies --synthetic-samples 170 `
  --synthetic-sampling epoch_stratified --prepare-only
```

`epoch_stratified` selects 170 non-overlapping windows per epoch while
preserving the operator proportions in the full package. A five-epoch run
therefore sees up to 850 unique synthetic windows without changing the
per-epoch class ratio. Every epoch's index hash and operator counts are saved
in `protocol.json`, and the exact schedule is saved as
`synthetic_epoch_indices.npy`.

Three explicit pure-verbalizer candidate configs are provided:

- `wadi_s1_numeric_synth_verbalizer.json`: all 80 active variables remain
  numeric MOIRAI tokens.
- `wadi_s2_dtt_replacement_synth_verbalizer.json`: 54 continuous numeric
  tokens plus 26 discrete endpoint-state texts.
- `wadi_s3_dtt_additive_synth_verbalizer.json`: all 80 numeric tokens plus
  endpoint-state text for the 26 discrete variables.

All three use L=64, MOIRAI Base layer 12, DirectProjector, classifier weight
zero, verbalizer weight one, and cloud-locked physical/evaluation batch four.

The pure-verbalizer configs above remain available for exact reproduction. The
current cloud launcher defaults to the next classifier-head development round:

- `e1`: all 80 numeric tokens, classifier CE, and 170 epoch-stratified
  synthetic windows per epoch.
- `e2`: all 80 numeric tokens plus additive compact DTT, classifier CE, and no
  synthetic anomalies.

Both reuse `configs/wadi_qwen3_06b.json` and express all method differences as
explicit `scripts/train.py` CLI overrides. Run both sequentially for seeds
2026/2027/2028 with:

```bash
bash scripts/run_strong_candidates_cloud.sh
```

The launcher calls the single `scripts/train.py` entry point. It writes
one log per run, prints a heartbeat every 60 seconds, stops on the first
failure, and refuses to reuse a non-empty output directory. Useful overrides:

```bash
SEEDS="2026" CANDIDATES="e1 e2" DRY_RUN=1 \
  bash scripts/run_strong_candidates_cloud.sh
RUN_TAG=rerun01 SEEDS="2027 2028" CANDIDATES="e1 e2" \
  bash scripts/run_strong_candidates_cloud.sh
```

The combined `e3` candidate (additive DTT plus synthetic anomalies plus
classifier CE) is intentionally excluded from the default run. Only after the
paired e1/e2 results have been reviewed should it be enabled explicitly:

```bash
ALLOW_COMBINED_CANDIDATE=1 CANDIDATES="e3" RUN_TAG=combo01 \
  bash scripts/run_strong_candidates_cloud.sh
```

Before concatenation, the loader verifies the normal-train file hash, active
channel order and types, window length, float32 shape, endpoint masks, labels,
source endpoints, finite values, and the package's train-only isolation flag.
The selected count, operator distribution, package path, and selection hash are
written to `protocol.json`.

## Projectors

- `linear`: LayerNorm -> Linear(`d_moirai`, 1024) -> RMSNorm.
- `direct`: LayerNorm -> Linear(768, 1536) -> GELU -> Dropout ->
  Linear(1536, 1024) -> RMSNorm for MOIRAI Base.
- `reprogramming`: direct residual plus cross-attention from MOIRAI tokens to a
  fixed set of Qwen text-prototype embeddings.
- `backbone=none`: frozen MOIRAI variable tokens plus a two-layer lightweight
  cross-variable Transformer and attention pooling; this is the required no-LLM
  baseline.

## Data isolation

The current `scripts/train.py` no longer imports or loads `WADI_TEST_X/Y` and
records every opened train/validation file explicitly. Existing pre-repair DTT
artifacts remain invalid for strict comparison because their protocol depended
on test X; they are not retroactively repaired by the code change.

Window labels always use the endpoint `Y[s + L - 1]`. The helper in
`mqids/data.py` splits the single validation attack into chronological support
and query portions with a guard of at least the largest configured window. This
reduces overlapping-window leakage, but does not turn one attack event into two
independent events.

Per-epoch development evaluation may use a configured stride greater than one
to control compute. It is diagnostic only; locked formal validation/test runs
must use endpoint stride 1 and report that setting explicitly.
