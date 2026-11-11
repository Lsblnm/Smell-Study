# AI-Friendly Eval（FeatureEnvy / GodClass）

基于 SWE-bench Multilingual，通过注入 FeatureEnvy 或 GodClass smell patch，评测
Codex、OpenCode 等 Agent 在坏味场景下的修复能力。

## Codex 批量注入 smell

入口脚本接受 smell 提示词、实例列表和并行数：

```bash
python scripts/inject_smells.py prompts/LongMethod.md ids-lm.txt 5
python scripts/inject_smells.py prompts/GodClass.md apache__lucene-13301,google__gson-2158 5
python scripts/inject_smells.py prompts/FeatureEnvy.md ids-fe.txt 5
python scripts/inject_smells.py --ids prompts/ids/ids-fe.txt --patch-root smell-patch/S2 --intro-root smell-patch-intro/S2 prompts/S2/FeatureEnvy.md idfs-300.txt 10
```

实例文件每行写一个 `instance_id`，空行和以 `#` 开头的行会被忽略。每个实例从
`SWE-bench/SWE-bench_Multilingual` 读取 issue、gold patch 和 test patch 等元数据，
源码则从对应 SWE-bench 镜像的 `/testbed` 导出独立临时副本，由一个全新的 Codex
Agent 处理；参数 3 只限制当前脚本的并发数，因此可以在三个终端分别启动
LongMethod、GodClass 和 FeatureEnvy。临时副本和锁文件默认写在项目下 `.tmp/`
（`/data1` 大盘），不要用系统盘 `/tmp`。可用 `SMELL_INJECT_TMPDIR` 覆盖。

运行前需确保 Docker 可用、镜像已列入 `multilingual_300_images.txt`，并配置：

```bash
python -m pip install '.[smell-injection]'
```

注入统一走 Codex SDK；凭据复用评测同一套 `~/.codex/config.toml` 与项目 `.env`
里的 `IKUNCODE_API_KEY`：

```bash
# 可选；不设置时默认 gpt-5.6-sol / high / ikuncode（Codex 号池）
export CODEX_MODEL='gpt-5.6-sol'
export CODEX_MODEL_EFFORT='high'
python scripts/inject_smells.py prompts/FeatureEnvy.md ids-fe.txt 5

# Grok 号池（.env 的 IKUNCODE_GROK_API_KEY + config.toml 的 ikuncode_grok）
export CODEX_MODEL_PROVIDER='ikuncode_grok'
export CODEX_MODEL='grok-4.6'
export CODEX_MODEL_EFFORT='high'
python scripts/inject_smells.py prompts/FeatureEnvy.md ids-fe.txt 5
```

Codex 注入会关掉 ChatGPT Apps / 插件 / 远程模型目录侧信道，避免卡在
`chatgpt.com`；实际请求走 `CODEX_MODEL_PROVIDER` 指定的 ikuncode。三个 smell
同时跑时，单窗口建议 8，合计不超过 12。

脚本仅以
`smell-patch-intro/{Smell}/{instance_id}-{缩写}.json` 是否存在判断任务是否完成：
存在即跳过；正常失败和不可注入也会生成说明文件；进程中断不会生成说明文件，
下次运行会自动重试。可用 `--ids prompts/ids/ids-fe.txt` 把范围收成该名单与
位置参数实例列表的交集；跳过条件仍只看这份 JSON，不看 patch。

每个实例使用显式 Agent 模式。若 Codex 首次只返回“准备分析”等中间回复，
调度器会在同一个 Agent 上继续执行，最多 3 轮；仍未返回合法 JSON `"Status"`
时按基础设施错误处理，不生成完成标记，以便下次重试。

数据集元数据会缓存到 `.cache/swebench_multilingual_test.json`，后续终端直接复用该
项目缓存，不再重复访问 Hugging Face。若确实要刷新数据集缓存，可临时设置：

```bash
SMELL_INJECT_DATASET_ONLINE=1 python scripts/inject_smells.py prompts/FeatureEnvy.md idfs.txt 1
```

## 评测原理

本地持久存储 `smell.patch`，测试前注入到容器内 `/testbed`：

```
官方 SWE-bench image
        |
        | 创建全新容器
        v
/testbed = base_commit 原始仓库
        |
        | 注入 A_smell.patch
        v
/testbed = base_commit + A 坏味变更
        |
        | 启动 Agent 或执行验证
        v
Agent 生成修复 patch
        |
        | 执行评测
        v
     评测结果
```

整体流程：本地构建镜像用于注入 → 注入 → 远程拉 bench 镜像生成评测轨迹 → 执行评测。

---

## 1. 准备镜像

本地构建（不是直接 pull 现成 instance 镜像）：

```bash
python -m swebench.harness.prepare_images \
  --dataset_name SWE-bench/SWE-bench_Multilingual \
  --split test \
  --instance_ids \
    apache__lucene-11760 \
    apache__lucene-12022 \
  --max_workers 2 \
  --tag latest \
  --env_image_tag latest
```

执行评测前可提前拉远程镜像（并行一般不超过 3）：

```bash
SMELL_DIR=$PWD/smell-patch/FeatureEnvy
ls "$SMELL_DIR"/*-FE.patch | sed 's|.*/||;s|-FE\.patch$||' | \
while read iid; do
  img=$(echo "docker.io/swebench/sweb.eval.x86_64.${iid//__/_1776_}:latest" | tr '[:upper:]' '[:lower:]')
  echo "$img"
done | xargs -P 2 -I{} bash -c 'echo "=== pull {} ==="; docker pull "{}"'
```

---

## 2. Codex / OpenCode（生成轨迹与 preds）

先安装并登录所需 CLI，确认以下命令能够正常输出版本：

```bash
codex --version
opencode --version
```

### 凭据配置

两个 CLI 都从同一个环境变量读取中转站的 API key（`~/.codex/config.toml` 的
`env_key`，`~/.config/opencode/opencode.json` 的 `{env:...}` 占位符）。为避免依赖
具体 shell 是否 `export` 过，runner 启动时会自动读取项目根目录的 `.env`：

```bash
# .env（已被 .gitignore 忽略，建议 chmod 600）
IKUNCODE_API_KEY=sk-...
```

已经导出的同名环境变量优先级更高，`.env` 只做兜底。

runner 在跑第一个实例之前会做一次自检：确认 CLI 配置可读、key 已就位、中转站
`/v1/models` 可达且包含目标模型，任一不满足立即退出并说明原因，不会空跑整个名单。
用 `--skip-preflight` 可以跳过。

自检失败时的常见原因：`HOME` 指向了错误的账户（例如 `sudo -i` 会把 `HOME` 变成
`/root`，而配置在 `/data1/<user>`）、`.env` 缺失、或模型名拼错。

runner 始终以 `--instances` 指定评测集合。传入 `--smell` 时会注入 smell 并将其
提交为实验基线；不传时直接以原始 base commit 为基线。Agent 在宿主机临时工作区
中编辑代码，需要依赖时通过对应 SWE-bench 容器运行命令。最终生成的
`model_patch` 只包含 Agent 相对于所选基线的修改。

实例名单由成功生成的基础 smell patch 决定：

- `ids-fe.txt`：`smell-patch/FeatureEnvy/*-FE.patch` 对应的 78 个实例
- `ids-gc.txt`：`smell-patch/GodClass/*-GC.patch` 对应的 62 个实例

同一种 smell 的 smell/no-smell 对照必须使用同一份名单。

Codex × FeatureEnvy：

```bash
python scripts/run_cli_swebench.py \
  --agent codex \
  --model '<codex-model>' \
  --instances "$PWD/ids-fe.txt" \
  --smell "$PWD/smell-patch/FeatureEnvy" \
  --output-dir "$PWD/output/codex/FeatureEnvy-smell" \
  --workers 4 \
  --timeout 2700 \
  --retries 2
```

Codex × GodClass：

```bash
python scripts/run_cli_swebench.py \
  --agent codex \
  --model '<codex-model>' \
  --instances "$PWD/ids-gc.txt" \
  --smell "$PWD/smell-patch/GodClass" \
  --output-dir "$PWD/output/codex/GodClass-smell" \
  --workers 4 \
  --timeout 2700 \
  --retries 2
```

OpenCode 的运行方式相同，模型名使用 `provider/model`：

```bash
python scripts/run_cli_swebench.py \
  --agent opencode \
  --model '<provider/model>' \
  --instances "$PWD/ids-fe.txt" \
  --smell "$PWD/smell-patch/FeatureEnvy" \
  --output-dir "$PWD/output/opencode/FeatureEnvy-smell" \
  --workers 4 \
  --timeout 2700 \
  --retries 2

python scripts/run_cli_swebench.py \
  --agent opencode \
  --model '<provider/model>' \
  --instances "$PWD/ids-gc.txt" \
  --smell "$PWD/smell-patch/GodClass" \
  --output-dir "$PWD/output/opencode/GodClass-smell" \
  --workers 4 \
  --timeout 2700 \
  --retries 2
```

无 smell 对照使用同一份 `--instances`，不传 `--smell`。例如 OpenCode ×
FeatureEnvy：

```bash
python scripts/run_cli_swebench.py \
  --agent opencode \
  --model '<provider/model>' \
  --instances "$PWD/ids-fe.txt" \
  --output-dir "$PWD/output/opencode/FeatureEnvy-no-smell" \
  --workers 4 \
  --timeout 2700 \
  --retries 2
```

Agent 工作区在开跑前会打上评测用的 `test.patch`（无 smell 用数据集原版；有 smell
时若存在 `{id}-*-test.patch` 则用重构版，否则仍用原版）。预测相对该基线导出，
因此 test 改动不会进入 `preds.json`。不要打 gold。

runner 默认跳过对应输出目录 `preds.json` 中 `model_patch` 非空的实例，缺失条目和空补丁会继续跑；`--redo` 则忽略 `preds.json`、按 `--instances` 全量重跑。每个实例保存：

- `prompt.txt`：实际发给 CLI 的提示词
- `trajectory.jsonl`：CLI 的 JSON 事件流，边跑边落盘，超时被杀也保留
- `stderr.log`：CLI 标准错误
- `patch.diff`：相对于（smell +）test 基线的预测补丁
- `patch.partial.diff`：仅失败实例产生，用于排查，不会进入 `preds.json`
- `result.json`：状态、耗时和补丁大小

汇总文件为输出目录下的 `preds.json` 和 `summary.json`，两者都跨多次运行累积。
重跑某个实例时其目录内的旧产物会先被清空，因此不会出现新旧结果混在一起。

OpenCode 的模型名必须带 provider 前缀（`ikuncode/gpt-5.6-terra`）；Codex 的 provider
已经写在 `~/.codex/config.toml` 的 `model_provider` 里，命令行传裸模型名即可
（`gpt-5.6-terra`）。

`--timeout` 建议不低于 2700 秒：成功实例的耗时 p90 约 1400 秒，1800 秒会把尾部
的正常长任务误判为超时。`--workers` 过高会加剧中转 API 的限流与超时，建议 4 以内。
`--retries` 用于整实例重试，可以吸收几分钟量级的 API 抖动，每次重试前分别等待
60 秒、120 秒。

---

## 3. SWE-bench 评测

### 单个案例（gold）

```bash
python -m swebench.harness.run_evaluation \
  -d SWE-bench/SWE-bench_Multilingual \
  -s test \
  -i apache__druid-13704 \
  -p gold \
  -id test_gold
```

### 多个案例

```bash
python -m swebench.harness.run_evaluation \
  -d SWE-bench/SWE-bench_Multilingual \
  -s test \
  -p $PWD/output/codex/FeatureEnvy-smell/preds.json \
  -i \
    apache__druid-13704 \
    apache__druid-14092 \
    apache__druid-14136 \
    apache__druid-15402 \
    apache__druid-16875 \
  --max_workers 2 \
  -id FE-5-smell \
  --smell $PWD/smell-patch/FeatureEnvy
```

### 按 FE 目录批量

```bash
SMELL_DIR=$PWD/smell-patch/FeatureEnvy
```

有 smell：

```bash
python -m swebench.harness.run_evaluation \
  -d SWE-bench/SWE-bench_Multilingual \
  -s test \
  -p $PWD/output/codex/FeatureEnvy-smell/preds.json \
  --max_workers 2 \
  -id codex-FE-smell \
  --smell "$SMELL_DIR"
```

评测器只会把 `*-FE.patch` / `*-GC.patch` 识别为 smell，不会误匹配旁边的
`-gold.patch` 和 `-test.patch`。如果存在 `-test.patch`，会在构建测试脚本前自动
替换数据集原始 test patch；不存在时继续使用原始 test patch。

无 smell 对照评测不传 `--smell`：

```bash
python -m swebench.harness.run_evaluation \
  -d SWE-bench/SWE-bench_Multilingual \
  -s test \
  -p $PWD/output/codex/FeatureEnvy-no-smell/preds.json \
  --max_workers 2 \
  -id codex-FE-no-smell
```

GodClass 或 OpenCode 只需替换目录、preds 路径和 run ID。例如：

```bash
python -m swebench.harness.run_evaluation \
  -d SWE-bench/SWE-bench_Multilingual \
  -s test \
  -p $PWD/output/opencode/GodClass-smell/preds.json \
  --max_workers 2 \
  -id opencode-GC-smell \
  --smell "$PWD/smell-patch/GodClass"
```
