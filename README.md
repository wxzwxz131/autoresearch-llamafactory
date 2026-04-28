# autoresearch-llamafactory

将 autoresearch 的自主实验思路移植到 **LLaMA-Factory** 微调场景。AI agent 自动修改 LoRA / 优化器配置，运行实验，对比 train_loss，保留最优配置，循环往复。

## 原理对照

| autoresearch（预训练）| autoresearch-llamafactory（微调）|
|---|---|
| `train.py` | `examples/train_lora/qwen3_lora_sft.yaml` |
| `prepare.py`（固定）| `run_experiment.py`（固定）|
| `program.md` | `program.md` |
| 指标：`val_bpb`（越低越好）| 指标：`train_loss`（越低越好）|
| 固定 5 分钟时间预算 | 固定 `num_train_epochs=3.0` 训练预算 |
| `uv run train.py > run.log 2>&1` | `python run_experiment.py > run.log 2>&1` |

## 文件说明

```
autoresearch-llamafactory/
├── program.md                              ← 人类编辑：Agent 操作手册
├── agent.py                                ← 自主实验 agent
├── run_experiment.py                       ← 固定不改：启动训练、解析指标、打印摘要
├── examples/train_lora/
│   └── qwen3_lora_sft.yaml                 ← Agent 编辑：所有可调超参数
├── qwen3_lora_sft.sh                       ← 参考：等效的 shell 启动脚本
├── results.tsv                             ← Agent 维护：实验记录（首次运行后自动创建）
└── README.md                               ← 本文件
```
## 直接开始

直接调用 autoresearch.ipynb

别忘了配置agent模型配置（agent.py）
'''
def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("API_KEY", ""),
        base_url=os.getenv("BASE_URL", "https://api.lkeap.cloud.tencent.com/plan/v3"),
        model=os.getenv("MODEL_ID", "glm-5.1"),
        max_tokens=4096,
    )
'''

## 快速开始

### 1. 安装 LLaMA-Factory

```bash
pip install llamafactory
# 或从源码安装
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory && pip install -e ".[torch,metrics]"
```

### 2. 配置模型与数据集

当前已配置（`examples/train_lora/qwen3_lora_sft.yaml` 顶部的固定参数，只改一次）：

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
template: qwen3_nothink
dataset: identity,alpaca_en_demo
```

调用方式：
```bash
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

如需换成其他模型，同时修改这三个参数与 `output_dir`。LLaMA-Factory 支持的内置数据集见其 [data/README.md](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md)。

调用的agent模型配置（agent.py）
'''
def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("API_KEY", ""),
        base_url=os.getenv("BASE_URL", "https://api.lkeap.cloud.tencent.com/plan/v3"),
        model=os.getenv("MODEL_ID", "glm-5.1"),
        max_tokens=4096,
    )
'''

### 3. 验证环境

```bash
llamafactory-cli version
python -c "import yaml; print('yaml ok')"
```

### 4. 手动跑一次基线实验

```bash
python run_experiment.py > run.log 2>&1
grep "^train_loss:\|^status:" run.log
```

### 5. 启动自主 Agent（agent.py）

`agent.py` 是一个无需人工干预、在服务器上持续运行的自主实验 agent。

#### 安装依赖

```bash
pip install langchain-openai pyyaml
```

#### 配置 API Key（可选，已内置默认值）

```bash
# 默认使用腾讯 LKE API + GLM 模型，无需额外配置
# 如需覆盖，设置以下环境变量：
export API_KEY=sk-tp-...
export BASE_URL=https://api.lkeap.cloud.tencent.com/plan/v3
export MODEL_ID=glm-5.1
```

#### 运行

```bash
# 先跑一次基线，确认环境正常
python agent.py --baseline-only

# 无限循环自主实验
python agent.py

# 限制最多 20 次实验
python agent.py --max-iters 20
```

#### 服务器后台运行

```bash
# 方法一：nohup
nohup python agent.py > nohup.out 2>&1 &

# 方法二：tmux（推荐，可随时 attach 查看进度）
tmux new -s research
python agent.py
# Ctrl+B D 挂后台；tmux attach -t research 重新连入
```

#### agent.py 工作流

1. 自动创建 `autoresearch/<tag>` git 分支，先跑一次 baseline 建立基线
2. 调用 LLM → 解析建议的 `train_config.yaml` → 写入文件 → git commit
3. 运行 `run_experiment.py`，解析 `train_loss`
4. 改善 → **keep**；无改善或 VRAM 暴涨 → `git revert` + **discard**
5. 记录到 `results.tsv`，循环

#### 命令行参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--max-iters` | `0`（无限）| 最大实验轮数 |
| `--tag` | 今日日期（如 `apr27`）| git 分支后缀 |
| `--baseline-only` | false | 仅跑基线后退出 |

运行日志同时写入终端和 `agent.log`。

## Agent 可调参数速查

| 参数 | 当前默认值 | 建议范围 |
|------|--------|中-----------|
| `lora_rank` | 8 | 4, 8, 16, 32, 64 |
| `lora_target` | all | all, q_proj,v_proj |
| `finetuning_type` | lora | lora, freeze, full |
| `learning_rate` | 1e-4 | 5e-5 ~ 5e-4 |
| `lr_scheduler_type` | cosine | cosine, linear, constant |
| `warmup_ratio` | 0.1 | 0.0 ~ 0.2 |
| `per_device_train_batch_size` | 1 | 1, 2, 4, 8 |
| `gradient_accumulation_steps` | 8 | 2, 4, 8, 16 |
| `max_samples` | 1000 | 500 ~ 全量 |
| `cutoff_len` | 2048 | 512, 1024, 2048 |
| `bf16` | true | true, false |

## 固定参数（不可改）

| 参数 | 值 | 原因 |
|------|-----|------|
| `num_train_epochs` | 3.0 | 固定训练预算，保证实验可比 |
| `model_name_or_path` | Qwen/Qwen3-4B-Instruct-2507 | 定义任务 |
| `template` | qwen3_nothink | 与模型绑定 |
| `dataset` | identity,alpaca_en_demo | 定义任务 |
| `output_dir` | saves/qwen3-4b/lora/sft | run_experiment.py 需要 |
| yaml 路径 | examples/train_lora/qwen3_lora_sft.yaml | run_experiment.py 硬编码 |

## results.tsv 格式

```
commit	train_loss	memory_gb	status	description
a1b2c3d	1.234567	8.0	keep	baseline lora_rank=8
b2c3d4e	1.198200	8.1	keep	lora_rank=16 lr=2e-4
c3d4e5f	1.256000	8.0	discard	cosine→linear (worse)
d4e5f6g	0.000000	0.0	crash	full fine-tuning OOM
```

## 注意事项

- **OOM 处理**：显存不足时先减小 `per_device_train_batch_size`，再减小 `lora_rank`，最后考虑加大 `gradient_accumulation_steps` 来维持等效 batch size
- **数据集格式**：自定义数据集需在 LLaMA-Factory 的 `data/dataset_info.json` 中注册
- **多 GPU**：本框架设计为单 GPU 实验；多 GPU 需修改启动命令（torchrun / deepspeed）
