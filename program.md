# autoresearch-llamafactory

This is an experiment to have the LLM autonomously optimize LLaMA-Factory fine-tuning configurations.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr24`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The folder is small. Read these files for full context:
   - `README.md` — repository context and setup instructions.
   - `autoresearch-llamafactory/run_experiment.py` — fixed runner: launches training, parses metrics, prints summary. Do not modify.
   - `examples/train_lora/qwen3_lora_sft.yaml` — the file you modify. All LoRA / optimizer / scheduling hyperparameters.
4. **Verify environment**: Confirm `llamafactory-cli` is available (`llamafactory-cli version`). If not, tell the human to install LLaMA-Factory first.
5. **Verify model & dataset**: Check that `model_name_or_path` and `dataset` in `examples/train_lora/qwen3_lora_sft.yaml` point to valid resources accessible from this machine. Confirm with the human before first run.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

---

## Experimentation

Each experiment runs for a **fixed epoch budget** (controlled by `num_train_epochs` in `train_config.yaml`). Do not change `num_train_epochs` — it is the "time budget" equivalent.

Run an experiment with:

```bash
# !python autoresearch-llamafactory/run_experiment.py > run.log 2>&1 # colab使用
python autoresearch-llamafactory/run_experiment.py > run.log 2>&1
```

**What you CAN do:**

Modify `examples/train_lora/qwen3_lora_sft.yaml` — this is the ONLY file you edit. Fair game:
<!-- - LoRA hyperparameters: `lora_rank`, `lora_target` -->
- Optimizer: `learning_rate`, `lr_scheduler_type`, `warmup_ratio`
- Batch shape: `per_device_train_batch_size`, `gradient_accumulation_steps`
- Data: `max_samples`, `cutoff_len`, `preprocessing_num_workers`, `dataloader_num_workers`
- Precision: `bf16`, `fp16`
- Enable eval: uncomment the `### eval` section (adds `val_size`, `eval_strategy`, `eval_steps`, `per_device_eval_batch_size`) to get `eval_loss` alongside `train_loss`

**What you CANNOT do:**
- Change `num_train_epochs` — this is the fixed training budget.
- Change `finetuning_type` — always `lora`. Do not switch to freeze or full fine-tuning.
- Change `model_name_or_path`, `dataset`, `template`, `trust_remote_code` — these are set by the human and define the task.
- Change `output_dir`, `overwrite_output_dir`, `report_to` — these are required by `autoresearch-llamafactory/run_experiment.py`.
- Modify `autoresearch-llamafactory/run_experiment.py`. It is the fixed evaluation harness.
- Rename or move `examples/train_lora/qwen3_lora_sft.yaml` — the path is hardcoded in `autoresearch-llamafactory/run_experiment.py`.
- Install new packages or add dependencies not already available.

**The goal is simple: get the lowest `train_loss`.** Since `num_train_epochs` is fixed, you don't need to worry about compute budget. If you enable eval, use `eval_loss` as the tie-breaker.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful train_loss gains, but don't cause OOM.

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement from 10 extra config lines? Not worth it. A 0.01 improvement from changing one value? Keep. Reverting to a simpler config that matches the best score? Definitely keep.

**The first run**: Your very first run should always be to establish the baseline — run with the config as-is.

---

## Output format

Once the script finishes it prints a summary like this:

```
---
train_loss:      0.987654
eval_loss:       N/A
wall_time_s:     483.2
peak_vram_mb:    8192.0
status:          OK
```

(`eval_loss` is `N/A` when the eval section is commented out in `train_config.yaml`.)

Extract the key metrics from the log file:

```bash
# !grep "^train_loss:\|^peak_vram_mb:\|^status:" run.log # colab使用
grep "^train_loss:\|^peak_vram_mb:\|^status:" run.log
```

---

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	train_loss	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. train_loss achieved (final value, e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 8.2 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	train_loss	memory_gb	status	description
a1b2c3d	1.234567	8.0	keep	baseline lora_rank=8
b2c3d4e	1.198200	8.1	keep	lora_rank=16 lr=2e-4
c3d4e5f	1.256000	8.0	discard	cosine→linear scheduler (worse)
d4e5f6g	0.000000	0.0	crash	finetuning_type=full (OOM)
```

---

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr24`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `examples/train_lora/qwen3_lora_sft.yaml` with an experimental idea — directly edit the YAML values.
3. `git add examples/train_lora/qwen3_lora_sft.yaml && git commit -m "<short description>"`
4. Run the experiment: `python autoresearch-llamafactory/run_experiment.py > run.log 2>&1`
5. Read out the results: `grep "^train_loss:\|^peak_vram_mb:\|^status:" run.log`
6. Decide: if `train_loss` improved (or simplified), keep; otherwise `git revert HEAD --no-edit` to restore the config.
7. Log to `results.tsv`.
8. Back to step 1.

### Decision heuristic

- **New best train_loss**: keep unconditionally (unless VRAM exploded).
- **Same train_loss, simpler config**: keep (simplification win).
- **Marginal improvement (< 0.005) with added complexity**: discard.
- **Crash / no train_loss in log**: always discard, restore config.
