"""
autoresearch-llamafactory: run_experiment.py
Fixed runner script — DO NOT MODIFY.

Launches `llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml`, waits for completion,
parses metrics from trainer_log.jsonl, and prints a standardized summary.

Primary metric is train_loss (final EMA-smoothed value). If eval is enabled
in the config, eval_loss is also reported as a bonus metric.

Usage:
    python run_experiment.py > run.log 2>&1

Extract key metric after the run:
    grep "^train_loss:\\|^peak_vram_mb:\\|^status:" run.log
"""

import json
import os
import subprocess
import sys
import time

import yaml

CONFIG_PATH = "examples/train_lora/qwen3_lora_sft.yaml"


def load_output_dir() -> str:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out = config.get("output_dir", "saves/current")
    return out


def parse_trainer_log(output_dir: str) -> dict:
    """
    Read trainer_log.jsonl produced by LLaMA-Factory.
    Returns best eval_loss, final train_loss, and peak_vram_mb.
    """
    log_path = os.path.join(output_dir, "trainer_log.jsonl")
    if not os.path.exists(log_path):
        return {}

    best_eval_loss = None
    final_train_loss = None
    peak_vram_mb = 0.0

    with open(log_path, encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            # Eval metrics (LLaMA-Factory uses "eval_loss" key)
            if "eval_loss" in entry:
                loss = float(entry["eval_loss"])
                if best_eval_loss is None or loss < best_eval_loss:
                    best_eval_loss = loss

            # Train loss (logged as "loss")
            if "loss" in entry:
                final_train_loss = float(entry["loss"])

            # VRAM (logged as "cuda_memory_allocated" in GB by some versions)
            for vram_key in ("cuda_memory_allocated", "gpu_memory_mb"):
                if vram_key in entry:
                    val = float(entry[vram_key])
                    # Normalize to MB
                    if vram_key == "cuda_memory_allocated":
                        val = val * 1024  # GB → MB
                    peak_vram_mb = max(peak_vram_mb, val)

    return {
        "best_eval_loss": best_eval_loss,
        "final_train_loss": final_train_loss,
        "peak_vram_mb": peak_vram_mb,
    }


def get_peak_vram_mb_from_torch() -> float:
    """
    Fallback: query PyTorch directly for peak VRAM after training.
    Only works if called in the same process — not useful here, kept as doc.
    """
    return 0.0


def main() -> None:
    t0 = time.time()

    print(f"[run_experiment] Launching: llamafactory-cli train {CONFIG_PATH}", flush=True)
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    print(f"[run_experiment] Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(flush=True)

    result = subprocess.run(
        ["llamafactory-cli", "train", CONFIG_PATH],
        text=True,
    )

    t1 = time.time()
    wall_time = t1 - t0
    success = result.returncode == 0

    output_dir = load_output_dir()
    metrics = parse_trainer_log(output_dir)

    print()
    print("---")

    if not success:
        print("status:          CRASH")
        print("train_loss:      0.000000")
        print("eval_loss:       N/A")
        print(f"wall_time_s:     {wall_time:.1f}")
        print("peak_vram_mb:    0.0")
        sys.exit(1)

    best_eval = metrics.get("best_eval_loss")
    final_train = metrics.get("final_train_loss")
    peak_vram = metrics.get("peak_vram_mb", 0.0)

    # train_loss is the primary metric (eval may be disabled in config)
    if final_train is None:
        print("status:          NO_TRAIN_LOSS")
        print("train_loss:      0.000000")
        print("eval_loss:       N/A")
        print(f"wall_time_s:     {wall_time:.1f}")
        print("peak_vram_mb:    0.0")
        sys.exit(1)

    print(f"train_loss:      {final_train:.6f}")
    if best_eval is not None:
        print(f"eval_loss:       {best_eval:.6f}")
    else:
        print("eval_loss:       N/A")
    print(f"wall_time_s:     {wall_time:.1f}")
    print(f"peak_vram_mb:    {peak_vram:.1f}")
    print("status:          OK")


if __name__ == "__main__":
    main()
