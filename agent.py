"""
autoresearch-llamafactory: agent.py
Autonomous research agent. Runs on a server, loops forever.

The agent reads program.md as its operating manual, then proposes changes
to examples/train_lora/qwen3_lora_sft.yaml, runs experiments, and logs results.

Usage:
    python agent.py                  # unlimited iterations
    python agent.py --max-iters 20   # stop after 20 experiments

Environment variables (all have defaults, override as needed):
    API_KEY    — LLM API key  (default: built-in key)
    BASE_URL   — API base URL (default: https://api.lkeap.cloud.tencent.com/plan/v3)
    MODEL_ID   — model name   (default: glm-5.1)
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all absolute, relative to this script's directory)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()  # always autoresearch-llamafactory/

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(BASE_DIR / "agent.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROGRAM_MD  = BASE_DIR / "program.md"
# CONFIG_PATH is relative to the LlamaFactory root (cwd when invoking agent),
# so llamafactory-cli and git can resolve it from the repo root.
CONFIG_PATH = "examples/train_lora/qwen3_lora_sft.yaml"
RESULTS_TSV = BASE_DIR / "results.tsv"
RUN_LOG     = BASE_DIR / "run.log"
RESULTS_HEADER = "exp_id\ttrain_loss\tmemory_gb\tstatus\tdescription\n"

# Fixed params the agent must never touch — enforced by post-edit diff check
FIXED_KEYS = {
    "model_name_or_path",
    "dataset",
    "template",
    "trust_remote_code",
    "num_train_epochs",
    "output_dir",
    "overwrite_output_dir",
    "report_to",
    "finetuning_type",
    "stage",
    "do_train",
}

# ---------------------------------------------------------------------------
# LLM client (langchain ChatOpenAI, OpenAI-compatible)
# ---------------------------------------------------------------------------

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("API_KEY", "sk-tp-OKlF6MD0w7UEEArAqAG1Ba4zhs7AqwxBYBrVCo8eaA1CbfWx"),
        base_url=os.getenv("BASE_URL", "https://api.lkeap.cloud.tencent.com/plan/v3"),
        model=os.getenv("MODEL_ID", "glm-5.1"),
        max_tokens=4096,
    )


def llm_chat(llm: ChatOpenAI, system: str, user: str) -> str:
    log.info(f"Calling {llm.model_name} @ {llm.openai_api_base}...")
    log.info(f"  system prompt: {len(system)} chars | user message: {len(user)} chars")
    t0 = time.time()
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    resp = llm.invoke(messages)
    elapsed = time.time() - t0
    log.info(f"  LLM responded in {elapsed:.1f}s | reply: {len(resp.content)} chars")
    return resp.content


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def read_config() -> str:
    return Path(CONFIG_PATH).read_text(encoding="utf-8")


def write_config(content: str) -> None:
    Path(CONFIG_PATH).write_text(content, encoding="utf-8")


def extract_fixed_values(config_text: str) -> dict:
    """Extract current values of fixed keys for violation checking."""
    values = {}
    for line in config_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            if key in FIXED_KEYS:
                values[key] = val.strip()
    return values


def check_fixed_keys_unchanged(original: str, proposed: str) -> list[str]:
    """Return list of fixed keys that were changed."""
    orig_vals = extract_fixed_values(original)
    new_vals = extract_fixed_values(proposed)
    violations = []
    for key in FIXED_KEYS:
        o = orig_vals.get(key, "")
        n = new_vals.get(key, "")
        if o and n and o != n:
            violations.append(f"{key}: '{o}' → '{n}'")
    return violations


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment() -> dict:
    """Run run_experiment.py, streaming output to terminal and log file simultaneously."""
    log.info(f"Running experiment (log \u2192 {RUN_LOG})...")
    t0 = time.time()

    with open(RUN_LOG, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [sys.executable, str(BASE_DIR / "run_experiment.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        proc.wait()

    elapsed = time.time() - t0
    returncode = proc.returncode
    log.info(f"  Training process finished in {elapsed:.0f}s (exit code {returncode})")

    log_text = RUN_LOG.read_text(encoding="utf-8")

    metrics = {
        "train_loss": None,
        "eval_loss": None,
        "peak_vram_mb": 0.0,
        "wall_time_s": 0.0,
        "status": "CRASH" if returncode != 0 else "UNKNOWN",
    }

    for line in log_text.splitlines():
        line = line.strip()
        m = re.match(r"^(train_loss|eval_loss|peak_vram_mb|wall_time_s|status):\s+(.+)$", line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            if key in ("train_loss", "eval_loss", "peak_vram_mb", "wall_time_s"):
                try:
                    metrics[key] = float(val) if val != "N/A" else None
                except ValueError:
                    pass
            else:
                metrics[key] = val

    log.info(
        f"  Parsed metrics: train_loss={metrics['train_loss']}  "
        f"eval_loss={metrics['eval_loss']}  "
        f"peak_vram_mb={metrics['peak_vram_mb']}  "
        f"wall_time_s={metrics['wall_time_s']}  "
        f"status={metrics['status']}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Results TSV
# ---------------------------------------------------------------------------

def init_results_tsv() -> None:
    if not Path(RESULTS_TSV).exists():
        Path(RESULTS_TSV).write_text(RESULTS_HEADER, encoding="utf-8")
        log.info(f"Created {RESULTS_TSV}")


def append_result(commit: str, metrics: dict, status: str, description: str) -> None:
    train_loss = metrics.get("train_loss") or 0.0
    peak_vram_mb = metrics.get("peak_vram_mb") or 0.0
    memory_gb = round(peak_vram_mb / 1024, 1) if status != "crash" else 0.0
    loss_str = f"{train_loss:.6f}" if train_loss else "0.000000"
    line = f"{commit}\t{loss_str}\t{memory_gb}\t{status}\t{description}\n"
    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        f.write(line)
    log.info(f"Logged to {RESULTS_TSV}: {line.strip()}")


def read_recent_results(n: int = 10) -> str:
    if not Path(RESULTS_TSV).exists():
        return "(no results yet)"
    lines = Path(RESULTS_TSV).read_text(encoding="utf-8").splitlines()
    recent = lines[:1] + lines[-(n):]  # header + last N
    return "\n".join(recent)


def best_train_loss() -> float | None:
    if not Path(RESULTS_TSV).exists():
        return None
    best = None
    for line in Path(RESULTS_TSV).read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 4 and parts[3] == "keep":
            try:
                val = float(parts[1])
                if val > 0 and (best is None or val < best):
                    best = val
            except ValueError:
                pass
    return best


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    program = Path(PROGRAM_MD).read_text(encoding="utf-8")
    return f"""You are an autonomous ML research agent. Your operating manual is below.

{program}

---

RESPONSE FORMAT (strictly follow this):

<reasoning>
Your analysis of recent results and what to try next.
</reasoning>

<description>
One short sentence describing this experiment (no commas — this goes in a TSV file).
</description>

<config>
[full contents of the new train_config.yaml — preserve ALL comments and structure]
</config>

IMPORTANT:
- Output the COMPLETE train_config.yaml inside <config> tags, not just the changed lines.
- Never change: {', '.join(sorted(FIXED_KEYS))}
- Keep all ### section headers and inline comments.
"""


def build_user_message(iteration: int, metrics_last: dict | None) -> str:
    current_config = read_config()
    recent = read_recent_results(10)
    best = best_train_loss()

    last_run_summary = ""
    if metrics_last:
        last_run_summary = f"""
## Last experiment result
```
train_loss:   {metrics_last.get('train_loss', 'N/A')}
eval_loss:    {metrics_last.get('eval_loss', 'N/A')}
peak_vram_mb: {metrics_last.get('peak_vram_mb', 'N/A')}
wall_time_s:  {metrics_last.get('wall_time_s', 'N/A')}
status:       {metrics_last.get('status', 'N/A')}
```
"""

    return f"""## Experiment iteration {iteration}
Best train_loss so far: {best}

## Recent results (results.tsv)
```
{recent}
```
{last_run_summary}
## Current train_config.yaml
```yaml
{current_config}
```

Propose the next experiment. Output reasoning, a short description, and the full new qwen3_lora_sft.yaml.
"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(response: str) -> tuple[str, str]:
    """Extract (description, new_config) from LLM response."""
    desc_match = re.search(r"<description>\s*(.*?)\s*</description>", response, re.DOTALL)
    config_match = re.search(r"<config>\s*(.*?)\s*</config>", response, re.DOTALL)

    description = desc_match.group(1).strip() if desc_match else "no description"
    new_config = config_match.group(1).strip() if config_match else ""

    # Strip leading ```yaml / ``` fences if LLM wrapped the config
    new_config = re.sub(r"^```(?:yaml)?\n?", "", new_config)
    new_config = re.sub(r"\n?```$", "", new_config)

    return description, new_config


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="autoresearch-llamafactory autonomous agent")
    parser.add_argument("--max-iters", type=int, default=0, help="Max iterations (0 = unlimited)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run a single baseline experiment then exit")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline run (use when resuming a session where baseline is already in results.tsv)")
    args = parser.parse_args()

    print("=" * 60)
    print("  autoresearch-llamafactory agent")
    print(f"  max_iters={args.max_iters or 'unlimited'}  baseline_only={args.baseline_only}  skip_baseline={args.skip_baseline}")
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"  CONFIG:   {CONFIG_PATH}")
    print(f"  LLM:      {os.getenv('MODEL_ID', 'glm-5.1')} @ {os.getenv('BASE_URL', 'https://api.lkeap.cloud.tencent.com/plan/v3')}")
    print("=" * 60)

    # Preflight checks
    log.info("Preflight: checking required files...")
    Path(os.path.dirname(CONFIG_PATH)).mkdir(parents=True, exist_ok=True)
    for f in [PROGRAM_MD, Path(CONFIG_PATH), BASE_DIR / "run_experiment.py"]:
        if not Path(f).exists():
            log.error(f"Required file not found: {f}")
            sys.exit(1)
        log.info(f"  OK: {f}")

    llm = build_llm()
    log.info(f"LLM client ready: {llm.model_name}")
    init_results_tsv()

    original_config = read_config()
    original_fixed = extract_fixed_values(original_config)
    system_prompt = build_system_prompt()

    iteration = 0
    metrics_last: dict | None = None

    # --- Baseline run (iteration 0) ---
    if args.skip_baseline:
        log.info("--skip-baseline: reading baseline from results.tsv...")
        baseline_loss = best_train_loss()
        if baseline_loss is None:
            log.error("--skip-baseline set but no 'keep' rows found in results.tsv. Run without --skip-baseline first.")
            sys.exit(1)
        log.info(f"Resuming with best train_loss from results.tsv: {baseline_loss:.6f}")
        metrics_last = {"train_loss": baseline_loss, "eval_loss": None, "peak_vram_mb": 0.0, "wall_time_s": 0.0, "status": "OK"}
        iteration = 1
    else:
        log.info("--- Iteration 0: baseline ---")
        metrics = run_experiment()
        exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_loss = metrics.get("train_loss")

        if metrics["status"] == "CRASH" or train_loss is None:
            log.error("Baseline crashed. Fix run_experiment.py / qwen3_lora_sft.yaml first.")
            append_result(exp_id, metrics, "crash", "baseline")
            sys.exit(1)

        append_result(exp_id, metrics, "keep", "baseline")
        log.info(f"Baseline train_loss: {train_loss:.6f}")
        metrics_last = metrics
        iteration = 1

    if args.baseline_only:
        log.info("--baseline-only: done.")
        return

    # --- Experiment loop ---
    while True:
        if args.max_iters > 0 and iteration > args.max_iters:
            log.info(f"Reached max_iters={args.max_iters}. Stopping.")
            break

        log.info(f"--- Iteration {iteration} ---")

        # 1. Ask LLM for next experiment
        user_msg = build_user_message(iteration, metrics_last)
        try:
            response = llm_chat(llm, system_prompt, user_msg)
        except Exception as e:
            log.error(f"LLM call failed: {e}. Retrying in 30s...")
            time.sleep(30)
            continue

        log.info("LLM response received.")
        # Print the full LLM response so it's visible in the terminal
        print("\n--- LLM response ---")
        print(response)
        print("--- end of response ---\n")

        # 2. Parse response
        description, new_config = parse_response(response)
        log.info(f"Proposed: {description}")

        if not new_config:
            log.warning("Could not parse <config> from LLM response. Skipping iteration.")
            iteration += 1
            continue

        # 3. Validate fixed keys unchanged
        violations = check_fixed_keys_unchanged(original_config, new_config)
        if violations:
            log.warning(f"LLM tried to change fixed keys: {violations}. Skipping.")
            iteration += 1
            continue
        log.info("  Fixed key validation passed.")

        # 4. Write new config
        prev_config = read_config()
        write_config(new_config)
        exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log.info(f"  Config written to {CONFIG_PATH} (exp_id={exp_id})")

        # 5. Run experiment
        metrics = run_experiment()
        train_loss_new = metrics.get("train_loss")
        peak_vram = metrics.get("peak_vram_mb", 0.0)

        # 6. Decide keep / discard / crash
        if metrics["status"] == "CRASH" or train_loss_new is None:
            log.info("Experiment crashed — restoring previous config.")
            write_config(original_config)
            append_result(exp_id, metrics, "crash", description)
            metrics_last = metrics

        else:
            current_best = best_train_loss()
            improved = current_best is None or train_loss_new < current_best

            # Penalize dramatic VRAM increase (>50% over baseline)
            baseline_vram = 0.0
            if Path(RESULTS_TSV).exists():
                first_lines = Path(RESULTS_TSV).read_text().splitlines()
                if len(first_lines) > 1:
                    try:
                        baseline_vram = float(first_lines[1].split("\t")[2]) * 1024  # GB → MB
                    except (IndexError, ValueError):
                        pass
            vram_explosion = (baseline_vram > 0 and peak_vram > baseline_vram * 1.5)

            if improved and not vram_explosion:
                log.info(f"Improved: {current_best} → {train_loss_new:.6f}. Keeping.")
                append_result(exp_id, metrics, "keep", description)
                original_config = new_config  # update baseline for fixed-key checks
            else:
                reason = "vram exploded" if vram_explosion else "no improvement"
                log.info(f"Discarding ({reason}): {train_loss_new:.6f} vs best {current_best}.")
                write_config(prev_config)  # restore config file to pre-experiment state
                append_result(exp_id, metrics, "discard", description)

            metrics_last = metrics

        iteration += 1
        best = best_train_loss()
        print("\n" + "=" * 60)
        print(f"  Iteration {iteration - 1} done | Best train_loss so far: {best}")
        print("=" * 60 + "\n")
        log.info(f"Best train_loss so far: {best}")


if __name__ == "__main__":
    main()
