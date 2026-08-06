#!/usr/bin/env python3
"""Run anatomical head Experiments B, C, and D using an idle-GPU pool.

The launcher polls physical GPUs 0-3 and keeps at most two training processes
active. A GPU is eligible only when it has no compute process and its used
memory is below the configured idle threshold. Run with ``--dry-run`` first to
inspect the commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# EDITABLE EXPERIMENT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "training" / "train_fa_dinov2_zone_attention.py"
OUTPUT_ROOT = PROJECT_ROOT / "fa_anatomical_head_experiments"
CSV_ROOT = PROJECT_ROOT / "fold_zone_masks_ready_patient_split"
DATASET_ROOT = "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical"
MASK_ROOT = "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical_fa_zone_masks"
GPU_IDS = [0, 1, 2, 3]
MAX_CONCURRENT_JOBS = 2
GPU_POLL_SECONDS = 10
# Allows a small amount of driver/display allocation while still treating a
# GPU as idle. Set to 0 for a strict memory-empty check.
GPU_IDLE_MEMORY_LIMIT_MB = 512
EXTRA_ENV: dict[str, str] = {}

# Experiment A is the existing shared-head model and is intentionally omitted.
HEAD_EXPERIMENTS = [
    {
        "label": "B",
        "name": "group_outputs",
        "head_variant": "group_outputs",
        "description": "Shared projection with one output layer per anatomical group",
    },
    {
        "label": "C",
        "name": "group_adapters",
        "head_variant": "group_adapters",
        "description": "Shared projection with residual group adapters and group outputs",
    },
    {
        "label": "D",
        "name": "group_mlps",
        "head_variant": "group_mlps",
        "description": "One complete MLP per anatomical group",
    },
]
BACKBONE_CONFIGS = {
    "dinov2": {
        "image_size": 392,
        "extra_args": ["--dinov2_arch", "dinov2_vitb14"],
        "gradient_checkpointing": False,
        "description": "General DINOv2 ViT-B/14 baseline",
    },
    "retfound_dinov2": {
        "image_size": 392,
        "extra_args": ["--dinov2_arch", "dinov2_vitl14"],
        "gradient_checkpointing": True,
        "description": "RETFound-DINOv2 ViT-L/14 specialist encoder",
    },
    "dinov3_vitl16": {
        "image_size": 384,
        "extra_args": [],
        "gradient_checkpointing": True,
        "description": "DINOv3 ViT-L/16 LVD-1689M generalist encoder",
    },
}

# Fixed geometry + CLAHE, low-LR configuration used by the established best run.
COMMON_ARGS = [
    "--dataset_path", DATASET_ROOT,
    "--mask_dataset_path", MASK_ROOT,
    "--gpu", "0",  # Each process sees one physical GPU as local cuda:0.
    "--epochs", "100",
    "--lr", "1e-5",
    "--head_lr", "1e-4",
    "--min_lr", "1e-7",
    "--weight_decay", "1e-4",
    "--warmup_epochs", "5",
    "--dropout", "0.2",
    "--patience", "20",
    "--seed", "0",
    "--amp",
    "--rotation",
    "--rotation_prob", "0.7",
    "--rotation_degrees", "10",
    "--translation",
    "--translation_prob", "0.5",
    "--translation_fraction", "0.05",
    "--scale",
    "--scale_prob", "0.5",
    "--scale_min", "0.9",
    "--scale_max", "1.1",
    "--clahe",
    "--clahe_prob", "0.3",
    "--clahe_clip_limit", "2",
    "--clahe_grid_size", "8",
]

SUCCESS_MARKER = "_SUCCESS.json"
SUMMARY_FILE = "experiment_summary.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them or creating files.")
    parser.add_argument("--resume", action="store_true", help="Skip folds with a valid success marker.")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging for every run.")
    parser.add_argument("--wandb-project", default="uveitis-fa-zone-attention")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=[experiment["label"] for experiment in HEAD_EXPERIMENTS],
        default=[experiment["label"] for experiment in HEAD_EXPERIMENTS],
        help="Anatomical heads to run. Use '--experiments D' for the separate group-MLP head.",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=sorted(BACKBONE_CONFIGS),
        default=["dinov2"],
        help="One or more encoders to compare for every selected head and fold.",
    )
    parser.add_argument("--retfound-dinov2-checkpoint", default="RETFound_dinov2_meh")
    parser.add_argument("--dinov3-model-id", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-GPU batch size. Keep 2 for direct comparison with Experiment A; benchmark 8, 16, or 32 for speed.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per training process.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Disable deterministic CUDA enforcement for potentially faster training.",
    )
    return parser.parse_args()


def query_free_gpus() -> list[int]:
    """Return configured GPUs with no compute process and little used memory."""
    try:
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        process_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not poll GPUs with nvidia-smi: {exc}") from exc

    busy_uuids = set()
    for line in process_query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0] and fields[0] != "[N/A]":
            busy_uuids.add(fields[0])

    free_gpus = []
    for line in gpu_query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            gpu_id = int(fields[0])
            memory_used_mb = int(fields[2])
        except ValueError:
            continue
        gpu_uuid = fields[1]
        if (
            gpu_id in GPU_IDS
            and gpu_uuid not in busy_uuids
            and memory_used_mb <= GPU_IDLE_MEMORY_LIMIT_MB
        ):
            free_gpus.append(gpu_id)
    return sorted(free_gpus)


def build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = []
    index = 0
    selected_experiments = [experiment for experiment in HEAD_EXPERIMENTS if experiment["label"] in args.experiments]
    preserve_legacy_paths = args.backbones == ["dinov2"] and args.experiments == ["B", "C", "D"]
    for backbone_name in args.backbones:
        backbone_config = BACKBONE_CONFIGS[backbone_name]
        for experiment in selected_experiments:
            for fold in range(5):
                experiment_root = args.output_root.expanduser().resolve()
                if not preserve_legacy_paths:
                    experiment_root = experiment_root / f"backbone_{backbone_name}"
                output_dir = experiment_root / f"experiment_{experiment['label']}_{experiment['name']}" / f"fold_{fold}"
                command = [
                    args.python,
                    str(TRAIN_SCRIPT),
                    "--csvpath", str(CSV_ROOT / f"fold_{fold}"),
                    "--output_path", str(output_dir),
                    "--head_variant", experiment["head_variant"],
                    "--backbone", backbone_name,
                    "--image_size", str(backbone_config["image_size"]),
                    "--batch_size", str(args.batch_size),
                    "--num_workers", str(args.num_workers),
                    *backbone_config["extra_args"],
                    *COMMON_ARGS,
                ]
                if backbone_name == "retfound_dinov2":
                    command.extend(["--retfound_dinov2_checkpoint", args.retfound_dinov2_checkpoint])
                if backbone_name == "dinov3_vitl16":
                    command.extend(["--dinov3_model_id", args.dinov3_model_id])
                if args.hf_local_files_only:
                    command.append("--hf_local_files_only")
                if backbone_config["gradient_checkpointing"]:
                    command.append("--gradient_checkpointing")
                if not args.fast:
                    command.append("--deterministic")
                if args.wandb:
                    command.extend(
                        [
                            "--wandb",
                            "--wandb_project", args.wandb_project,
                            "--wandb_name", f"{backbone_name}_head_{experiment['label']}_{experiment['name']}_fold_{fold}",
                            "--wandb_group", "fa-anatomical-head-backbone-comparison",
                            "--wandb_tags", (
                                f"head-ablation,experiment-{experiment['label']},backbone-{backbone_name},"
                                "geometry-clahe,lr-1e-5"
                            ),
                        ]
                    )
                tasks.append(
                    {
                        "index": index,
                        "experiment": experiment["label"],
                        "name": f"{backbone_name}_{experiment['name']}",
                        "description": f"{backbone_config['description']}; {experiment['description']}",
                        "fold": fold,
                        "output_dir": output_dir,
                        "command": command,
                    }
                )
                index += 1
    return tasks


def valid_success_marker(task: dict[str, Any]) -> dict[str, Any] | None:
    marker = task["output_dir"] / SUCCESS_MARKER
    test_summary = task["output_dir"] / "test_summary.json"
    if not marker.is_file() or not test_summary.is_file():
        return None
    try:
        result = json.loads(marker.read_text(encoding="utf-8"))
        json.loads(test_summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    result["status"] = "skipped_success"
    return result


def write_summary(results: list[dict[str, Any]], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / SUMMARY_FILE
    temporary = output_root / f".{SUMMARY_FILE}.tmp"
    fields = ["experiment", "name", "fold", "status", "gpu", "start_time", "end_time", "exit_code", "output_path"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item["index"]):
            writer.writerow({field: result.get(field, "") for field in fields})
    os.replace(temporary, path)
    return path


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()


def launch(task: dict[str, Any], gpu_id: int) -> dict[str, Any]:
    output_dir: Path = task["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SUCCESS_MARKER).unlink(missing_ok=True)
    start_time = utc_now()
    env = os.environ.copy()
    env.update(EXTRA_ENV)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_handle = (output_dir / "training.log").open("a", encoding="utf-8", buffering=1)
    log_handle.write(f"\n{'=' * 80}\nStart: {start_time}\nPhysical GPU: {gpu_id}\n")
    log_handle.write(f"Command: {shlex.join(task['command'])}\n{'=' * 80}\n")
    (output_dir / "command.json").write_text(
        json.dumps(
            {
                "experiment": task["experiment"],
                "name": task["name"],
                "fold": task["fold"],
                "gpu": gpu_id,
                "command": task["command"],
                "start_time": start_time,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        task["command"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=(os.name == "posix"),
    )
    return {**task, "gpu": gpu_id, "process": process, "log_handle": log_handle, "start_time": start_time}


def finish_run(run: dict[str, Any]) -> dict[str, Any]:
    process: subprocess.Popen[str] = run["process"]
    exit_code = int(process.returncode)
    run["log_handle"].close()
    has_summary = (run["output_dir"] / "test_summary.json").is_file()
    status = "success" if exit_code == 0 and has_summary else "failed"
    result = {
        "index": run["index"],
        "experiment": run["experiment"],
        "name": run["name"],
        "fold": run["fold"],
        "status": status,
        "gpu": run["gpu"],
        "start_time": run["start_time"],
        "end_time": utc_now(),
        "exit_code": exit_code,
        "output_path": str(run["output_dir"]),
    }
    if status == "success":
        (run["output_dir"] / SUCCESS_MARKER).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Require --batch-size >= 1 and --num-workers >= 0.")
    if len(GPU_IDS) != len(set(GPU_IDS)) or not GPU_IDS:
        raise ValueError("GPU_IDS must contain unique physical GPU indices.")
    if MAX_CONCURRENT_JOBS != 2:
        raise ValueError("MAX_CONCURRENT_JOBS must remain 2 for this experiment suite.")
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")
    tasks = build_tasks(args)

    if args.dry_run:
        print(f"Dry run: {len(tasks)} commands; no files or processes will be created.\n")
        print(f"GPU pool: {GPU_IDS}; maximum concurrent jobs: {MAX_CONCURRENT_JOBS}\n")
        for task in tasks:
            print(f"[{task['index'] + 1:02d}/{len(tasks)}] Experiment {task['experiment']} {task['name']} fold {task['fold']}")
            print(f"  {shlex.join(task['command'])}\n")
        return 0

    results: list[dict[str, Any]] = []
    pending = []
    for task in tasks:
        resumed = valid_success_marker(task) if args.resume else None
        if resumed:
            results.append(resumed)
            print(f"[skip] Experiment {task['experiment']} fold {task['fold']}")
        else:
            pending.append(task)

    active: dict[int, dict[str, Any]] = {}
    last_wait_message = 0.0

    def fill_free_gpus() -> bool:
        """Fill idle GPUs up to the concurrency limit; return whether one launched."""
        nonlocal last_wait_message
        if not pending or len(active) >= MAX_CONCURRENT_JOBS:
            return False
        free_gpus = [gpu_id for gpu_id in query_free_gpus() if gpu_id not in active]
        launch_count = min(MAX_CONCURRENT_JOBS - len(active), len(pending), len(free_gpus))
        launched = False
        for gpu_id in free_gpus[:launch_count]:
            task = pending.pop(0)
            try:
                active[gpu_id] = launch(task, gpu_id)
                launched = True
                print(f"[start] Experiment {task['experiment']} {task['name']} fold {task['fold']} on GPU {gpu_id}")
            except Exception as exc:
                now = utc_now()
                results.append(
                    {
                        "index": task["index"], "experiment": task["experiment"], "name": task["name"],
                        "fold": task["fold"], "status": "failed", "gpu": gpu_id,
                        "start_time": now, "end_time": now, "exit_code": -1,
                        "output_path": str(task["output_dir"]), "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"[fail] Could not launch Experiment {task['experiment']} fold {task['fold']}: {exc}", file=sys.stderr)
        if not launched and pending and len(active) < MAX_CONCURRENT_JOBS:
            now = time.monotonic()
            if now - last_wait_message >= 60:
                print(f"[wait] No eligible GPU among {GPU_IDS}; polling every {GPU_POLL_SECONDS}s")
                last_wait_message = now
        return launched

    interrupted = False
    try:
        while pending or active:
            completed_gpus = [gpu_id for gpu_id, run in active.items() if run["process"].poll() is not None]
            for gpu_id in completed_gpus:
                run = active.pop(gpu_id)
                result = finish_run(run)
                results.append(result)
                label = "done" if result["status"] == "success" else "fail"
                print(f"[{label}] Experiment {result['experiment']} fold {result['fold']} on GPU {result['gpu']} (exit={result['exit_code']})")
                write_summary(results, args.output_root.expanduser().resolve())
            launched = fill_free_gpus()
            if pending or active:
                waiting_for_gpu = pending and len(active) < MAX_CONCURRENT_JOBS and not launched
                time.sleep(GPU_POLL_SECONDS if waiting_for_gpu else 1)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; terminating active training processes...", file=sys.stderr)
    finally:
        for run in active.values():
            terminate_process(run["process"])
            run["log_handle"].close()

    summary = write_summary(results, args.output_root.expanduser().resolve())
    successes = sum(result["status"] in {"success", "skipped_success"} for result in results)
    failures = sum(result["status"] == "failed" for result in results)
    print(f"Summary: {summary}")
    print(f"Successful/skipped: {successes}; failed: {failures}; recorded: {len(results)}/15")
    return 130 if interrupted else (1 if failures else 0)


if __name__ == "__main__":
    raise SystemExit(main())
