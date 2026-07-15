#!/usr/bin/env python3
"""Run an ordered FA augmentation ablation study on two GPUs.

Edit the configuration block below, then run:

    python run_fa_augmentation_experiments.py

Use ``--dry-run`` to inspect commands and ``--resume`` to skip experiments that
already contain a successful completion marker.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# EDITABLE CONFIGURATION
# =============================================================================

# Repository-relative defaults for the FA zone-attention trainer. These remain
# editable and contain no machine-specific absolute paths.
TRAIN_SCRIPT = "training/train_fa_dinov2_zone_attention.py"
BASE_ARGS: list[str] = []
OUTPUT_ROOT = "fa_augmentation_experiments"
EXTRA_ENV: dict[str, str] = {}

# Worker slot -> physical GPU(s) exposed to that worker. Each training process
# sees exactly one GPU, as local cuda:0, even when the physical GPU is GPU 1.
CUDA_VISIBLE_DEVICES: dict[int, str] = {0: "0", 1: "1"}

# The current FA trainer uses --output_path and has no --config argument. For a
# different trainer, these can be changed back to "--config" and "--output".
CONFIG_NAME = ""
CONFIG_FLAG: str | None = None
OUTPUT_FLAG: str | None = "--output_path"
WORKING_DIRECTORY: str | None = str(Path(__file__).resolve().parent)

# Map readable augmentation names to the actual CLI flags accepted by the
# training script. These currently match train_fa_dinov2_zone_attention.py;
# edit only this mapping if another training entry point uses different flags.
AUGMENTATION_FLAGS: dict[str, list[str]] = {
    "rotation": ["--rotation"],
    "translation": ["--translation"],
    "scale": ["--scale"],
    "brightness": ["--brightness"],
    "contrast": ["--contrast"],
    "gamma": ["--gamma"],
    "clahe": ["--clahe"],
    "gaussian_noise": ["--gaussian_noise"],
    "gaussian_blur": ["--gaussian_blur"],
}

# Ten experiments, in launch order. ``extra_args`` can contain any additional
# command-line overrides required for one experiment.
EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "baseline_no_aug",
        "augmentations": [],
        "extra_args": [],
    },
    {
        "name": "geom_only",
        "augmentations": ["rotation", "translation", "scale"],
        "extra_args": [],
    },
    {
        "name": "intensity_only",
        "augmentations": ["brightness", "contrast", "gamma"],
        "extra_args": [],
    },
    {
        "name": "clahe_only",
        "augmentations": ["clahe"],
        "extra_args": [],
    },
    {
        "name": "geom_plus_intensity",
        "augmentations": ["rotation", "translation", "scale", "brightness", "contrast", "gamma"],
        "extra_args": [],
    },
    {
        "name": "geom_plus_clahe",
        "augmentations": ["rotation", "translation", "scale", "clahe"],
        "extra_args": [],
    },
    {
        "name": "intensity_plus_clahe",
        "augmentations": ["brightness", "contrast", "gamma", "clahe"],
        "extra_args": [],
    },
    {
        "name": "geom_intensity_clahe",
        "augmentations": ["rotation", "translation", "scale", "brightness", "contrast", "gamma", "clahe"],
        "extra_args": [],
    },
    {
        "name": "best_core_plus_noise",
        "augmentations": [
            "rotation",
            "translation",
            "scale",
            "brightness",
            "contrast",
            "gamma",
            "clahe",
            "gaussian_noise",
        ],
        "extra_args": [
            "--gaussian_noise_sigma_min",
            "2",
            "--gaussian_noise_sigma_max",
            "5",
        ],
    },
    {
        "name": "best_core_plus_blur",
        "augmentations": [
            "rotation",
            "translation",
            "scale",
            "brightness",
            "contrast",
            "gamma",
            "clahe",
            "gaussian_blur",
        ],
        "extra_args": [
            "--gaussian_blur_sigma_min",
            "0.1",
            "--gaussian_blur_sigma_max",
            "0.5",
        ],
    },
]


# =============================================================================
# IMPLEMENTATION
# =============================================================================

SUCCESS_MARKER = "_SUCCESS.json"
LOG_FILENAME = "training.log"
COMMAND_FILENAME = "command.json"
SUMMARY_FILENAME = "experiment_summary.csv"
_CURRENT_CHILD: subprocess.Popen[str] | None = None


def utc_now() -> str:
    """Return an unambiguous, machine-readable UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print all commands without executing or writing files.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"Skip output directories that already contain {SUCCESS_MARKER}.",
    )
    parser.add_argument("--train-script", default=TRAIN_SCRIPT, help="Override TRAIN_SCRIPT.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT, help="Override OUTPUT_ROOT.")
    parser.add_argument("--config", default=CONFIG_NAME, help="Override CONFIG_NAME.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch training.")
    return parser.parse_args()


def validate_configuration() -> None:
    if len(CUDA_VISIBLE_DEVICES) != 2 or set(CUDA_VISIBLE_DEVICES) != {0, 1}:
        raise ValueError("CUDA_VISIBLE_DEVICES must define exactly worker slots 0 and 1.")
    if len(EXPERIMENTS) != 10:
        raise ValueError(f"Exactly 10 experiments are required; found {len(EXPERIMENTS)}.")
    names = [str(experiment.get("name", "")) for experiment in EXPERIMENTS]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Every experiment must have a unique, non-empty name.")
    for experiment in EXPERIMENTS:
        unknown = set(experiment.get("augmentations", [])) - set(AUGMENTATION_FLAGS)
        if unknown:
            raise ValueError(f"Experiment {experiment['name']!r} uses unknown augmentations: {sorted(unknown)}")


def resolve_training_script(train_script: str) -> Path:
    """Resolve and validate the training entry point before starting workers."""
    path = Path(train_script).expanduser()
    if not path.is_absolute():
        base = Path(WORKING_DIRECTORY) if WORKING_DIRECTORY else Path.cwd()
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Training script not found: {path}. Edit TRAIN_SCRIPT at the top "
            "of this launcher or pass --train-script."
        )
    return path


def augmentation_args(experiment: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for augmentation in experiment.get("augmentations", []):
        args.extend(AUGMENTATION_FLAGS[augmentation])
    args.extend(str(value) for value in experiment.get("extra_args", []))
    return args


def build_command(
    experiment: dict[str, Any],
    output_dir: Path,
    python_executable: str,
    train_script: str,
    config_name: str,
) -> list[str]:
    command = [python_executable, train_script]
    if CONFIG_FLAG and config_name:
        command.extend([CONFIG_FLAG, config_name])
    if OUTPUT_FLAG:
        command.extend([OUTPUT_FLAG, str(output_dir)])
    command.extend(str(value) for value in BASE_ARGS)
    command.extend(augmentation_args(experiment))
    return command


def terminate_subprocess(process: subprocess.Popen[str] | None, timeout: float = 10.0) -> None:
    """Terminate a training process and its process group when possible."""
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=timeout)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass


def worker_signal_handler(signum: int, _frame: Any) -> None:
    """Ensure a worker never leaves its training subprocess behind."""
    terminate_subprocess(_CURRENT_CHILD)
    raise SystemExit(128 + signum)


def run_one_experiment(task: dict[str, Any], gpu_slot: int) -> dict[str, Any]:
    """Run one command, capturing all console output in its output directory."""
    global _CURRENT_CHILD

    name = task["name"]
    output_dir = Path(task["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    success_marker = output_dir / SUCCESS_MARKER
    success_marker.unlink(missing_ok=True)
    start_time = utc_now()
    exit_code = -1
    status = "failed"
    error: str | None = None

    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in EXTRA_ENV.items()})
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES[gpu_slot]

    command_record = {
        "experiment": name,
        "command": task["command"],
        "command_shell": shlex.join(task["command"]),
        "gpu_slot": gpu_slot,
        "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES[gpu_slot],
        "start_time": start_time,
    }
    (output_dir / COMMAND_FILENAME).write_text(json.dumps(command_record, indent=2) + "\n", encoding="utf-8")

    try:
        with (output_dir / LOG_FILENAME).open("a", encoding="utf-8", buffering=1) as log_file:
            log_file.write(f"\n{'=' * 80}\n")
            log_file.write(f"Start: {start_time}\n")
            log_file.write(f"GPU: {CUDA_VISIBLE_DEVICES[gpu_slot]} (worker slot {gpu_slot})\n")
            log_file.write(f"Command: {shlex.join(task['command'])}\n")
            log_file.write(f"{'=' * 80}\n")
            _CURRENT_CHILD = subprocess.Popen(
                task["command"],
                cwd=WORKING_DIRECTORY,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=(os.name == "posix"),
            )
            exit_code = _CURRENT_CHILD.wait()
            status = "success" if exit_code == 0 else "failed"
    except Exception as exc:  # Continue the study even if process creation/logging fails.
        error = f"{type(exc).__name__}: {exc}"
        status = "failed"
        exit_code = _CURRENT_CHILD.returncode if _CURRENT_CHILD and _CURRENT_CHILD.returncode is not None else -1
    finally:
        terminate_subprocess(_CURRENT_CHILD)
        _CURRENT_CHILD = None

    end_time = utc_now()
    result = {
        "index": task["index"],
        "name": name,
        "status": status,
        "gpu": CUDA_VISIBLE_DEVICES[gpu_slot],
        "gpu_slot": gpu_slot,
        "start_time": start_time,
        "end_time": end_time,
        "exit_code": exit_code,
        "output_path": str(output_dir),
        "error": error,
    }
    if status == "success":
        success_marker.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elif error:
        with (output_dir / LOG_FILENAME).open("a", encoding="utf-8") as log_file:
            log_file.write(f"\nLauncher error: {error}\n")
    return result


def gpu_worker(gpu_slot: int, task_queue: mp.Queue[Any], result_queue: mp.Queue[Any]) -> None:
    """Persistent worker permanently assigned to one physical GPU mapping."""
    signal.signal(signal.SIGTERM, worker_signal_handler)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    while True:
        task = task_queue.get()
        if task is None:
            return
        try:
            result = run_one_experiment(task, gpu_slot)
        except Exception as exc:
            now = utc_now()
            result = {
                "index": task["index"],
                "name": task["name"],
                "status": "failed",
                "gpu": CUDA_VISIBLE_DEVICES[gpu_slot],
                "gpu_slot": gpu_slot,
                "start_time": now,
                "end_time": now,
                "exit_code": -1,
                "output_path": task["output_path"],
                "error": f"Unhandled worker error: {type(exc).__name__}: {exc}",
            }
        result_queue.put(result)


def load_resume_result(index: int, name: str, output_dir: Path) -> dict[str, Any] | None:
    marker = output_dir / SUCCESS_MARKER
    if not marker.is_file():
        return None
    try:
        prior = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        prior = {}
    return {
        "index": index,
        "name": name,
        "status": "skipped_success",
        "gpu": prior.get("gpu", ""),
        "gpu_slot": prior.get("gpu_slot", ""),
        "start_time": prior.get("start_time", ""),
        "end_time": prior.get("end_time", ""),
        "exit_code": prior.get("exit_code", 0),
        "output_path": str(output_dir),
        "error": None,
    }


def write_summary(results: list[dict[str, Any]], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / SUMMARY_FILENAME
    temporary_path = output_root / f".{SUMMARY_FILENAME}.tmp"
    fields = ["experiment_name", "status", "gpu_used", "start_time", "end_time", "exit_code", "output_path"]
    with temporary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item["index"]):
            writer.writerow(
                {
                    "experiment_name": result["name"],
                    "status": result["status"],
                    "gpu_used": result["gpu"],
                    "start_time": result["start_time"],
                    "end_time": result["end_time"],
                    "exit_code": result["exit_code"],
                    "output_path": result["output_path"],
                }
            )
    os.replace(temporary_path, summary_path)
    return summary_path


def stop_workers(workers: dict[int, mp.Process], task_queues: dict[int, mp.Queue[Any]]) -> None:
    """Stop workers gracefully, escalating if a child does not exit."""
    for gpu_slot, process in workers.items():
        if process.is_alive():
            try:
                task_queues[gpu_slot].put_nowait(None)
            except queue.Full:
                pass
    for process in workers.values():
        process.join(timeout=5.0)
    for process in workers.values():
        if process.is_alive():
            process.terminate()  # Worker handler terminates its training process group.
    for process in workers.values():
        process.join(timeout=10.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=5.0)


def main() -> int:
    args = parse_args()
    validate_configuration()
    if not args.dry_run:
        resolve_training_script(args.train_script)
    output_root = Path(args.output_root).expanduser().resolve()

    tasks: list[dict[str, Any]] = []
    for index, experiment in enumerate(EXPERIMENTS):
        output_dir = output_root / experiment["name"]
        tasks.append(
            {
                "index": index,
                "name": experiment["name"],
                "output_path": str(output_dir),
                "command": build_command(
                    experiment,
                    output_dir,
                    python_executable=args.python,
                    train_script=args.train_script,
                    config_name=args.config,
                ),
            }
        )

    if args.dry_run:
        print("Dry run: no processes or files will be created.\n")
        for task in tasks:
            planned_slot = task["index"] % 2
            print(f"[{task['index'] + 1:02d}/10] {task['name']} | planned GPU {CUDA_VISIBLE_DEVICES[planned_slot]}")
            print(f"  {shlex.join(task['command'])}\n")
        return 0

    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in tasks:
        resumed = load_resume_result(task["index"], task["name"], Path(task["output_path"])) if args.resume else None
        if resumed is not None:
            results.append(resumed)
            print(f"[skip] {task['name']}: success marker found")
        else:
            pending.append(task)

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    task_queues = {slot: context.Queue(maxsize=1) for slot in (0, 1)}
    workers: dict[int, mp.Process] = {}
    active: dict[int, dict[str, Any]] = {}
    launcher_error: str | None = None

    def request_shutdown(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, request_shutdown)

    def start_worker(slot: int) -> None:
        worker = context.Process(
            target=gpu_worker,
            args=(slot, task_queues[slot], result_queue),
            name=f"gpu-worker-{slot}",
        )
        worker.start()
        workers[slot] = worker

    def dispatch_next(slot: int) -> bool:
        if not pending:
            return False
        task = pending.pop(0)  # FIFO preserves the experiment launch order.
        task["dispatch_time"] = utc_now()
        active[slot] = task
        task_queues[slot].put(task)
        print(
            f"[start] {task['index'] + 1:02d}/10 {task['name']} "
            f"on GPU {CUDA_VISIBLE_DEVICES[slot]} -> {task['output_path']}"
        )
        return True

    interrupted = False
    try:
        for slot in (0, 1):
            start_worker(slot)
            dispatch_next(slot)

        while active:
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty:
                # Detect an unexpected worker death instead of hanging forever.
                for slot, worker in list(workers.items()):
                    if slot in active and not worker.is_alive():
                        task = active.pop(slot)
                        now = utc_now()
                        result = {
                            "index": task["index"],
                            "name": task["name"],
                            "status": "failed",
                            "gpu": CUDA_VISIBLE_DEVICES[slot],
                            "gpu_slot": slot,
                            "start_time": "",
                            "end_time": now,
                            "exit_code": worker.exitcode if worker.exitcode is not None else -1,
                            "output_path": task["output_path"],
                            "error": "GPU worker exited unexpectedly.",
                        }
                        results.append(result)
                        print(f"[fail]  {task['name']} on GPU {CUDA_VISIBLE_DEVICES[slot]}: worker exited unexpectedly")
                        if pending:
                            # Discard the old queue in case the dead worker failed
                            # before consuming its task.
                            task_queues[slot] = context.Queue(maxsize=1)
                            start_worker(slot)
                            dispatch_next(slot)
                continue

            slot = int(result["gpu_slot"])
            active.pop(slot, None)
            results.append(result)
            label = "done" if result["status"] == "success" else "fail"
            print(
                f"[{label}]  {result['name']} on GPU {result['gpu']} "
                f"(exit={result['exit_code']})"
            )
            dispatch_next(slot)  # Immediately refill the GPU that became free.
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; terminating workers and their training processes...", file=sys.stderr)
    except Exception as exc:
        launcher_error = f"{type(exc).__name__}: {exc}"
        print(f"\nLauncher error: {launcher_error}", file=sys.stderr)
    finally:
        stop_workers(workers, task_queues)
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    # If the launcher itself was stopped, still produce a complete ten-row
    # summary rather than silently omitting active and pending experiments.
    if interrupted or launcher_error:
        terminal_status = "interrupted" if interrupted else "launcher_failed"
        now = utc_now()
        recorded_indices = {int(result["index"]) for result in results}
        for slot, task in active.items():
            if task["index"] not in recorded_indices:
                results.append(
                    {
                        "index": task["index"],
                        "name": task["name"],
                        "status": terminal_status,
                        "gpu": CUDA_VISIBLE_DEVICES[slot],
                        "gpu_slot": slot,
                        "start_time": task.get("dispatch_time", ""),
                        "end_time": now,
                        "exit_code": -1,
                        "output_path": task["output_path"],
                        "error": launcher_error,
                    }
                )
        for task in pending:
            if task["index"] not in recorded_indices:
                results.append(
                    {
                        "index": task["index"],
                        "name": task["name"],
                        "status": "not_started",
                        "gpu": "",
                        "gpu_slot": "",
                        "start_time": "",
                        "end_time": now,
                        "exit_code": "",
                        "output_path": task["output_path"],
                        "error": launcher_error,
                    }
                )

    summary_path = write_summary(results, output_root)
    successes = sum(result["status"] in {"success", "skipped_success"} for result in results)
    failures = sum(result["status"] == "failed" for result in results)
    print(f"Summary: {summary_path}")
    print(f"Successful/skipped: {successes}; failed: {failures}; recorded: {len(results)}/10")
    if interrupted:
        return 130
    if launcher_error:
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
