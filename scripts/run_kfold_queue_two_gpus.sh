#!/usr/bin/env bash
set -u

# Queue 5-fold training jobs across two GPUs with at most one job per GPU.
# As soon as one fold finishes, the next fold starts on that freed GPU.
#
# Usage:
#   bash scripts/run_kfold_queue_two_gpus.sh \
#     --dataset_path "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_OD_canonical"
#
# Optional:
#   --gpus "0,1"                 (default: 0,1)
#   --folds "0,1,2,3,4"          (default: 0,1,2,3,4)
#   --output_prefix "output_fold" (default: output_fold)
#   --extra_args "--hflip"       (passed through to train_kFold.py)

DATASET_PATH=""
GPUS_CSV="0,1"
FOLDS_CSV="0,1,2,3,4"
OUTPUT_PREFIX="output_fold"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --gpus)
      GPUS_CSV="$2"
      shift 2
      ;;
    --folds)
      FOLDS_CSV="$2"
      shift 2
      ;;
    --output_prefix)
      OUTPUT_PREFIX="$2"
      shift 2
      ;;
    --extra_args)
      EXTRA_ARGS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET_PATH" ]]; then
  echo "Error: --dataset_path is required."
  exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
IFS=',' read -r -a FOLDS <<< "$FOLDS_CSV"

if [[ "${#GPUS[@]}" -ne 2 ]]; then
  echo "Error: This script expects exactly 2 GPUs via --gpus (e.g. \"0,1\")."
  exit 1
fi

mkdir -p logs

declare -A PID_TO_GPU
declare -A PID_TO_FOLD

next_fold_idx=0
completed=0
failed=0
total="${#FOLDS[@]}"

start_job() {
  local fold="$1"
  local gpu="$2"
  local output_dir="${OUTPUT_PREFIX}_${fold}_zones"
  local log_file="logs/train_fold_${fold}_gpu_${gpu}.log"

  echo "[START] fold=${fold} gpu=${gpu} output=${output_dir} log=${log_file}"

  # shellcheck disable=SC2086
  python training/train_kFold.py \
    --mode train \
    --csvpath "fold_${fold}" \
    --dataset_path "$DATASET_PATH" \
    --output_path "$output_dir" \
    --protocol scratch \
    --model resnet50 \
    --loss CE \
    --unweighted \
    --image_size 224 \
    --batch_size 24 \
    --num_epochs 100 \
    --earlystop \
    --gpu "$gpu" \
    $EXTRA_ARGS \
    > "$log_file" 2>&1 &

  local pid=$!
  PID_TO_GPU["$pid"]="$gpu"
  PID_TO_FOLD["$pid"]="$fold"
}

# Initial launch: up to one job per GPU
for gpu in "${GPUS[@]}"; do
  if [[ "$next_fold_idx" -lt "$total" ]]; then
    start_job "${FOLDS[$next_fold_idx]}" "$gpu"
    next_fold_idx=$((next_fold_idx + 1))
  fi
done

while [[ "${#PID_TO_GPU[@]}" -gt 0 ]]; do
  wait -n

  # Identify which PID(s) exited; usually one, but handle multiple safely.
  for pid in "${!PID_TO_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      gpu="${PID_TO_GPU[$pid]}"
      fold="${PID_TO_FOLD[$pid]}"

      if wait "$pid"; then
        echo "[DONE] fold=${fold} gpu=${gpu}"
        completed=$((completed + 1))
      else
        rc=$?
        echo "[FAIL] fold=${fold} gpu=${gpu} exit_code=${rc}"
        failed=$((failed + 1))
      fi

      unset "PID_TO_GPU[$pid]"
      unset "PID_TO_FOLD[$pid]"

      if [[ "$next_fold_idx" -lt "$total" ]]; then
        start_job "${FOLDS[$next_fold_idx]}" "$gpu"
        next_fold_idx=$((next_fold_idx + 1))
      fi
    fi
  done
done

echo "[SUMMARY] total=${total} completed=${completed} failed=${failed}"
if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
