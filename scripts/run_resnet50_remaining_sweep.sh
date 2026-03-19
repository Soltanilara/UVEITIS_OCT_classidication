#!/usr/bin/env bash
set -u

# Sweep remaining ResNet50 configs across folds 0..4.
# Excludes runs already completed by user:
# 1) protocol=scratch, model=resnet50, loss=CE, --unweighted
# 2) protocol=finetune, pretraining=supervised, model=resnet50, loss=CE, --unweighted
#
# Concurrency: up to 4 jobs per GPU (default), across 2 GPUs.
#
# Example:
# bash scripts/run_resnet50_remaining_sweep.sh \
#   --dataset_path "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_OD_canonical"

DATASET_PATH=""
GPUS_CSV="0,1"
SLOTS_PER_GPU=4
FOLDS_CSV="0,1,2,3,4"
ROOT_OUTPUT_DIR="output_resnet50_sweep_remaining"
LOG_DIR="logs/resnet50_sweep_remaining"

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
    --slots_per_gpu)
      SLOTS_PER_GPU="$2"
      shift 2
      ;;
    --folds)
      FOLDS_CSV="$2"
      shift 2
      ;;
    --root_output_dir)
      ROOT_OUTPUT_DIR="$2"
      shift 2
      ;;
    --log_dir)
      LOG_DIR="$2"
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
  echo "Error: expected exactly 2 GPUs (e.g. --gpus \"0,1\")."
  exit 1
fi

mkdir -p "$ROOT_OUTPUT_DIR" "$LOG_DIR"

# Remaining suggested configs (excluding already-completed runs).
# Format: "experiment_name|extra args"
EXPERIMENTS=(
  "lin_eval_supervised_ce_unweighted|--protocol lin_eval --pretraining supervised --loss CE --unweighted"
  "lin_eval_supervised_ce_class_weighted|--protocol lin_eval --pretraining supervised --loss CE"
  "lin_eval_supervised_ce_weighted_sampling_unweighted|--protocol lin_eval --pretraining supervised --loss CE --unweighted --weightedSampling"
  "lin_eval_supervised_focal_g1p5_unweighted|--protocol lin_eval --pretraining supervised --loss focal --gamma 1.5 --unweighted"
  "lin_eval_supervised_focal_g2p0_unweighted|--protocol lin_eval --pretraining supervised --loss focal --gamma 2.0 --unweighted"
  "lin_eval_supervised_ce_unweighted_hflip|--protocol lin_eval --pretraining supervised --loss CE --unweighted --hflip"
  "lin_eval_supervised_ce_unweighted_hflip_brightness_contrast|--protocol lin_eval --pretraining supervised --loss CE --unweighted --hflip --brightness --contrast"
  "lin_eval_supervised_ce_unweighted_hflip_elastic_gnoise|--protocol lin_eval --pretraining supervised --loss CE --unweighted --hflip --elastic --gnoise"
  "lin_eval_supervised_ce_unweighted_mixup_0p2|--protocol lin_eval --pretraining supervised --loss CE --unweighted --mixup --mixup_alpha 0.2 --mixup_beta 0.2"
  "lin_eval_supervised_ce_unweighted_mixup_0p8|--protocol lin_eval --pretraining supervised --loss CE --unweighted --mixup --mixup_alpha 0.8 --mixup_beta 0.8"
  "finetune_supervised_focal_g1p5_unweighted|--protocol finetune --pretraining supervised --loss focal --gamma 1.5 --unweighted"
  "finetune_supervised_focal_g2p0_unweighted|--protocol finetune --pretraining supervised --loss focal --gamma 2.0 --unweighted"
  "finetune_supervised_ce_weighted_sampling_unweighted|--protocol finetune --pretraining supervised --loss CE --unweighted --weightedSampling"
  "finetune_supervised_ce_class_weighted|--protocol finetune --pretraining supervised --loss CE"
  "finetune_supervised_ce_unweighted_hflip|--protocol finetune --pretraining supervised --loss CE --unweighted --hflip"
  "finetune_supervised_ce_unweighted_hflip_brightness_contrast|--protocol finetune --pretraining supervised --loss CE --unweighted --hflip --brightness --contrast"
  "finetune_supervised_ce_unweighted_hflip_elastic_gnoise|--protocol finetune --pretraining supervised --loss CE --unweighted --hflip --elastic --gnoise"
  "finetune_supervised_ce_unweighted_mixup_0p2|--protocol finetune --pretraining supervised --loss CE --unweighted --mixup --mixup_alpha 0.2 --mixup_beta 0.2"
  "finetune_supervised_ce_unweighted_mixup_0p8|--protocol finetune --pretraining supervised --loss CE --unweighted --mixup --mixup_alpha 0.8 --mixup_beta 0.8"
)

# Build job queue: one job per (experiment, fold)
JOBS=()
for exp in "${EXPERIMENTS[@]}"; do
  exp_name="${exp%%|*}"
  exp_args="${exp#*|}"
  for fold in "${FOLDS[@]}"; do
    JOBS+=("${exp_name}|${fold}|${exp_args}")
  done
done

declare -A GPU_ACTIVE
for gpu in "${GPUS[@]}"; do
  GPU_ACTIVE["$gpu"]=0
done

declare -A PID_TO_GPU
declare -A PID_TO_TAG
declare -A PID_TO_LOG

job_idx=0
total_jobs="${#JOBS[@]}"
ok_jobs=0
failed_jobs=0

start_job() {
  local gpu="$1"
  local payload="$2"

  local exp_name fold extra_args
  exp_name="${payload%%|*}"
  payload="${payload#*|}"
  fold="${payload%%|*}"
  extra_args="${payload#*|}"

  local fold_csv="fold_${fold}"
  local out_dir="${ROOT_OUTPUT_DIR}/${exp_name}/fold_${fold}"
  local log_file="${LOG_DIR}/${exp_name}__fold_${fold}__gpu_${gpu}.log"
  local tag="${exp_name} fold=${fold} gpu=${gpu}"

  mkdir -p "$out_dir"

  read -r -a EXTRA_ARR <<< "$extra_args"

  echo "[START] ${tag}"
  python training/train_kFold.py \
    --mode train \
    --csvpath "$fold_csv" \
    --dataset_path "$DATASET_PATH" \
    --output_path "$out_dir" \
    --model resnet50 \
    --image_size 224 \
    --batch_size 24 \
    --num_epochs 100 \
    --earlystop \
    --gpu "$gpu" \
    "${EXTRA_ARR[@]}" \
    > "$log_file" 2>&1 &

  local pid=$!
  PID_TO_GPU["$pid"]="$gpu"
  PID_TO_TAG["$pid"]="$tag"
  PID_TO_LOG["$pid"]="$log_file"
  GPU_ACTIVE["$gpu"]=$((GPU_ACTIVE["$gpu"] + 1))
}

has_open_slot() {
  local g
  for g in "${GPUS[@]}"; do
    if [[ "${GPU_ACTIVE[$g]}" -lt "$SLOTS_PER_GPU" ]]; then
      return 0
    fi
  done
  return 1
}

next_open_gpu() {
  local g
  for g in "${GPUS[@]}"; do
    if [[ "${GPU_ACTIVE[$g]}" -lt "$SLOTS_PER_GPU" ]]; then
      echo "$g"
      return 0
    fi
  done
  return 1
}

# Main scheduler loop
while [[ "$job_idx" -lt "$total_jobs" || "${#PID_TO_GPU[@]}" -gt 0 ]]; do
  while [[ "$job_idx" -lt "$total_jobs" ]] && has_open_slot; do
    g="$(next_open_gpu)"
    start_job "$g" "${JOBS[$job_idx]}"
    job_idx=$((job_idx + 1))
  done

  if [[ "${#PID_TO_GPU[@]}" -eq 0 ]]; then
    break
  fi

  wait -n

  # Clean up all exited pids (usually one per cycle).
  for pid in "${!PID_TO_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      gpu="${PID_TO_GPU[$pid]}"
      tag="${PID_TO_TAG[$pid]}"
      logf="${PID_TO_LOG[$pid]}"

      if wait "$pid"; then
        echo "[DONE] ${tag}"
        ok_jobs=$((ok_jobs + 1))
      else
        rc=$?
        echo "[FAIL] ${tag} exit_code=${rc} log=${logf}"
        failed_jobs=$((failed_jobs + 1))
      fi

      GPU_ACTIVE["$gpu"]=$((GPU_ACTIVE["$gpu"] - 1))
      unset "PID_TO_GPU[$pid]"
      unset "PID_TO_TAG[$pid]"
      unset "PID_TO_LOG[$pid]"
    fi
  done
done

echo "[SUMMARY] total_jobs=${total_jobs} success=${ok_jobs} failed=${failed_jobs}"
if [[ "$failed_jobs" -gt 0 ]]; then
  exit 1
fi
