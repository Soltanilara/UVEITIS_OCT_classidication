#!/usr/bin/env bash
set -u

# Zone-crop-only ResNet50 sweep.
#
# Default sweep:
# - 5 folds
# - 3 seeds
# - shared ResNet50 crop encoder
# - fixed 10-zone crop template
# - corrected zone masking
#
# Example:
# bash scripts/run_zone_crops_only.sh \
#   --dataset_path "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_OD_canonical"

DATASET_PATH=""
ZONE_TEMPLATE_JSON="configs/fundus_10zone_template.json"
GPUS_CSV="0,1"
SLOTS_PER_GPU=2
FOLDS_CSV="0,1,2,3,4"
SEEDS_CSV="0,1,2"
ROOT_OUTPUT_DIR="output_fundus_zone_crops"
LOG_DIR="logs/fundus_zone_crops"
CSV_PREFIX="fold"
THRESHOLDS_JSON=""
FUNDUS_PRETRAINED_CKPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --zone_template_json)
      ZONE_TEMPLATE_JSON="$2"
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
    --seeds)
      SEEDS_CSV="$2"
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
    --csv_prefix)
      CSV_PREFIX="$2"
      shift 2
      ;;
    --thresholds_json)
      THRESHOLDS_JSON="$2"
      shift 2
      ;;
    --fundus_pretrained_ckpt)
      FUNDUS_PRETRAINED_CKPT="$2"
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
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

mkdir -p "$ROOT_OUTPUT_DIR" "$LOG_DIR"

COMMON_ARGS=(
  --mode train
  --model resnet50
  --protocol finetune
  --pretraining supervised
  --input_mode zone_crops_shared
  --zone_template_json "$ZONE_TEMPLATE_JSON"
  --drop_missing_zone_rows all
  --image_size 224
  --zone_crop_size 224
  --batch_size 24
  --num_epochs 100
  --earlystop
  --brightness
  --contrast
)

if [[ -n "$THRESHOLDS_JSON" ]]; then
  COMMON_ARGS+=(--thresholds_json "$THRESHOLDS_JSON")
fi

if [[ -n "$FUNDUS_PRETRAINED_CKPT" ]]; then
  COMMON_ARGS+=(--fundus_pretrained_ckpt "$FUNDUS_PRETRAINED_CKPT")
fi

EXPERIMENTS=(
  "ce_class_weighted|--loss CE"
  "focal_g1p5_class_weighted|--loss focal --gamma 1.5"
  "ce_label_smoothing_0p05_class_weighted|--loss CE --label_smoothing 0.05"
)

JOBS=()
for exp in "${EXPERIMENTS[@]}"; do
  exp_name="${exp%%|*}"
  exp_args="${exp#*|}"
  for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      JOBS+=("${exp_name}|${fold}|${seed}|${exp_args}")
    done
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

  local exp_name fold seed extra_args
  exp_name="${payload%%|*}"
  payload="${payload#*|}"
  fold="${payload%%|*}"
  payload="${payload#*|}"
  seed="${payload%%|*}"
  extra_args="${payload#*|}"

  local fold_csv="${CSV_PREFIX}_${fold}"
  local out_dir="${ROOT_OUTPUT_DIR}/${exp_name}/fold_${fold}/seed_${seed}"
  local log_file="${LOG_DIR}/${exp_name}__fold_${fold}__seed_${seed}__gpu_${gpu}.log"
  local tag="${exp_name} fold=${fold} seed=${seed} gpu=${gpu}"

  mkdir -p "$out_dir"
  read -r -a EXTRA_ARR <<< "$extra_args"

  echo "[START] ${tag}"
  python training/train_kFold_binary.py \
    --csvpath "$fold_csv" \
    --dataset_path "$DATASET_PATH" \
    --output_path "$out_dir" \
    --seed "$seed" \
    --gpu "$gpu" \
    "${COMMON_ARGS[@]}" \
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
