#!/usr/bin/env bash
set -u

# Fundus zone classification with RETFound-MAE or RETFound-DINOv2.
#
# Examples:
# bash scripts/run_retfound_fundus_zone.sh \
#   --dataset_path "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_OD_canonical" \
#   --pretraining retfound
#
# bash scripts/run_retfound_fundus_zone.sh \
#   --dataset_path "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_OD_canonical" \
#   --pretraining retfound_dinov2

DATASET_PATH=""
PRETRAINING="retfound"
PROTOCOL="finetune"
GPUS_CSV="0,1"
SLOTS_PER_GPU=1
FOLDS_CSV="0,1,2,3,4"
SEEDS_CSV="0,1,2"
ROOT_OUTPUT_DIR="output_retfound_fundus_zone"
LOG_DIR="logs/retfound_fundus_zone"
CSV_PREFIX="fold_final_ready_fundus_only/fold"
THRESHOLDS_JSON=""
FUNDUS_PRETRAINED_CKPT=""
IMAGE_COLUMN="Image File"
IMAGE_ABSOLUTE_COLUMN=""
MASK_COLUMN=""
MASK_ABSOLUTE_COLUMN=""
IMAGE_RESOLVER="direct"
APPLY_MASK=0
INPUT_MODE="full_image_zone_head"
BATCH_SIZE=8
IMAGE_SIZE=224
ZONE_CROP_SIZE=224
RETFOUND_CHECKPOINT="RETFound_mae_natureCFP"
RETFOUND_INPUT_SIZE=224
RETFOUND_CLS_TOKEN=0
RETFOUND_DINOV2_CHECKPOINT="RETFound_dinov2_meh"
RETFOUND_DINOV2_ARCH="dinov2_vitl14"
RETFOUND_DINOV2_INPUT_SIZE=224
WANDB=0
WANDB_PROJECT="uveitis-fundus-zone"
WANDB_ENTITY=""
WANDB_NAME=""
WANDB_TAGS=""
WANDB_MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --pretraining)
      PRETRAINING="$2"
      shift 2
      ;;
    --protocol)
      PROTOCOL="$2"
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
    --image_column)
      IMAGE_COLUMN="$2"
      shift 2
      ;;
    --image_absolute_column)
      IMAGE_ABSOLUTE_COLUMN="$2"
      shift 2
      ;;
    --mask_column)
      MASK_COLUMN="$2"
      shift 2
      ;;
    --mask_absolute_column)
      MASK_ABSOLUTE_COLUMN="$2"
      shift 2
      ;;
    --image_resolver)
      IMAGE_RESOLVER="$2"
      shift 2
      ;;
    --apply_mask)
      APPLY_MASK=1
      shift 1
      ;;
    --input_mode)
      INPUT_MODE="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --image_size)
      IMAGE_SIZE="$2"
      shift 2
      ;;
    --zone_crop_size)
      ZONE_CROP_SIZE="$2"
      shift 2
      ;;
    --retfound_checkpoint)
      RETFOUND_CHECKPOINT="$2"
      shift 2
      ;;
    --retfound_input_size)
      RETFOUND_INPUT_SIZE="$2"
      shift 2
      ;;
    --retfound_cls_token)
      RETFOUND_CLS_TOKEN=1
      shift 1
      ;;
    --retfound_dinov2_checkpoint)
      RETFOUND_DINOV2_CHECKPOINT="$2"
      shift 2
      ;;
    --retfound_dinov2_arch)
      RETFOUND_DINOV2_ARCH="$2"
      shift 2
      ;;
    --retfound_dinov2_input_size)
      RETFOUND_DINOV2_INPUT_SIZE="$2"
      shift 2
      ;;
    --wandb)
      WANDB=1
      shift 1
      ;;
    --wandb_project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb_entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --wandb_name)
      WANDB_NAME="$2"
      shift 2
      ;;
    --wandb_tags)
      WANDB_TAGS="$2"
      shift 2
      ;;
    --wandb_mode)
      WANDB_MODE="$2"
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

if [[ "$PRETRAINING" != "retfound" && "$PRETRAINING" != "retfound_dinov2" ]]; then
  echo "Error: --pretraining must be retfound or retfound_dinov2."
  exit 1
fi

if [[ "$PROTOCOL" != "finetune" && "$PROTOCOL" != "lin_eval" ]]; then
  echo "Error: --protocol must be finetune or lin_eval."
  exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
IFS=',' read -r -a FOLDS <<< "$FOLDS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

mkdir -p "$ROOT_OUTPUT_DIR" "$LOG_DIR"

COMMON_ARGS=(
  --mode train
  --model "$PRETRAINING"
  --protocol "$PROTOCOL"
  --pretraining "$PRETRAINING"
  --input_mode "$INPUT_MODE"
  --drop_missing_zone_rows all
  --image_size "$IMAGE_SIZE"
  --zone_crop_size "$ZONE_CROP_SIZE"
  --batch_size "$BATCH_SIZE"
  --num_epochs 100
  --earlystop
  --brightness
  --contrast
  --image_column "$IMAGE_COLUMN"
  --image_resolver "$IMAGE_RESOLVER"
  --retfound_checkpoint "$RETFOUND_CHECKPOINT"
  --retfound_input_size "$RETFOUND_INPUT_SIZE"
  --retfound_dinov2_arch "$RETFOUND_DINOV2_ARCH"
  --retfound_dinov2_input_size "$RETFOUND_DINOV2_INPUT_SIZE"
)

if [[ "$RETFOUND_CLS_TOKEN" -eq 1 ]]; then
  COMMON_ARGS+=(--retfound_cls_token)
fi

if [[ -n "$RETFOUND_DINOV2_CHECKPOINT" ]]; then
  COMMON_ARGS+=(--retfound_dinov2_checkpoint "$RETFOUND_DINOV2_CHECKPOINT")
fi

if [[ "$WANDB" -eq 1 ]]; then
  COMMON_ARGS+=(--wandb --wandb_project "$WANDB_PROJECT")
  if [[ -n "$WANDB_ENTITY" ]]; then
    COMMON_ARGS+=(--wandb_entity "$WANDB_ENTITY")
  fi
  if [[ -n "$WANDB_TAGS" ]]; then
    COMMON_ARGS+=(--wandb_tags "$WANDB_TAGS")
  fi
  if [[ -n "$WANDB_MODE" ]]; then
    COMMON_ARGS+=(--wandb_mode "$WANDB_MODE")
  fi
fi

if [[ -n "$THRESHOLDS_JSON" ]]; then
  COMMON_ARGS+=(--thresholds_json "$THRESHOLDS_JSON")
fi

if [[ -n "$FUNDUS_PRETRAINED_CKPT" ]]; then
  COMMON_ARGS+=(--fundus_pretrained_ckpt "$FUNDUS_PRETRAINED_CKPT")
fi

if [[ -n "$IMAGE_ABSOLUTE_COLUMN" ]]; then
  COMMON_ARGS+=(--image_absolute_column "$IMAGE_ABSOLUTE_COLUMN")
fi

if [[ -n "$MASK_COLUMN" ]]; then
  COMMON_ARGS+=(--mask_column "$MASK_COLUMN")
fi

if [[ -n "$MASK_ABSOLUTE_COLUMN" ]]; then
  COMMON_ARGS+=(--mask_absolute_column "$MASK_ABSOLUTE_COLUMN")
fi

if [[ "$APPLY_MASK" -eq 1 ]]; then
  COMMON_ARGS+=(--apply_mask)
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
  local out_dir="${ROOT_OUTPUT_DIR}/${PRETRAINING}/${PROTOCOL}/${exp_name}/fold_${fold}/seed_${seed}"
  local log_file="${LOG_DIR}/${PRETRAINING}__${PROTOCOL}__${exp_name}__fold_${fold}__seed_${seed}__gpu_${gpu}.log"
  local tag="${PRETRAINING} ${PROTOCOL} ${exp_name} fold=${fold} seed=${seed} gpu=${gpu}"
  local job_wandb_name="${WANDB_NAME:-${PRETRAINING}_${PROTOCOL}_${exp_name}_fold_${fold}_seed_${seed}}"

  mkdir -p "$out_dir"
  read -r -a EXTRA_ARR <<< "$extra_args"
  local -a JOB_WANDB_ARGS=()
  if [[ "$WANDB" -eq 1 ]]; then
    JOB_WANDB_ARGS+=(--wandb_name "$job_wandb_name")
  fi

  echo "[START] ${tag}"
  python training/train_kFold_binary.py \
    --csvpath "$fold_csv" \
    --dataset_path "$DATASET_PATH" \
    --output_path "$out_dir" \
    --seed "$seed" \
    --gpu "$gpu" \
    "${COMMON_ARGS[@]}" \
    "${JOB_WANDB_ARGS[@]}" \
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

  for pid in "${!PID_TO_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      gpu="${PID_TO_GPU[$pid]}"
      tag="${PID_TO_TAG[$pid]}"
      log_file="${PID_TO_LOG[$pid]}"
      wait "$pid"
      status=$?
      GPU_ACTIVE["$gpu"]=$((GPU_ACTIVE["$gpu"] - 1))
      unset PID_TO_GPU["$pid"]
      unset PID_TO_TAG["$pid"]
      unset PID_TO_LOG["$pid"]

      if [[ "$status" -eq 0 ]]; then
        ok_jobs=$((ok_jobs + 1))
        echo "[DONE] ${tag}"
      else
        failed_jobs=$((failed_jobs + 1))
        echo "[FAIL] ${tag} status=${status} log=${log_file}"
      fi
    fi
  done

  sleep 5
done

echo "Finished ${ok_jobs}/${total_jobs} jobs successfully; failed=${failed_jobs}."
if [[ "$failed_jobs" -gt 0 ]]; then
  exit 1
fi
