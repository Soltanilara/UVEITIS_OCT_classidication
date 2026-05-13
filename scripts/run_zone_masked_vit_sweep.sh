#!/usr/bin/env bash
set -euo pipefail

# Canonical masked-zone sweep for ViT-family backbones.
#
# Mirrors the ResNet canonical masked-zone sweep, but uses a ViT backbone in
# zone_masked_shared mode for fair comparison.
#
# Supported examples:
# 1) Classic ViT:
#    bash scripts/run_zone_masked_vit_sweep.sh \
#      --dataset_path "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical" \
#      --ready_csv "/home/shashank/UVEITIS_OCT_classidication/logs/final_fa_training_dataset_20260428/fa_final_master_unique_paths_ready.csv"
#
# 2) DINOv2 ViT:
#    bash scripts/run_zone_masked_vit_sweep.sh \
#      --dataset_path "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical" \
#      --ready_csv "/home/shashank/UVEITIS_OCT_classidication/logs/final_fa_training_dataset_20260428/fa_final_master_unique_paths_ready.csv" \
#      --model vitb14_dino \
#      --pretraining dino \
#      --batch_size 16
#
# 3) Run both backbones in one sweep:
#    bash scripts/run_zone_masked_vit_sweep.sh \
#      --dataset_path "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical" \
#      --ready_csv "/home/shashank/UVEITIS_OCT_classidication/logs/final_fa_training_dataset_20260428/fa_final_master_unique_paths_ready.csv" \
#      --backbones both

DATASET_PATH=""
READY_CSV=""
INPUT_FOLDS_ROOT="fold_masked_server_clean"
PREPARED_FOLDS_ROOT="fold_masked_vit_canonical_v2"
MASK_SUFFIX="_masks_v2.npy"
GPUS_CSV="0,1"
SLOTS_PER_GPU=1
FOLDS_CSV="0,1,2,3,4"
SEEDS_CSV="0,1,2"
ROOT_OUTPUT_DIR="output_zone_masked_vit_canonical_v2"
LOG_DIR="logs/zone_masked_vit_canonical_v2"
MODEL="L_16_imagenet1k"
PRETRAINING="vit"
BATCH_SIZE=24
DINO_BATCH_SIZE=16
BACKBONES="single"
IMAGE_SIZE=224
ZONE_CROP_SIZE=224

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --ready_csv)
      READY_CSV="$2"
      shift 2
      ;;
    --input_folds_root)
      INPUT_FOLDS_ROOT="$2"
      shift 2
      ;;
    --prepared_folds_root)
      PREPARED_FOLDS_ROOT="$2"
      shift 2
      ;;
    --mask_suffix)
      MASK_SUFFIX="$2"
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
    --model)
      MODEL="$2"
      shift 2
      ;;
    --pretraining)
      PRETRAINING="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --dino_batch_size)
      DINO_BATCH_SIZE="$2"
      shift 2
      ;;
    --backbones)
      BACKBONES="$2"
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

if [[ -z "$READY_CSV" ]]; then
  echo "Error: --ready_csv is required."
  exit 1
fi

validate_pair() {
  local model="$1"
  local pretraining="$2"

  case "$model" in
    L_16_imagenet1k|vitb14_dino)
      ;;
    *)
      echo "Error: unsupported model ${model}. Use L_16_imagenet1k or vitb14_dino."
      exit 1
      ;;
  esac

  case "$pretraining" in
    vit|dino|oct)
      ;;
    *)
      echo "Error: unsupported pretraining ${pretraining}. Use vit, dino, or oct."
      exit 1
      ;;
  esac

  if [[ "$model" == "L_16_imagenet1k" && "$pretraining" == "dino" ]]; then
    echo "Error: model L_16_imagenet1k is not supported with --pretraining dino."
    exit 1
  fi

  if [[ "$model" == "vitb14_dino" && "$pretraining" == "vit" ]]; then
    echo "Error: model vitb14_dino is not supported with --pretraining vit."
    exit 1
  fi
}

case "$BACKBONES" in
  single)
    validate_pair "$MODEL" "$PRETRAINING"
    ;;
  both)
    ;;
  *)
    echo "Error: unsupported --backbones ${BACKBONES}. Use single or both."
    exit 1
    ;;
esac

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
IFS=',' read -r -a FOLDS <<< "$FOLDS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "Error: provide at least one GPU."
  exit 1
fi

mkdir -p "$ROOT_OUTPUT_DIR" "$LOG_DIR"

echo "[PREP] Enriching folds into ${PREPARED_FOLDS_ROOT} using mask suffix ${MASK_SUFFIX}"
python scripts/attach_fa_masks_to_splits.py \
  --ready-csv "$READY_CSV" \
  --input-root "$INPUT_FOLDS_ROOT" \
  --output-root "$PREPARED_FOLDS_ROOT" \
  --mask-suffix "$MASK_SUFFIX" \
  --drop-missing-mask

BACKBONE_CONFIGS=()
if [[ "$BACKBONES" == "both" ]]; then
  BACKBONE_CONFIGS+=("L_16_imagenet1k|vit|${BATCH_SIZE}")
  BACKBONE_CONFIGS+=("vitb14_dino|dino|${DINO_BATCH_SIZE}")
else
  BACKBONE_CONFIGS+=("${MODEL}|${PRETRAINING}|${BATCH_SIZE}")
fi

EXPERIMENTS=(
  "ce_class_weighted|--loss CE"
  "focal_g1p5_class_weighted|--loss focal --gamma 1.5"
  "ce_label_smoothing_0p05_class_weighted|--loss CE --label_smoothing 0.05"
)

JOBS=()
for backbone in "${BACKBONE_CONFIGS[@]}"; do
  backbone_model="${backbone%%|*}"
  rest="${backbone#*|}"
  backbone_pretraining="${rest%%|*}"
  backbone_batch_size="${rest##*|}"
  for exp in "${EXPERIMENTS[@]}"; do
    exp_name="${exp%%|*}"
    exp_args="${exp#*|}"
    for fold in "${FOLDS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        JOBS+=("${backbone_model}|${backbone_pretraining}|${backbone_batch_size}|${exp_name}|${fold}|${seed}|${exp_args}")
      done
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

  local model pretraining batch_size exp_name fold seed extra_args
  model="${payload%%|*}"
  payload="${payload#*|}"
  pretraining="${payload%%|*}"
  payload="${payload#*|}"
  batch_size="${payload%%|*}"
  payload="${payload#*|}"
  exp_name="${payload%%|*}"
  payload="${payload#*|}"
  fold="${payload%%|*}"
  payload="${payload#*|}"
  seed="${payload%%|*}"
  extra_args="${payload#*|}"

  local backbone_tag="${model}_${pretraining}"
  local fold_csv="${PREPARED_FOLDS_ROOT}/fold_${fold}"
  local out_dir="${ROOT_OUTPUT_DIR}/${backbone_tag}/${exp_name}/fold_${fold}/seed_${seed}"
  local log_file="${LOG_DIR}/${backbone_tag}__${exp_name}__fold_${fold}__seed_${seed}__gpu_${gpu}.log"
  local tag="${backbone_tag} ${exp_name} fold=${fold} seed=${seed} gpu=${gpu}"
  local -a common_args=(
    --mode train
    --model "$model"
    --protocol finetune
    --pretraining "$pretraining"
    --input_mode zone_masked_shared
    --drop_missing_zone_rows all
    --image_size "$IMAGE_SIZE"
    --zone_crop_size "$ZONE_CROP_SIZE"
    --batch_size "$batch_size"
    --num_epochs 100
    --earlystop
    --brightness
    --contrast
    --image_column "Image File"
    --mask_column "FA_Mask_Path"
    --image_resolver "fundus_from_fa_pair"
  )

  mkdir -p "$out_dir"
  read -r -a EXTRA_ARR <<< "$extra_args"

  echo "[START] ${tag}"
  python training/train_kFold_binary.py \
    --csvpath "$fold_csv" \
    --dataset_path "$DATASET_PATH" \
    --output_path "$out_dir" \
    --seed "$seed" \
    --gpu "$gpu" \
    "${common_args[@]}" \
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
