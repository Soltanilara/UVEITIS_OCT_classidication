#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical}"
INPUT_CSV="${2:-fa_final_training_dataset_ready.csv}"
OUTPUT_ROOT="${3:-fold_final_ready_fundus_only}"

python preprocessing/CSV_k_fold_generator_ready_csv.py \
  --csv_path "$INPUT_CSV" \
  --output_root "$OUTPUT_ROOT" \
  --n_folds 5 \
  --n_val 10 \
  --seed 42 \
  --image_column UWFFP \
  --group_column Patient_ID \
  --drop_missing_zone_rows all \
  --dataset_root "$DATASET_ROOT" \
  --drop_missing_images
