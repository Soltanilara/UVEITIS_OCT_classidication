#!/usr/bin/env bash
set -euo pipefail

XLSX_PATH="${1:-dataset/UWF_FP_Annotations_2.8.2026 Names removed.xlsx}"
OUT_PREFIX="${2:-fold}"
N_FOLDS="${N_FOLDS:-5}"
N_VAL="${N_VAL:-10}"
SEED="${SEED:-42}"
SHEET_NAME="${SHEET_NAME:-Data}"

python preprocessing/CSV_k_fold_generator_xlsx.py \
  --xlsx_path "$XLSX_PATH" \
  --sheet_name "$SHEET_NAME" \
  --n_folds "$N_FOLDS" \
  --n_val "$N_VAL" \
  --seed "$SEED" \
  --out_prefix "$OUT_PREFIX" \
  --image_column UWFFP \
  --group_column Patient_ID \
  --drop_missing_zone_rows all
