# UVEITIS OCT Classification

Script-based research repository for OCT uveitis experiments (binary, graded, k-fold, explainability, and analysis).

## Repository Layout
- `training/`: model training and pretraining entrypoints
- `evaluation/`: evaluation-focused scripts
- `preprocessing/`: split generation and image/data preprocessing
- `explainability/`: Grad-CAM / IG / SHAP and meta-score utilities
- `analysis/`: post-hoc metrics and visualization scripts
- `scripts/`: saved experiment command lists

See [REPO_STRUCTURE.md](REPO_STRUCTURE.md) for a short description of each folder.

## Run From Repo Root
All commands below assume your current working directory is the repository root:

```bash
cd /home/mshashank02/UVEITIS_OCT_classidication
```

## Environment
Use a Python environment with the dependencies used by scripts, including:
- `torch`, `torchvision`, `timm`
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`, `Pillow`
- `opencv-python`, `captum`, `shap`, `pytorch-pretrained-vit`
- `huggingface_hub` if using automatic RETFound checkpoint download
- `wandb` if using optional Weights & Biases logging

## Data Expectations
- Split CSVs should contain at least:
  - `Image File`
  - `Label`
- Binary scripts treat:
  - `negative -> 0`
  - all other labels -> `1`
- Common defaults:
  - dataset directory: `Dataset 01032025`
  - split directories: `split_*` or `fold_*`

## Cleaned FA/FP Annotation Files

The cleaned full-dataset FA annotation file is:

```text
uveitis_fa_annotations_cleaned.csv
```

It contains one row per annotated FA image and includes:

- `Image_File(FA)`: dataset-relative FA image path, e.g. `Patient010/20240723/Patient010_20240723_OD_FA_0001.png`
- `UWFFA`: original full Windows-style FA source path from the annotation sheet
- `UWFFP`: original full Windows-style paired fundus photo path
- `Patient_ID`, `Eye`, `Visit_Date`
- `Zone1_label` ... `Zone10_label`
- path-cleaning audit columns such as `FA_Path_Correction_Status`, `FA_Path_Correction_Reason`, and `Row_Exclusion_Status`

On the remote server, the canonical image dataset root is:

```text
/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical/
```

Because this path contains spaces, quote it in shell commands:

```bash
cd "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical"
```

Rows that could not produce a usable zone mask are excluded from:

```text
uveitis_fa_annotations_cleaned_zone_masks_ready.csv
```

That ready CSV is derived from the original cleaned CSV and does not delete or modify any source images, masks, or the original annotation file.

## FA Zone Masks and FP-Masked PNGs

The batch script for creating FA zone masks and applying them to paired FP images is:

```text
batch_slice_fa_apply_fp.py
```

Typical remote-server command:

```bash
python batch_slice_fa_apply_fp.py \
  --output-root "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical_fa_zone_masks" \
  --resume \
  --allow-missing-fp \
  --fallback-existing-mask \
  --no-label-png \
  --workers 4
```

Important flags:

- `--resume`: skip rows whose expected outputs already exist
- `--allow-missing-fp`: still create FA masks when a paired FP image is missing
- `--fallback-existing-mask`: if yellow crosshair detection fails, reuse a sibling `*_masks_v2.npy`, `*_masks.npy`, or `*_zone_masks.npy` mask when available
- `--no-label-png`: skip debug label PNGs to reduce output size and write time
- `--workers N`: process multiple rows in parallel

The output root used for the canonical run is:

```text
/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical_fa_zone_masks/
```

Expected output structure:

```text
batch_stdout.log
manifest.csv
failures.csv
summary.json
masks_npy/
mask_labels_png/
fp_masked_zones/
```

`masks_npy/` contains the main FA zone masks. Each file is a NumPy array with shape:

```text
(10, H, W)
```

The first axis is the zone index:

```text
0 = Zone 1
1 = Zone 2
...
9 = Zone 10
```

Values are binary:

```text
0 = outside zone
1 = inside zone
```

Example:

```text
masks_npy/Patient013/20210921/Patient013_20210921_OS_FA_0001_zone_masks.npy
```

`fp_masked_zones/` contains paired FP images with each FA-derived zone mask applied. Each successful FP row has 10 PNGs:

```text
fp_masked_zones/Patient013/20210921/Patient013_20210921_OS_FA_0001/
  01_Zone_01_inner_upper_nasal.png
  02_Zone_02_inner_upper_temporal.png
  ...
  10_Zone_10_far_periphery.png
```

Pixels outside the selected zone are transparent.

`mask_labels_png/` contains optional debug/QC label images when `--no-label-png` is not used. Pixel values are zone IDs:

```text
0 = background/no zone
1..10 = zone number
```

`manifest.csv` records successful or skipped rows and links source images to generated outputs. Useful columns include:

- `row_index`
- `status`
- `fa_rel`, `fp_rel`
- `fa_path`, `fp_path`
- `mask_npy`
- `fp_zone_dir`
- `mask_source`
- `fallback_mask_path`

`failures.csv` records rows that still failed after detection and fallback. Useful columns include:

- `row_index`
- `fa_value`
- `error_type`
- `error`

After the fallback recovery pass, only a small set of rows remained unusable because they had neither detectable yellow overlay/crosshair geometry nor an existing sibling mask file.

## Preprocessing Commands

### 1) Create a single train/val/test split
```bash
python preprocessing/CSV_split_generator.py
```

### 2) Create 10-fold splits
```bash
python preprocessing/CSV_k_fold_generator.py
```

### 3) Extract FA zone masks from yellow overlays
```bash
python preprocessing/extract_fa_zone_masks.py \
  --input-glob "Dataset 01032025/FA_annotated/**/*.png" \
  --output-dir extracted_fa_zone_masks
```

The extractor uses the ImageJ overlay calibration by default: 53 px/mm, a
3.0 mm inner radius, a 16.0 mm outer radius, and a 3 px overlay stroke. The
overlay is interpreted as fovea-centered and rotated so the fovea-ONH axis is
the nasal/temporal meridian; the underlying FA image is not rotated.

Outputs are written per-image and include:
- `zone_01.png` ... `zone_10.png`
- `label_map.png`
- `qc_overlay.png`
- `geometry.json`

If a registered fundus image shares the same pixel grid as the FA image, the saved masks can be reused directly on the fundus image.

## Training Commands

### 1) Binary training (single split)
```bash
python training/train_OCT.py \
  --csvpath split_10 \
  --dataset_path "Dataset 01032025" \
  --output_path output_split_10 \
  --hflip \
  --unweighted
```

### 2) Binary eval using saved checkpoint
```bash
python training/train_OCT.py \
  --mode eval \
  --csvpath split_10 \
  --dataset_path "Dataset 01032025" \
  --output_path output_split_10 \
  --hflip \
  --unweighted \
  --checkpoint checkpoint.pt
```

### 3) K-fold training
```bash
python training/train_kFold.py \
  --csvpath fold_0 \
  --dataset_path "Dataset 01032025" \
  --output_path output_fold_0 \
  --hflip \
  --unweighted \
  --earlystop
```

### 4) K-fold final training + explainability export
```bash
python training/train_kFold.py \
  --final \
  --csvpath fold_0 \
  --dataset_path "Dataset 01032025" \
  --output_path output_fold_0 \
  --metadata_path <best_run_folder_name> \
  --hflip \
  --unweighted \
  --gradcam \
  --IG \
  --shap
```

### 5) Graded training
```bash
python training/train_OCT_graded.py \
  --csvpath split_10 \
  --dataset_path "Dataset 01032025" \
  --output_path output_graded_split_10 \
  --hflip \
  --unweighted
```

### 6) OCT2017 pretraining
```bash
python training/pretrain_backbone_OCT2017.py \
  --dataset_path /path/to/OCT2017 \
  --output_path pretraining_OCT2017_output \
  --hflip \
  --unweighted
```

## Evaluation Commands

### 1) Eval script
```bash
python evaluation/eval_OCT.py \
  --mode eval \
  --csvpath split_10 \
  --dataset_path "Dataset 01032025" \
  --output_path output_split_10
```

### 2) Latent eval script
```bash
python evaluation/eval_OCT_latent.py \
  --mode eval \
  --csvpath split_10 \
  --dataset_path "Dataset 01032025" \
  --output_path output_split_10
```

## Run Saved Command Bundles

```bash
bash scripts/run_commands.sh
bash scripts/run_commands_kfold.sh
bash scripts/run_commands_ext_test.sh
bash scripts/run_pretraining_commands.sh
```

## Notes
- Many scripts rely on relative paths and naming conventions; run from repo root.
- If dataset/split paths differ on your machine, always override:
  - `--dataset_path`
  - `--csvpath`
  - `--output_path`
