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
