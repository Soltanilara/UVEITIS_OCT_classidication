# Experiment D with RETFound-DINOv2 and DINOv3 ViT-Large

This pipeline keeps the established Experiment D design:

- the full FA image is letterboxed together with its anatomical masks;
- Zone 10 is removed from the image;
- dense transformer patch tokens are pooled with soft attention inside Zones 1-9;
- the pooled zone feature is concatenated with the encoder class token;
- three separate MLPs classify Zones 1-4, Zones 5-8, and optic-nerve Zone 9;
- the encoder and Experiment D head are fine-tuned end to end with separate learning rates.

The five-fold launcher enables gradient checkpointing for both ViT-L encoders to reduce activation memory while retaining full end-to-end fine-tuning.

Full-resolution masks are validated lazily as samples enter a batch. This avoids an expensive pre-training scan of every mask over NAS storage. Add `--eager_mask_scan` only when a complete preflight mask audit is specifically required.

The two new encoders are:

| CLI value | Pretrained model | Input | Patch grid |
|---|---|---:|---:|
| `retfound_dinov2` | RETFound-DINOv2 ViT-L/14 (`YukunZhou/RETFound_dinov2_meh`) | 392 | 28x28 |
| `dinov3_vitl16` | DINOv3 ViT-L/16 LVD-1689M (`facebook/dinov3-vitl16-pretrain-lvd1689m`) | 384 | 24x24 |

DINOv3 has four register tokens. The adapter removes these explicitly before anatomical mask pooling, so only spatial patch tokens enter the zone attention module.

## Environment

Create and activate an isolated environment:

```bash
conda create -p .conda/fa-foundation-models python=3.11 pip -y
conda activate "$PWD/.conda/fa-foundation-models"
```

On a CUDA 12.6-compatible GPU server, install PyTorch and the remaining dependencies:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-fa-foundation-models.txt
```

For CPU-only validation, replace `cu126` with `cpu`. If the server uses a different CUDA driver/toolkit, select the matching command from the official PyTorch installer.

## Model access

Both pretrained repositories require accepting their model terms on Hugging Face. Accept access for:

- `YukunZhou/RETFound_dinov2_meh`
- `facebook/dinov3-vitl16-pretrain-lvd1689m`

Then authenticate without putting a token in a command or tracked file:

```bash
hf auth login
```

`HF_TOKEN` is also supported. For offline jobs, pre-download the models, pass a local checkpoint/model-directory path, and add `--hf-local-files-only` to the launcher.

## Five-fold Experiment D comparison

Inspect all ten commands without creating outputs:

```bash
python run_fa_anatomical_head_experiments.py \
  --dry-run \
  --experiments D \
  --backbones retfound_dinov2 dinov3_vitl16 \
  --python "$PWD/.conda/fa-foundation-models/bin/python" \
  --batch-size 2
```

Launch the ten runs on the configured idle-GPU pool:

```bash
python run_fa_anatomical_head_experiments.py \
  --experiments D \
  --backbones retfound_dinov2 dinov3_vitl16 \
  --python "$PWD/.conda/fa-foundation-models/bin/python" \
  --batch-size 2 \
  --resume
```

Outputs remain separate:

```text
fa_anatomical_head_experiments/
  backbone_retfound_dinov2/experiment_D_group_mlps/fold_0..4/
  backbone_dinov3_vitl16/experiment_D_group_mlps/fold_0..4/
```

ViT-L fine-tuning uses substantially more memory than the old ViT-B experiment. Start with batch size 2. If it does not fit, use batch size 1; if memory allows, increase it consistently for both encoders.

## Single-fold commands

RETFound-DINOv2:

```bash
python training/train_fa_dinov2_zone_attention.py \
  --csvpath fold_zone_masks_ready_patient_split/fold_0 \
  --output_path fa_anatomical_head_experiments/backbone_retfound_dinov2/experiment_D_group_mlps/fold_0 \
  --backbone retfound_dinov2 \
  --dinov2_arch dinov2_vitl14 \
  --retfound_dinov2_checkpoint RETFound_dinov2_meh \
  --image_size 392 \
  --head_variant group_mlps
```

DINOv3 ViT-Large:

```bash
python training/train_fa_dinov2_zone_attention.py \
  --csvpath fold_zone_masks_ready_patient_split/fold_0 \
  --output_path fa_anatomical_head_experiments/backbone_dinov3_vitl16/experiment_D_group_mlps/fold_0 \
  --backbone dinov3_vitl16 \
  --dinov3_model_id facebook/dinov3-vitl16-pretrain-lvd1689m \
  --image_size 384 \
  --head_variant group_mlps
```
