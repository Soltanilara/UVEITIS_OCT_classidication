# Repository Structure

## `training/`
- Model training and pretraining entrypoints.
- Includes binary, graded, k-fold, and AE variants.

## `evaluation/`
- Evaluation-focused scripts (including latent feature export/eval).

## `preprocessing/`
- Dataset split generation, cropping, image-quality preprocessing, and label/weight prep.

## `explainability/`
- Grad-CAM/IG/SHAP generation, overlays, and meta-score map creation.

## `analysis/`
- Post-training analysis scripts (ROC/PR plots, bootstrap CI, latent visualization, correlation studies).

## `scripts/`
- Command collections for running experiments/hyperparameter sweeps.

