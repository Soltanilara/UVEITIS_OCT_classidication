"""
Visualise per-patient SHAP/IG/Grad-CAM *signed* scores that were saved as
      {index}.npy   ( index == row-number in test.csv )

For every unique folder   Patient###/########/##/##/
( e.g.  Patient089/20250402/OD/FO/ )
the script

1. loads the corresponding score arrays in **row order**;
2. ReLU’s each array ➜ sums over channels ➜ background-fills ROI;
3. collapses the height dimension to a 1 × 512 profile;
4. repeats that row so the stack of all patient images re-forms a 512 × 512 map;
5. flips the map vertically, scales it to 0–255, and saves it as a PNG in
   “Shap Score Region Visualizations/Patient089_20250402_OD_FO.png”, … .

Author: **your-name**
---------------------------------------------------------------------------
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ------------------------------------------------------------------
#  USER SETTINGS ----------------------------------------------------
# ------------------------------------------------------------------
TEST_CSV      = "ext_test/test.csv"                       # path to test.csv
SCORE_DIR     = "output_ext_test/final_finetune_resnet50_pretraining_swav_CE_loss_unweighted_batch_64_lr_0.001_decay_1e-05_10_10_1_1e-05_epochs_100_hflip_seed_0/SHAP_Positive_Scores"           # folder that holds 0.npy, 1.npy, …
OUT_DIR       = "Explainability Score Region Visualizations/SHAP"
CHANNEL_FIRST = True                             # True if array is (3,496,512)

# ------------------------------------------------------------------
#  PREP -------------------------------------------------------------
# ------------------------------------------------------------------
df = pd.read_csv(TEST_CSV)
out_path = Path(OUT_DIR)
out_path.mkdir(parents=True, exist_ok=True)

# Build mapping  group_key  ➜  [row_indices]
groups = {}
for idx, img_rel_path in enumerate(df["Image File"]):
    # group_key =  Patient089/20250402/OD/FO/
    parts = img_rel_path.split("/")[:4]
    group_key = "/".join(parts) + "/"            # keep the trailing “/”
    groups.setdefault(group_key, []).append(idx)

# ------------------------------------------------------------------
#  MAIN LOOP --------------------------------------------------------
# ------------------------------------------------------------------
for group_key, idx_list in groups.items():
    idx_list.sort()                              # ensure row order
    n_imgs        = len(idx_list)
    rows_per_img  = 512 // n_imgs                # integer division
    processed_rows = []

    for i in idx_list:
        f_score = Path(SCORE_DIR) / f"{i}.npy"
        if not f_score.is_file():
            raise FileNotFoundError(f"Score file missing: {f_score}")

        score = np.load(f_score)                 # (3,496,512)

        # --- step 1: ReLU & collapse channels ---------------------
        if CHANNEL_FIRST:                        # (3,H,W) → (H,W)
            score = np.maximum(score, 0).sum(axis=0)
        else:                                    # (H,W,3)
            score = np.maximum(score, 0).sum(axis=-1)

        # --- step 2: background fill ------------------------------
        bg_mean = score[420:, 60:].mean()
        score[420:, :60] = bg_mean

        # --- step 3: collapse height → (512,) ---------------------
        row_profile = score.sum(axis=0)          # (512,)

        # --- step 4: tile vertically ------------------------------
        tiled = np.tile(row_profile, (rows_per_img, 1))  # (rows_per_img,512)
        processed_rows.append(tiled)

    # --- assemble full 512 × 512 map ------------------------------
    patient_map = np.vstack(processed_rows)      # should be (<=512,512)

    # # If integer division left a gap, repeat the last row to fill to 512
    # missing = 512 - patient_map.shape[0]
    # if missing > 0:
    #     patient_map = np.vstack(
    #         [patient_map, np.tile(patient_map[-1:], (missing, 1))]
    #     )

    # --- step 5: flip vertically & rescale to 0–255 ---------------
    patient_map = np.flipud(patient_map)
    vmax = patient_map.max()
    if vmax > 0:
        patient_img = (patient_map / vmax * 255).astype(np.uint8)
    else:                                         # all-zero map
        patient_img = patient_map.astype(np.uint8)

    # --- save ------------------------------------------------------
    # e.g.  Patient089_20250402_OD_FO.png
    save_name = group_key.rstrip("/").replace("/", "_") + ".png"
    Image.fromarray(patient_img, mode="L").save(out_path / save_name)

    print(f"✓ saved {out_path / save_name}")

print("\nAll patient visualisations written to:", out_path.resolve())
