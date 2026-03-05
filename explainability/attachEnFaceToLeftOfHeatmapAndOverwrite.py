import os
from pathlib import Path
import pandas as pd
from PIL import Image

def attach_enface_left(heatmap_dir: str, csv_path: str, parent_dir: str) -> None:
    """
    For each image file in `heatmap_dir`, find the matching row in `csv_path`
    (match by basename of the 'Image File' column), load that "en-face" image
    from parent_dir / Image File, crop the top-left 496x496 (or smaller if needed),
    attach it to the LEFT of the heatmap image, then overwrite the original heatmap.
    """

    heatmap_dir = Path(heatmap_dir)
    parent_dir = Path(parent_dir)

    # 1) Read CSV and build a map: basename -> full en-face path
    df = pd.read_csv(csv_path)
    if "Image File" not in df.columns:
        raise ValueError("CSV must include an 'Image File' column.")

    # Normalize paths and build lookup by basename
    mapping = {}
    duplicates = set()
    for rel_path in df["Image File"].astype(str):
        basename = os.path.basename(rel_path)
        full_path = parent_dir / rel_path
        if basename in mapping:
            # Track duplicates; keep the first occurrence
            duplicates.add(basename)
        else:
            mapping[basename] = full_path

    if duplicates:
        print(f"Warning: duplicate basenames found in CSV (using first occurrence): {sorted(list(duplicates))[:5]}{' ...' if len(duplicates) > 5 else ''}")

    # 2) Process every image in heatmap_dir
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    heatmap_files = [p for p in heatmap_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    for heatmap_path in heatmap_files:
        basename = heatmap_path.name

        if basename not in mapping:
            print(f"[SKIP] No CSV match for heatmap '{basename}'.")
            continue

        enface_path = mapping[basename]
        if not enface_path.exists():
            print(f"[SKIP] En-face image not found on disk: {enface_path}")
            continue

        try:
            # Load images (convert to RGB to avoid mode issues)
            heatmap_img = Image.open(heatmap_path).convert("RGB")
            enface_img = Image.open(enface_path).convert("RGB")

            # 3) Crop top-left 496x496 (cap by actual size if smaller)
            crop_w = min(496, enface_img.width)
            crop_h = min(496, enface_img.height)
            crop = enface_img.crop((0, 0, crop_w, crop_h))

            # 4) Create a new canvas; don't resize either image—pad as needed
            final_h = max(heatmap_img.height, crop_h)
            final_w = crop_w + heatmap_img.width
            canvas = Image.new("RGB", (final_w, final_h), color=(0, 0, 0))

            # Paste the 496x496 (or smaller) crop at the left/top
            canvas.paste(crop, (0, 0))
            # Paste the heatmap to the right/top
            canvas.paste(heatmap_img, (crop_w, 0))

            # 5) Overwrite the original heatmap file
            canvas.save(heatmap_path)
            # print(f"[OK] Updated: {heatmap_path}")

        except Exception as e:
            print(f"[ERROR] Failed '{heatmap_path.name}': {e}")


# -----------------------------
# Example usage (uncomment and edit paths):
heatmap_dir = "output_ext_test/Threshold_Outputs_SHAP_Modified_Factor2/maxF1/FP"
csv_path = "ext_test/test.csv"
parent_dir = "From New Uveitis Code/Test02082025 (Before Crop and Resize)"
attach_enface_left(heatmap_dir, csv_path, parent_dir)
