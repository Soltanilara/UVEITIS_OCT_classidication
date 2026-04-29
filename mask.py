"""
preprocessing/mask.py 

Example Usage
    "python3 mask.py --image_dir /mnt/NAS/Tim/Datasets/Sample_02_25_26_OD --csv_path /home/tim/UVEITIS_OCT_classidication/fold_0/train.csv"

"""
import cv2
import numpy as np
import pandas as pd
import argparse 
import unicodedata
import json

from pathlib import Path 

from preprocessing.extract_fa_zone_masks import (
    build_label_map,
    build_retina_mask,
    build_zone_masks,
    choose_concentric_pair,
    detect_axes,
    detect_disc_circle,
    detect_yellow_overlay,
    hough_circle_candidates,
    load_rgb,
    orient_axes,
    refine_radii,
    Geometry,
)


def get_centroid(contour):
    """
    Computes the centroid (center of mass) of a contour using image moments.
    
    Moments are weighted sums of pixel positions:
        m00 = total area (pixel count)
        m10 = sum of all x coordinates
        m01 = sum of all y coordinates
    
    Centroid formula:
        cx = m10 / m00  (mean x position)
        cy = m01 / m00  (mean y position)

    Returns (cx, cy) as ints, or (None, None) if contour has zero area.
    """
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    

def label_contours_for_ten(contours, H, W): 
    labeled_mask = np.zeros((H, W), dtype=np.uint8)


    cv2.drawContours(labeled_mask, [contours[0]], -1, 255, thickness=cv2.FILLED)
    labeled_mask = np.where(labeled_mask == 0, 10, 0).astype(np.uint8) #Label the inversion (Zone 10) of largest contour outer circle

    #Next four largest: Zones 5-8
    #Sort by centroid X and Y coordinate
    quad_contours = contours[1:5] 
    centroids = [(get_centroid(c), c) for c in quad_contours]
    centroids_sorted_by_y = sorted(centroids, key=lambda x: x[0][1])
    upper_two = sorted(centroids_sorted_by_y[:2], key=lambda x: x[0][0])
    lower_two = sorted(centroids_sorted_by_y[2:], key=lambda x: x[0][0])

    for zone_num, ((_cx, _cy), contour) in zip([8, 7], lower_two):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)
    for zone_num, ((_cx, _cy), contour) in zip([5, 6], upper_two):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)

    #5 Contours: Zone 9 (Optic Disk/ Circle) + 4 Inner Quadrants 
    small_contours = contours[5:10]        
    small_centroids = [(get_centroid(c), c) for c in small_contours]
    small_sorted_by_x = sorted(small_centroids, key=lambda x: x[0][0])
    
    leftmost = small_sorted_by_x[0] #Leftmost is optic disk 
    remaining_four = small_sorted_by_x[1:]  
    (_cx, _cy), contour = leftmost               
    cv2.drawContours(labeled_mask, [contour], -1, 9, thickness=cv2.FILLED)

    remaining_sorted_by_y = sorted(remaining_four, key=lambda x: x[0][1])
    upper_two_small = sorted(remaining_sorted_by_y[:2], key=lambda x: x[0][0])
    lower_two_small = sorted(remaining_sorted_by_y[2:], key=lambda x: x[0][0])

    for zone_num, ((_cx, _cy), contour) in zip([3, 4], lower_two_small):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)
    for zone_num, ((_cx, _cy), contour) in zip([1, 2], upper_two_small):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)

    return labeled_mask
    

def label_contours_for_eleven(contours, H, W): 
    labeled_mask = np.zeros((H, W), dtype=np.uint8)

    cv2.drawContours(labeled_mask, [contours[0]], -1, 255, thickness=cv2.FILLED) #Placeholder 255 value
    labeled_mask = np.where(labeled_mask == 0, 10, 0).astype(np.uint8)

    quad_contours = contours[1:5] # 
    centroids = [(get_centroid(c), c) for c in quad_contours]
    centroids_sorted_by_y = sorted(centroids, key=lambda x: x[0][1])
    upper_two = sorted(centroids_sorted_by_y[:2], key=lambda x: x[0][0])
    lower_two = sorted(centroids_sorted_by_y[2:], key=lambda x: x[0][0])

    for zone_num, ((_cx, _cy), contour) in zip([8, 7], lower_two):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)
    for zone_num, ((_cx, _cy), contour) in zip([5, 6], upper_two):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)

    small_contours = contours[5:11]
    small_centroids = [(get_centroid(c), c) for c in small_contours]
    small_sorted_by_x = sorted(small_centroids, key=lambda x: x[0][0])
    leftmost_two = small_sorted_by_x[:2]
    remaining_four = small_sorted_by_x[2:]

    for (_cx, _cy), contour in leftmost_two: #Optic disk split into 2 due to quadrant line 
        cv2.drawContours(labeled_mask, [contour], -1, 9, thickness=cv2.FILLED)

    remaining_sorted_by_y = sorted(remaining_four, key=lambda x: x[0][1])
    upper_two_small = sorted(remaining_sorted_by_y[:2], key=lambda x: x[0][0])
    lower_two_small = sorted(remaining_sorted_by_y[2:], key=lambda x: x[0][0])

    for zone_num, ((_cx, _cy), contour) in zip([3, 4], lower_two_small):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)
    for zone_num, ((_cx, _cy), contour) in zip([1, 2], upper_two_small):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)
        
    return labeled_mask


def label_contours_generalized(contours, H, W):
    """
    Handle near-miss contour sets where the annotated overlay is present but the
    optic-disc contour is missing or split into extra pieces.
    """
    if len(contours) < 9:
        raise RuntimeError(f"found {len(contours)} contours, expected at least 9")

    labeled_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(labeled_mask, [contours[0]], -1, 255, thickness=cv2.FILLED)
    labeled_mask = np.where(labeled_mask == 0, 10, 0).astype(np.uint8)

    outer_contours = contours[1:5]
    if len(outer_contours) < 4:
        raise RuntimeError("not enough outer quadrant contours")

    outer_centroids = [(get_centroid(c), c) for c in outer_contours]
    outer_sorted_by_y = sorted(outer_centroids, key=lambda x: x[0][1])
    upper_two = sorted(outer_sorted_by_y[:2], key=lambda x: x[0][0])
    lower_two = sorted(outer_sorted_by_y[2:4], key=lambda x: x[0][0])

    for zone_num, ((_cx, _cy), contour) in zip([8, 7], lower_two):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)
    for zone_num, ((_cx, _cy), contour) in zip([5, 6], upper_two):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)

    inner_candidates = contours[5:]
    if len(inner_candidates) < 4:
        raise RuntimeError("not enough inner contours")
    inner_candidates = sorted(inner_candidates, key=cv2.contourArea, reverse=True)
    inner_contours = inner_candidates[:4]

    inner_centroids = [(get_centroid(c), c) for c in inner_contours]
    inner_sorted_by_y = sorted(inner_centroids, key=lambda x: x[0][1])
    upper_two_small = sorted(inner_sorted_by_y[:2], key=lambda x: x[0][0])
    lower_two_small = sorted(inner_sorted_by_y[2:4], key=lambda x: x[0][0])

    for zone_num, ((_cx, _cy), contour) in zip([3, 4], lower_two_small):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)
    for zone_num, ((_cx, _cy), contour) in zip([1, 2], upper_two_small):
        cv2.drawContours(labeled_mask, [contour], -1, zone_num, thickness=cv2.FILLED)

    # Synthesize zone 9 when the disc contour is absent or fragmented.
    (center_x, center_y), outer_radius = cv2.minEnclosingCircle(contours[0])
    inner_radii = [cv2.minEnclosingCircle(c)[1] for c in inner_contours]
    disc_radius = float(np.median(inner_radii))
    inner_xs = [centroid[0][0] for centroid in inner_centroids]
    disc_cx = float(min(inner_xs) - 0.9 * disc_radius)
    disc_cy = float(center_y)
    cv2.circle(labeled_mask, (int(round(disc_cx)), int(round(disc_cy))), int(round(disc_radius)), 9, thickness=cv2.FILLED)

    return labeled_mask

def load_image_rgb(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_filename_token(text):
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def resolve_image_path(image_dir, relative_path):
    rel_path = Path(str(relative_path).replace("\\", "/"))
    full_path = Path(image_dir) / rel_path
    if full_path.exists():
        return full_path

    candidate_dir = full_path.parent
    stem = rel_path.stem
    preferred_exts = [rel_path.suffix, ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

    for ext in preferred_exts:
        if not ext:
            continue
        candidate = candidate_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    if candidate_dir.exists():
        normalized_name = normalize_filename_token(rel_path.name)
        normalized_stem = normalize_filename_token(stem)
        for candidate in candidate_dir.iterdir():
            if not candidate.is_file():
                continue
            candidate_name = normalize_filename_token(candidate.name)
            candidate_stem = normalize_filename_token(candidate.stem)
            if candidate_name == normalized_name or candidate_stem == normalized_stem:
                return candidate

    return full_path


def detect_contours(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    kernel = np.ones((3, 3), np.uint8)
    yellow = cv2.dilate(yellow, kernel, iterations=1)

    contours, _ = cv2.findContours(yellow, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 500]
    return sorted(contours, key=cv2.contourArea, reverse=True)


def create_zone_masks_from_contours(image):
    H, W = image.shape[:2]
    contours = detect_contours(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if len(contours) == 11: 
        labeled_mask = label_contours_for_eleven(contours, H, W) 
    elif len(contours) == 10: 
        labeled_mask = label_contours_for_ten(contours, H, W)
    elif len(contours) >= 9:
        labeled_mask = label_contours_generalized(contours, H, W)
    else:
        raise RuntimeError(f"found {len(contours)} contours, expected 10/11 or recoverable 9+")

    return labeled_mask, len(contours)


def create_zone_masks_from_geometry(image_path):
    path = Path(image_path)
    rgb = load_rgb(path)
    retina_mask = build_retina_mask(rgb, threshold=8)
    yellow_mask = detect_yellow_overlay(rgb, sat_threshold=30, val_threshold=80)

    circles = hough_circle_candidates(yellow_mask)
    outer_circle, inner_circle = choose_concentric_pair(circles, rgb.shape[:2])
    center_xy = (
        float(outer_circle[0] + inner_circle[0]) / 2.0,
        float(outer_circle[1] + inner_circle[1]) / 2.0,
    )
    inner_radius, outer_radius = refine_radii(
        yellow_mask,
        center_xy=center_xy,
        outer_radius_guess=float(outer_circle[2]),
        inner_radius_guess=float(inner_circle[2]),
    )
    disc_center_xy, disc_radius = detect_disc_circle(
        yellow_mask,
        center_xy=center_xy,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )
    axis_a, axis_b = detect_axes(
        yellow_mask,
        center_xy=center_xy,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )
    disc_axis, vertical_axis, eye = orient_axes(disc_center_xy, center_xy, axis_a, axis_b)

    geometry = Geometry(
        center_xy=center_xy,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        disc_center_xy=disc_center_xy,
        disc_radius=disc_radius,
        eye=eye,
        disc_axis_xy=(float(disc_axis[0]), float(disc_axis[1])),
        vertical_axis_xy=(float(vertical_axis[0]), float(vertical_axis[1])),
    )
    zone_masks = build_zone_masks(rgb.shape[:2], retina_mask, geometry)
    return build_label_map(zone_masks)


def create_zone_masks(image_path):
    image = load_image_rgb(image_path)
    contour_error = None
    contour_count = None

    try:
        labeled_mask, contour_count = create_zone_masks_from_contours(image)
        method = f"contours_{contour_count}"
    except Exception as exc:
        contour_error = str(exc)
        labeled_mask = create_zone_masks_from_geometry(image_path)
        method = "geometry_fallback"

    save_path = str(Path(image_path).with_suffix("")) + "_masks.npy"
    np.save(save_path, labeled_mask)
    return {
        "save_path": save_path,
        "method": method,
        "contour_count": contour_count,
        "contour_error": contour_error,
    }


def run_mask_extraction(image_dir, paths, log_path=None, summary_path=None):
    ok, skip, err = 0, 0, 0
    method_counts = {}
    log_records = []
    log_handle = None

    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "w", encoding="utf-8")

    try:
        for i, img_path in enumerate(paths):
            full_path = resolve_image_path(image_dir, img_path)
            try:
                result = create_zone_masks(str(full_path))
                if result:
                    ok += 1
                    method = result["method"]
                    method_counts[method] = method_counts.get(method, 0) + 1
                    extra = ""
                    if result["contour_error"]:
                        extra = f" | contour_fallback_reason={result['contour_error']}"
                    if str(full_path).replace("\\", "/") != str(Path(image_dir) / str(img_path).replace("\\", "/")):
                        extra += f" | resolved_path={full_path.name}"
                    print(f"[{i+1}/{len(paths)}] OK: {Path(img_path).name} | method={method}{extra}", flush=True)
                    record = (
                        {
                            "input_path": str(img_path),
                            "resolved_path": str(full_path),
                            "status": "ok",
                            "method": method,
                            "save_path": result["save_path"],
                            "contour_count": result["contour_count"],
                            "contour_error": result["contour_error"],
                        }
                    )
                    log_records.append(record)
                else:
                    skip += 1
                    record = (
                        {
                            "input_path": str(img_path),
                            "resolved_path": str(full_path),
                            "status": "skipped",
                            "method": None,
                            "save_path": None,
                            "contour_count": None,
                            "contour_error": None,
                        }
                    )
                    log_records.append(record)
            except Exception as e:
                err += 1
                print(f"[{i+1}/{len(paths)}] ERROR: {Path(img_path).name} — {e}", flush=True)
                record = (
                    {
                        "input_path": str(img_path),
                        "resolved_path": str(full_path),
                        "status": "error",
                        "method": None,
                        "save_path": None,
                        "contour_count": None,
                        "contour_error": None,
                        "error": str(e),
                    }
                )
                log_records.append(record)

            if log_handle is not None:
                log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_handle.flush()
    finally:
        if log_handle is not None:
            log_handle.close()

    print(f"\n[Summary] Done — {ok} saved, {skip} skipped, {err} errors", flush=True)
    if method_counts:
        print(f"[Summary] Methods: {method_counts}", flush=True)
    summary = {
        "total": len(paths),
        "ok": ok,
        "skipped": skip,
        "errors": err,
        "method_counts": method_counts,
    }
    if summary_path:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def create_masks_from_csv(image_dir, csv_path, log_path=None, summary_path=None): 
    df = pd.read_csv(csv_path)
    
    df["Image_File(FA)"] = df["Image_File(FA)"].str.replace("\\", "/", regex=False)
    paths = df["Image_File(FA)"].dropna().unique().tolist()
    
    print(f"[count] Processing {len(paths)} unique images...")
    return run_mask_extraction(image_dir, paths, log_path=log_path, summary_path=summary_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute zone masks from a CSV of image paths")
    parser.add_argument("--image_dir", type=str, required=True, help="Root directory containing images")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file with Image_File(FA) column")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for per-image extraction logs")
    parser.add_argument("--summary_path", type=str, default=None, help="Optional JSON path for aggregate extraction summary")
    args = parser.parse_args()

    create_masks_from_csv(args.image_dir, args.csv_path, log_path=args.log_path, summary_path=args.summary_path)
