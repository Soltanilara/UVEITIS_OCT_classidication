"""
Retinal Zone Slicer – Auto Fovea + Auto Crosshair Angle + Clean Outputs
========================================================================

This version:
  1. Detects the fovea automatically from the yellow crosshair intersection.
  2. Detects the crosshair rotation angle automatically.
  3. Slices the UWFFA image into 10 zones.
  4. Removes the yellow overlay/circles/crosshair from the saved zone outputs.

Requirements:
  pip install pillow numpy opencv-python

Example:
  python slice_zones_auto_fovea_clean_overlay_FINAL.py --image "Patient010_20230613_OD_FA_0001.png"

Manual override still works:
  python slice_zones_auto_fovea_clean_overlay_FINAL.py --image "scan.png" --cx 1008 --cy 981 --angle_deg 5.0

Optional full-size outputs:
  python slice_zones_auto_fovea_clean_overlay_FINAL.py --image "scan.png" --save_full_size
"""

import argparse
import math
import os
import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except Exception as exc:
    raise ImportError(
        "OpenCV (cv2) failed to import.\n\n"
        "Install it with:\n"
        "  pip install opencv-python\n"
    ) from exc

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_IMAGE   = r"C:\Users\Mohammad\Downloads\Patient095_20251014_OD_FA_0001.png"
DEFAULT_CX      = None      # None = auto-detect from yellow crosshair
DEFAULT_CY      = None      # None = auto-detect from yellow crosshair
PX_PER_MM       = 53
INNER_R_MM      = 3.0
OUTER_R_MM      = 16.0
ANGLE_DEG       = None      # None = auto-detect from yellow crosshair

# Optic disc ellipse offset FROM fovea (pixels) and semi-axes.
# Tune these if needed for your dataset.
ONH_OFFSET_X    = 270
ONH_OFFSET_Y    = 0
ONH_RX          = 80
ONH_RY          = 95
OUTPUT_DIR      = "zones_output"
# ---------------------------------------------------------------------------


def resolve_path(image_path):
    """Resolve relative image paths against the current folder and script folder."""
    if os.path.isabs(image_path):
        return image_path

    if os.path.exists(image_path):
        return image_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, image_path)
    if os.path.exists(candidate):
        return candidate

    return image_path


def make_yellow_mask(arr):
    """
    Detect yellow overlay pixels.
    Works for dim/anti-aliased yellow-green lines on UWFFA images.
    """
    rgb = arr[:, :, :3].astype(np.int16)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # Yellow overlay usually has R and G both high relative to B.
    # This threshold is intentionally tolerant because the overlay may be faint.
    yellow = (
        (r > 40) &
        (g > 40) &
        (b < 100) &
        (np.abs(r - g) < 85) &
        ((r - b) > 20) &
        ((g - b) > 20)
    )
    return yellow


def remove_yellow_overlay(arr, inpaint_radius=3, dilate_iterations=1):
    """
    Remove yellow segmentation overlay from an RGBA image array.

    Method:
      - Detect yellow pixels.
      - Slightly dilate the mask so anti-aliased edges are also removed.
      - Use OpenCV inpainting to fill those pixels from surrounding angiography.

    Returns:
      cleaned RGBA array
    """
    yellow = make_yellow_mask(arr)

    if int(yellow.sum()) == 0:
        return arr.copy()

    mask = (yellow.astype(np.uint8) * 255)

    if dilate_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)

    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3].copy()

    cleaned_rgb = cv2.inpaint(
        rgb,
        mask,
        inpaintRadius=inpaint_radius,
        flags=cv2.INPAINT_TELEA,
    )

    cleaned = arr.copy()
    cleaned[:, :, :3] = cleaned_rgb
    cleaned[:, :, 3] = alpha
    return cleaned


def line_intersection(line1, line2):
    """Return intersection point of two line segments extended as infinite lines."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return float(px), float(py)


def normalize_angle_deg(angle):
    """Normalize angle to approximately [-90, 90)."""
    while angle >= 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return angle


def detect_crosshair_from_yellow(arr, output_dir=None, save_debug=True):
    """
    Detect fovea center and crosshair angle from the yellow crosshair.

    Important:
      This function must run BEFORE yellow overlay removal.

    Returns:
      cx, cy, angle_deg, yellow_pixel_count
    """
    H, W = arr.shape[:2]
    yellow = make_yellow_mask(arr)
    yellow_count = int(yellow.sum())

    if yellow_count < 200:
        raise ValueError(
            "Could not find enough yellow overlay pixels. Make sure the image "
            "contains yellow circles/crosshair, or provide --cx and --cy manually."
        )

    mask_u8 = (yellow.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)

    # Hough line detection.
    min_len = max(80, int(min(H, W) * 0.08))
    lines = cv2.HoughLinesP(
        mask_u8,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=min_len,
        maxLineGap=35,
    )

    if lines is None or len(lines) < 2:
        raise ValueError("Yellow overlay found, but crosshair lines could not be detected.")

    horizontal = []
    vertical = []

    for item in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, item)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue

        angle = normalize_angle_deg(math.degrees(math.atan2(dy, dx)))

        # Horizontal-ish line is near 0 degrees.
        # Vertical-ish line is near +/-90 degrees.
        if abs(angle) <= 25:
            horizontal.append((length, angle, (x1, y1, x2, y2)))
        elif abs(abs(angle) - 90) <= 25:
            vertical.append((length, angle, (x1, y1, x2, y2)))

    if not horizontal or not vertical:
        raise ValueError("Could not separate horizontal and vertical yellow crosshair lines.")

    # Use the longest detected horizontal and vertical line segments.
    h_len, h_angle, h_line = max(horizontal, key=lambda x: x[0])
    v_len, v_angle, v_line = max(vertical, key=lambda x: x[0])

    point = line_intersection(h_line, v_line)
    if point is None:
        raise ValueError("Detected crosshair lines are parallel; cannot estimate fovea.")

    cx, cy = point
    if not (0 <= cx < W and 0 <= cy < H):
        raise ValueError(f"Detected fovea is outside image bounds: ({cx:.1f}, {cy:.1f})")

    # Use the horizontal crosshair angle as the zone rotation angle.
    angle_deg = h_angle

    if save_debug and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        debug_img = Image.fromarray(arr).convert("RGBA")
        draw = ImageDraw.Draw(debug_img)
        draw.line(h_line, fill=(255, 0, 0, 255), width=4)
        draw.line(v_line, fill=(0, 255, 0, 255), width=4)
        r = 18
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 0, 0, 255), width=5)
        draw.text(
            (cx + 25, cy + 25),
            f"fovea=({cx:.0f},{cy:.0f}), angle={angle_deg:.2f} deg",
            fill=(255, 0, 0, 255),
        )
        debug_img.save(os.path.join(output_dir, "debug_detected_fovea.png"))

    return int(round(cx)), int(round(cy)), float(angle_deg), yellow_count


def make_masks(width, height, cx, cy, r_inner, r_outer,
               angle_deg, onh_offset_x, onh_offset_y, onh_rx, onh_ry):
    """Return boolean masks for Zones 1 through 10."""
    yy, xx = np.ogrid[:height, :width]
    dx = xx - cx
    dy = yy - cy

    dist = np.sqrt(dx**2 + dy**2)

    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    # Rotate by -angle so the crosshair axes become mathematical x/y axes.
    rx = dx * cos_a + dy * sin_a
    ry = -dx * sin_a + dy * cos_a

    upper = ry <= 0
    lower = ry > 0
    nasal = rx <= 0
    temporal = rx > 0

    in_inner = dist <= r_inner
    in_ring = (dist > r_inner) & (dist <= r_outer)
    outside = dist > r_outer

    onh_cx = cx + onh_offset_x
    onh_cy = cy + onh_offset_y
    dx_onh = xx - onh_cx
    dy_onh = yy - onh_cy
    in_onh = (dx_onh**2 / onh_rx**2 + dy_onh**2 / onh_ry**2) <= 1.0

    masks = {
        "Zone_01_inner_upper_nasal":    in_inner & upper & nasal,
        "Zone_02_inner_upper_temporal": in_inner & upper & temporal,
        "Zone_03_inner_lower_temporal": in_inner & lower & temporal,
        "Zone_04_inner_lower_nasal":    in_inner & lower & nasal,
        "Zone_05_ring_upper_nasal":     in_ring & upper & nasal,
        "Zone_06_ring_upper_temporal":  in_ring & upper & temporal,
        "Zone_07_ring_lower_temporal":  in_ring & lower & temporal,
        "Zone_08_ring_lower_nasal":     in_ring & lower & nasal,
        "Zone_09_optic_disc":           in_onh,
        "Zone_10_far_periphery":        outside,
    }
    return masks


def crop_to_content(img_array):
    """Crop an RGBA array to its non-transparent bounding box."""
    alpha = img_array[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any() or not cols.any():
        return img_array

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img_array[rmin:rmax + 1, cmin:cmax + 1]


def save_detection_summary(output_dir, image_path, W, H, cx, cy, angle_deg,
                           yellow_count, r_inner, r_outer, overlay_removed):
    """Save a simple text summary for QC."""
    summary_path = os.path.join(output_dir, "detection_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Retinal Zone Slicer Detection Summary\n")
        f.write("====================================\n")
        f.write(f"Image path       : {image_path}\n")
        f.write(f"Image size       : {W} x {H} px\n")
        f.write(f"Fovea center     : ({cx}, {cy})\n")
        f.write(f"Rotation angle   : {angle_deg:.2f} degrees\n")
        f.write(f"Yellow pixels    : {yellow_count:,}\n")
        f.write(f"Inner radius px  : {r_inner:.1f}\n")
        f.write(f"Outer radius px  : {r_outer:.1f}\n")
        f.write(f"Overlay removed  : {overlay_removed}\n")


def slice_zones(image_path, cx, cy, px_per_mm,
                inner_r_mm, outer_r_mm, angle_deg,
                onh_offset_x, onh_offset_y, onh_rx, onh_ry,
                output_dir, save_full_size=False, keep_yellow_overlay=False,
                save_clean_full_image=True):

    os.makedirs(output_dir, exist_ok=True)
    image_path = resolve_path(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    img = Image.open(image_path).convert("RGBA")
    original_arr = np.array(img)
    H, W = original_arr.shape[:2]

    yellow_count = 0

    # Step 1: Detect fovea and angle from ORIGINAL image before cleaning yellow overlay.
    if cx is None or cy is None or angle_deg is None:
        detected_cx, detected_cy, detected_angle, yellow_count = detect_crosshair_from_yellow(
            original_arr,
            output_dir=output_dir,
            save_debug=True,
        )
        print(f"Yellow pixels   : {yellow_count:,}")
        print("Fovea detection : yellow crosshair intersection")

        if cx is None:
            cx = detected_cx
        if cy is None:
            cy = detected_cy
        if angle_deg is None:
            angle_deg = detected_angle
    else:
        yellow_count = int(make_yellow_mask(original_arr).sum())
        print("Fovea detection : manual coordinates/angle")

    # Step 2: Clean image for OUTPUTS only.
    # This removes yellow circles/crosshair from all zone images.
    if keep_yellow_overlay:
        arr_for_output = original_arr.copy()
        overlay_removed = False
    else:
        arr_for_output = remove_yellow_overlay(
            original_arr,
            inpaint_radius=3,
            dilate_iterations=1,
        )
        overlay_removed = True

    if save_clean_full_image:
        clean_full_path = os.path.join(output_dir, "clean_full_image_no_yellow_overlay.png")
        Image.fromarray(arr_for_output, mode="RGBA").save(clean_full_path)

    r_inner = inner_r_mm * px_per_mm
    r_outer = outer_r_mm * px_per_mm

    print(f"Image size      : {W} x {H} px")
    print(f"Fovea centre    : ({cx}, {cy})")
    print(f"Inner radius    : {r_inner:.1f} px  ({inner_r_mm} mm)")
    print(f"Outer radius    : {r_outer:.1f} px  ({outer_r_mm} mm)")
    print(f"Rotation        : {angle_deg:.2f} degrees")
    print(f"Overlay removed : {overlay_removed}")
    print(f"Output folder   : {output_dir}/\n")

    save_detection_summary(
        output_dir=output_dir,
        image_path=image_path,
        W=W,
        H=H,
        cx=cx,
        cy=cy,
        angle_deg=angle_deg,
        yellow_count=yellow_count,
        r_inner=r_inner,
        r_outer=r_outer,
        overlay_removed=overlay_removed,
    )

    masks = make_masks(
        W, H, cx, cy, r_inner, r_outer,
        angle_deg, onh_offset_x, onh_offset_y, onh_rx, onh_ry,
    )

    # Step 3: Slice zones from the CLEANED image.
    for zone_name, mask in masks.items():
        zone_arr = arr_for_output.copy()
        zone_arr[~mask, 3] = 0

        if not save_full_size:
            zone_arr = crop_to_content(zone_arr)

        out_img = Image.fromarray(zone_arr, mode="RGBA")
        out_path = os.path.join(output_dir, f"{zone_name}.png")
        out_img.save(out_path)
        print(f"  Saved {zone_name}.png  ({int(mask.sum()):,} pixels)")

    print(f"\nDone! {len(masks)} zone images saved to '{output_dir}/'")
    print("Yellow overlay was removed from the saved output images.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Slice retinal UWFFA image into 10 zones with auto fovea detection and clean outputs."
    )
    p.add_argument("--image", default=DEFAULT_IMAGE, help="Path to input image")
    p.add_argument("--cx", type=int, default=DEFAULT_CX, help="Manual fovea x-center. Omit for auto-detection.")
    p.add_argument("--cy", type=int, default=DEFAULT_CY, help="Manual fovea y-center. Omit for auto-detection.")
    p.add_argument("--px_per_mm", type=float, default=PX_PER_MM, help="Pixels per mm")
    p.add_argument("--inner_r_mm", type=float, default=INNER_R_MM, help="Inner circle radius in mm")
    p.add_argument("--outer_r_mm", type=float, default=OUTER_R_MM, help="Outer circle radius in mm")
    p.add_argument("--angle_deg", type=float, default=ANGLE_DEG, help="Manual crosshair rotation angle. Omit for auto-detection.")
    p.add_argument("--onh_offset_x", type=int, default=ONH_OFFSET_X, help="ONH x-offset from fovea in pixels")
    p.add_argument("--onh_offset_y", type=int, default=ONH_OFFSET_Y, help="ONH y-offset from fovea in pixels")
    p.add_argument("--onh_rx", type=int, default=ONH_RX, help="ONH ellipse horizontal semi-axis in pixels")
    p.add_argument("--onh_ry", type=int, default=ONH_RY, help="ONH ellipse vertical semi-axis in pixels")
    p.add_argument("--output_dir", default=OUTPUT_DIR, help="Output folder")
    p.add_argument("--save_full_size", action="store_true", help="Save each zone as full-size image instead of cropped zone image")
    p.add_argument("--keep_yellow_overlay", action="store_true", help="Keep yellow overlay in outputs instead of removing it")
    p.add_argument("--no_clean_full_image", action="store_true", help="Do not save the full cleaned image")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    slice_zones(
        image_path=args.image,
        cx=args.cx,
        cy=args.cy,
        px_per_mm=args.px_per_mm,
        inner_r_mm=args.inner_r_mm,
        outer_r_mm=args.outer_r_mm,
        angle_deg=args.angle_deg,
        onh_offset_x=args.onh_offset_x,
        onh_offset_y=args.onh_offset_y,
        onh_rx=args.onh_rx,
        onh_ry=args.onh_ry,
        output_dir=args.output_dir,
        save_full_size=args.save_full_size,
        keep_yellow_overlay=args.keep_yellow_overlay,
        save_clean_full_image=not args.no_clean_full_image,
    )
