#!/usr/bin/env python3
"""
Extract 10 binary zone masks from fluorescein angiography (FA) images that
contain yellow zone overlays.

Expected overlay geometry
-------------------------
- One large circle centered near the fovea
- One smaller concentric circle
- Two orthogonal radial axes crossing at the same center
- One small optic-disc circle offset horizontally from the center

The script infers those primitives directly from the yellow overlay and saves:
- `zone_01.png` ... `zone_10.png`
- `label_map.png` with integer labels 0..10
- `qc_overlay.png` for visual inspection
- `geometry.json` with the recovered centers, radii, and laterality

Zone numbering follows the user's examples:
- Zone 1: disc-side upper inner quadrant
- Zone 2: opposite-side upper inner quadrant
- Zone 3: disc-side lower inner quadrant
- Zone 4: opposite-side lower inner quadrant
- Zone 5: disc-side upper outer quadrant
- Zone 6: opposite-side upper outer quadrant
- Zone 7: opposite-side lower outer quadrant
- Zone 8: disc-side lower outer quadrant
- Zone 9: optic disc circle
- Zone 10: visible retina outside zones 1-9
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for script usage
    raise SystemExit(
        "This script requires OpenCV. Install `opencv-python` in your Python environment "
        "and rerun the command."
    ) from exc


@dataclass
class Geometry:
    center_xy: tuple[float, float]
    inner_radius: float
    outer_radius: float
    disc_center_xy: tuple[float, float]
    disc_radius: float
    eye: str
    disc_axis_xy: tuple[float, float]
    vertical_axis_xy: tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Glob for annotated FA images, e.g. 'Dataset/FA_annotated/**/*.png'",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where extracted masks will be written, preserving input relative paths",
    )
    parser.add_argument(
        "--min-retina-threshold",
        type=int,
        default=8,
        help="Minimum grayscale value used to define the visible retina mask",
    )
    parser.add_argument(
        "--yellow-s-threshold",
        type=int,
        default=30,
        help="Minimum saturation for yellow overlay detection in HSV space",
    )
    parser.add_argument(
        "--yellow-v-threshold",
        type=int,
        default=80,
        help="Minimum value for yellow overlay detection in HSV space",
    )
    parser.add_argument(
        "--qc-limit",
        type=int,
        default=0,
        help="If > 0, stop after saving QC outputs for this many images",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def infer_input_root(pattern: str) -> Path:
    wildcard_chars = {"*", "?", "["}
    first_wildcard = min((pattern.find(ch) for ch in wildcard_chars if ch in pattern), default=-1)
    static_prefix = pattern if first_wildcard == -1 else pattern[:first_wildcard]
    static_path = Path(static_prefix).expanduser()

    if static_path.suffix:
        static_path = static_path.parent

    while not static_path.exists() and static_path != static_path.parent:
        static_path = static_path.parent

    return static_path.resolve()


def build_output_dir_for_image(path: Path, output_dir: Path, input_root: Path) -> Path:
    try:
        rel = path.resolve().relative_to(input_root)
    except ValueError:
        rel = Path(path.name)
    return output_dir / rel.with_suffix("")


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest


def build_retina_mask(rgb: np.ndarray, threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    mask = keep_largest_component(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2) > 0
    return keep_largest_component(mask)


def detect_yellow_overlay(rgb: np.ndarray, sat_threshold: int, val_threshold: int) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask_hsv = (h >= 15) & (h <= 45) & (s >= sat_threshold) & (v >= val_threshold)

    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    mask_rgb = (r > 110) & (g > 110) & (b < 180) & (np.abs(r - g) < 70) & ((r + g) - 2 * b > 70)

    mask = mask_hsv | mask_rgb
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1) > 0
    return mask


def hough_circle_candidates(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    blur = cv2.GaussianBlur(mask_u8, (9, 9), 1.5)
    min_dim = min(mask.shape[:2])
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=max(20, int(min_dim * 0.04)),
        param1=80,
        param2=12,
        minRadius=max(10, int(min_dim * 0.02)),
        maxRadius=max(20, int(min_dim * 0.42)),
    )
    if circles is None:
        raise RuntimeError("Could not detect zone circles from yellow overlay.")
    return circles[0]


def choose_concentric_pair(circles: np.ndarray, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    min_dim = min(image_shape[:2])
    circles = np.array(sorted(circles, key=lambda c: c[2], reverse=True), dtype=np.float32)
    best_pair = None
    best_score = None

    def consider_pair(c1: np.ndarray, c2: np.ndarray, center_tol_frac: float, min_ratio: float, target_ratio: float, ratio_weight: float) -> None:
        nonlocal best_pair, best_score
        center_dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
        if center_dist > center_tol_frac * min_dim:
            return
        outer, inner = (c1, c2) if c1[2] > c2[2] else (c2, c1)
        radius_ratio = outer[2] / max(inner[2], 1.0)
        if radius_ratio < min_ratio:
            return
        score = center_dist + ratio_weight * abs(radius_ratio - target_ratio)
        if best_score is None or score < best_score:
            best_score = score
            best_pair = (outer, inner)

    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            consider_pair(circles[i], circles[j], center_tol_frac=0.04, min_ratio=2.0, target_ratio=4.5, ratio_weight=1.0)

    if best_pair is None:
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                consider_pair(circles[i], circles[j], center_tol_frac=0.08, min_ratio=1.5, target_ratio=3.5, ratio_weight=0.5)

    if best_pair is None and len(circles) >= 2:
        largest = circles[: min(6, len(circles))]
        for i in range(len(largest)):
            for j in range(i + 1, len(largest)):
                c1, c2 = largest[i], largest[j]
                outer, inner = (c1, c2) if c1[2] > c2[2] else (c2, c1)
                radius_ratio = outer[2] / max(inner[2], 1.0)
                center_dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                score = center_dist + 0.25 * abs(radius_ratio - 3.0)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (outer, inner)

    if best_pair is None:
        raise RuntimeError("Could not identify concentric inner/outer circles.")
    return best_pair


def radial_histogram_peaks(mask: np.ndarray, cx: float, cy: float, min_radius: int, max_radius: int) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("No yellow overlay pixels found after thresholding.")
    distances = np.hypot(xs - cx, ys - cy)
    hist = np.bincount(np.clip(np.round(distances).astype(int), 0, max_radius), minlength=max_radius + 1).astype(np.float32)
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    hist = np.convolve(hist, kernel / kernel.sum(), mode="same")
    return hist[min_radius:max_radius + 1]


def refine_radii(mask: np.ndarray, center_xy: tuple[float, float], outer_radius_guess: float, inner_radius_guess: float) -> tuple[float, float]:
    cx, cy = center_xy
    max_radius = int(min(mask.shape[:2]) * 0.48)
    hist = radial_histogram_peaks(mask, cx, cy, 1, max_radius)

    def local_peak(guess: float, frac: float) -> float:
        radius = int(round(guess))
        window = max(6, int(frac * guess))
        start = max(1, radius - window)
        end = min(max_radius, radius + window)
        sub = hist[start - 1:end]
        peak = np.argmax(sub) + start
        return float(peak)

    outer_radius = local_peak(outer_radius_guess, 0.12)
    inner_radius = local_peak(inner_radius_guess, 0.20)
    return inner_radius, outer_radius


def get_centroid(contour: np.ndarray) -> tuple[int | None, int | None]:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None, None
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def label_contours_for_ten(contours: list[np.ndarray], height: int, width: int) -> np.ndarray:
    label_map = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(label_map, [contours[0]], -1, 255, thickness=cv2.FILLED)
    label_map = np.where(label_map == 0, 10, 0).astype(np.uint8)

    quad_contours = contours[1:5]
    centroids = [(get_centroid(contour), contour) for contour in quad_contours]
    centroids_sorted_by_y = sorted(centroids, key=lambda item: item[0][1])
    upper_two = sorted(centroids_sorted_by_y[:2], key=lambda item: item[0][0])
    lower_two = sorted(centroids_sorted_by_y[2:], key=lambda item: item[0][0])

    for zone_num, ((_cx, _cy), contour) in zip([8, 7], lower_two):
        cv2.drawContours(label_map, [contour], -1, zone_num, thickness=cv2.FILLED)
    for zone_num, ((_cx, _cy), contour) in zip([5, 6], upper_two):
        cv2.drawContours(label_map, [contour], -1, zone_num, thickness=cv2.FILLED)

    small_contours = contours[5:10]
    small_centroids = [(get_centroid(contour), contour) for contour in small_contours]
    small_sorted_by_x = sorted(small_centroids, key=lambda item: item[0][0])

    (_cx, _cy), disc_contour = small_sorted_by_x[0]
    cv2.drawContours(label_map, [disc_contour], -1, 9, thickness=cv2.FILLED)

    remaining_sorted_by_y = sorted(small_sorted_by_x[1:], key=lambda item: item[0][1])
    upper_two_small = sorted(remaining_sorted_by_y[:2], key=lambda item: item[0][0])
    lower_two_small = sorted(remaining_sorted_by_y[2:], key=lambda item: item[0][0])

    for zone_num, ((_cx, _cy), contour) in zip([3, 4], lower_two_small):
        cv2.drawContours(label_map, [contour], -1, zone_num, thickness=cv2.FILLED)
    for zone_num, ((_cx, _cy), contour) in zip([1, 2], upper_two_small):
        cv2.drawContours(label_map, [contour], -1, zone_num, thickness=cv2.FILLED)

    return label_map


def label_contours_for_eleven(contours: list[np.ndarray], height: int, width: int) -> np.ndarray:
    label_map = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(label_map, [contours[0]], -1, 255, thickness=cv2.FILLED)
    label_map = np.where(label_map == 0, 10, 0).astype(np.uint8)

    quad_contours = contours[1:5]
    centroids = [(get_centroid(contour), contour) for contour in quad_contours]
    centroids_sorted_by_y = sorted(centroids, key=lambda item: item[0][1])
    upper_two = sorted(centroids_sorted_by_y[:2], key=lambda item: item[0][0])
    lower_two = sorted(centroids_sorted_by_y[2:], key=lambda item: item[0][0])

    for zone_num, ((_cx, _cy), contour) in zip([8, 7], lower_two):
        cv2.drawContours(label_map, [contour], -1, zone_num, thickness=cv2.FILLED)
    for zone_num, ((_cx, _cy), contour) in zip([5, 6], upper_two):
        cv2.drawContours(label_map, [contour], -1, zone_num, thickness=cv2.FILLED)

    small_contours = contours[5:11]
    small_centroids = [(get_centroid(contour), contour) for contour in small_contours]
    small_sorted_by_x = sorted(small_centroids, key=lambda item: item[0][0])

    for (_cx, _cy), contour in small_sorted_by_x[:2]:
        cv2.drawContours(label_map, [contour], -1, 9, thickness=cv2.FILLED)

    remaining_sorted_by_y = sorted(small_sorted_by_x[2:], key=lambda item: item[0][1])
    upper_two_small = sorted(remaining_sorted_by_y[:2], key=lambda item: item[0][0])
    lower_two_small = sorted(remaining_sorted_by_y[2:], key=lambda item: item[0][0])

    for zone_num, ((_cx, _cy), contour) in zip([3, 4], lower_two_small):
        cv2.drawContours(label_map, [contour], -1, zone_num, thickness=cv2.FILLED)
    for zone_num, ((_cx, _cy), contour) in zip([1, 2], upper_two_small):
        cv2.drawContours(label_map, [contour], -1, zone_num, thickness=cv2.FILLED)

    return label_map


def contour_label_map_from_rgb(rgb: np.ndarray) -> tuple[np.ndarray, int]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    yellow = cv2.dilate(yellow, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(yellow, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 500]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    height, width = rgb.shape[:2]

    if len(contours) == 10:
        return label_contours_for_ten(contours, height, width), 10
    if len(contours) == 11:
        return label_contours_for_eleven(contours, height, width), 11
    raise RuntimeError(f"Contour fallback found {len(contours)} contours, expected 10/11.")


def detect_disc_circle(mask: np.ndarray, center_xy: tuple[float, float], inner_radius: float, outer_radius: float) -> tuple[tuple[float, float], float]:
    cy_grid, cx_grid = np.indices(mask.shape)
    cx, cy = center_xy
    dist = np.hypot(cx_grid - cx, cy_grid - cy)
    ring_tol_outer = max(2.5, outer_radius * 0.015)
    ring_tol_inner = max(2.5, inner_radius * 0.03)

    residual = mask.copy()
    residual &= np.abs(dist - outer_radius) > ring_tol_outer
    residual &= np.abs(dist - inner_radius) > ring_tol_inner
    residual &= dist < outer_radius * 0.85

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(residual.astype(np.uint8), connectivity=8)
    candidates: list[tuple[float, tuple[float, float], float]] = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 20:
            continue
        component = labels == label
        ys, xs = np.nonzero(component)
        if len(xs) < 10:
            continue
        (x_disc, y_disc), disc_radius = cv2.minEnclosingCircle(np.column_stack([xs, ys]).astype(np.float32))
        if disc_radius < max(5.0, inner_radius * 0.15) or disc_radius > outer_radius * 0.25:
            continue
        center_dist = math.hypot(x_disc - cx, y_disc - cy)
        if center_dist < inner_radius * 0.7 or center_dist > outer_radius * 0.6:
            continue
        circularity_score = abs(center_dist - inner_radius * 1.2)
        candidates.append((circularity_score, (float(x_disc), float(y_disc)), float(disc_radius)))

    if not candidates:
        mask_u8 = (residual.astype(np.uint8) * 255)
        blur = cv2.GaussianBlur(mask_u8, (7, 7), 1.2)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=max(12, int(inner_radius * 0.3)),
            param1=60,
            param2=8,
            minRadius=max(5, int(inner_radius * 0.15)),
            maxRadius=max(8, int(outer_radius * 0.22)),
        )
        if circles is not None:
            for x_disc, y_disc, disc_radius in circles[0]:
                center_dist = math.hypot(x_disc - cx, y_disc - cy)
                if center_dist < inner_radius * 0.5 or center_dist > outer_radius * 0.7:
                    continue
                horizontal_bias = abs(y_disc - cy)
                candidates.append((horizontal_bias + abs(center_dist - inner_radius * 1.15), (float(x_disc), float(y_disc)), float(disc_radius)))

    if not candidates:
        try:
            all_circles = hough_circle_candidates(mask)
        except RuntimeError:
            all_circles = np.empty((0, 3), dtype=np.float32)
        for x_disc, y_disc, disc_radius in all_circles:
            center_dist = math.hypot(x_disc - cx, y_disc - cy)
            if disc_radius > outer_radius * 0.25 or disc_radius < max(4.0, inner_radius * 0.08):
                continue
            if center_dist < inner_radius * 0.5 or center_dist > outer_radius * 0.7:
                continue
            score = abs(y_disc - cy) + abs(center_dist - inner_radius * 1.15) + 0.5 * disc_radius
            candidates.append((score, (float(x_disc), float(y_disc)), float(disc_radius)))

    if not candidates:
        raise RuntimeError("Could not isolate the optic-disc circle.")

    _, disc_center_xy, disc_radius = min(candidates, key=lambda item: item[0])
    return disc_center_xy, disc_radius


def detect_axes(mask: np.ndarray, center_xy: tuple[float, float], inner_radius: float, outer_radius: float) -> tuple[np.ndarray, np.ndarray]:
    cy_grid, cx_grid = np.indices(mask.shape)
    cx, cy = center_xy
    dist = np.hypot(cx_grid - cx, cy_grid - cy)

    residual = mask.copy()
    residual &= dist > inner_radius * 0.6
    residual &= dist < outer_radius * 1.05
    residual &= np.abs(dist - inner_radius) > max(2.5, inner_radius * 0.03)
    residual &= np.abs(dist - outer_radius) > max(2.5, outer_radius * 0.015)

    ys, xs = np.nonzero(residual)
    if len(xs) == 0:
        raise RuntimeError("Could not isolate radial line pixels.")

    angles = (np.degrees(np.arctan2(-(ys - cy), xs - cx)) + 180.0) % 180.0
    bins = np.arange(181)
    hist, _ = np.histogram(angles, bins=bins)
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    hist = np.convolve(hist.astype(np.float32), kernel / kernel.sum(), mode="same")

    axis_1_deg = float(np.argmax(hist))
    suppress = np.abs(((np.arange(len(hist)) - axis_1_deg + 90) % 180) - 90) < 25
    hist_2 = hist.copy()
    hist_2[suppress] = 0
    axis_2_deg = float(np.argmax(hist_2))

    def vec_from_deg(deg: float) -> np.ndarray:
        rad = math.radians(deg)
        return np.array([math.cos(rad), -math.sin(rad)], dtype=np.float32)

    v1 = vec_from_deg(axis_1_deg)
    v2 = vec_from_deg(axis_2_deg)
    if abs(float(np.dot(v1, v2))) > 0.4:
        if abs(v1[0]) > abs(v2[0]):
            v2 = np.array([-v1[1], v1[0]], dtype=np.float32)
        else:
            v1 = np.array([v2[1], -v2[0]], dtype=np.float32)
    return v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)


def orient_axes(disc_center_xy: tuple[float, float], center_xy: tuple[float, float], axis_a: np.ndarray, axis_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    disc_vec = np.array([disc_center_xy[0] - center_xy[0], disc_center_xy[1] - center_xy[1]], dtype=np.float32)
    disc_vec /= max(np.linalg.norm(disc_vec), 1e-6)

    if abs(float(np.dot(axis_a, disc_vec))) >= abs(float(np.dot(axis_b, disc_vec))):
        disc_axis = axis_a.copy()
        vertical_axis = axis_b.copy()
    else:
        disc_axis = axis_b.copy()
        vertical_axis = axis_a.copy()

    if float(np.dot(disc_axis, disc_vec)) < 0:
        disc_axis *= -1.0
    if vertical_axis[1] > 0:
        vertical_axis *= -1.0

    eye = "OS" if disc_center_xy[0] < center_xy[0] else "OD"
    return disc_axis, vertical_axis, eye


def build_zone_masks(image_shape: tuple[int, int], retina_mask: np.ndarray, geometry: Geometry) -> dict[int, np.ndarray]:
    h, w = image_shape
    yy, xx = np.indices((h, w))
    cx, cy = geometry.center_xy
    dx = xx - cx
    dy = yy - cy
    dist = np.hypot(dx, dy)

    disc_axis = np.array(geometry.disc_axis_xy, dtype=np.float32)
    vertical_axis = np.array(geometry.vertical_axis_xy, dtype=np.float32)
    disc_proj = dx * disc_axis[0] + dy * disc_axis[1]
    vert_proj = dx * vertical_axis[0] + dy * vertical_axis[1]

    disc_cx, disc_cy = geometry.disc_center_xy
    disc_dist = np.hypot(xx - disc_cx, yy - disc_cy)
    disc_mask = disc_dist <= geometry.disc_radius

    inner_region = retina_mask & (dist <= geometry.inner_radius) & ~disc_mask
    outer_region = retina_mask & (dist > geometry.inner_radius) & (dist <= geometry.outer_radius) & ~disc_mask

    masks = {
        1: inner_region & (disc_proj >= 0) & (vert_proj >= 0),
        2: inner_region & (disc_proj < 0) & (vert_proj >= 0),
        3: inner_region & (disc_proj >= 0) & (vert_proj < 0),
        4: inner_region & (disc_proj < 0) & (vert_proj < 0),
        5: outer_region & (disc_proj >= 0) & (vert_proj >= 0),
        6: outer_region & (disc_proj < 0) & (vert_proj >= 0),
        7: outer_region & (disc_proj < 0) & (vert_proj < 0),
        8: outer_region & (disc_proj >= 0) & (vert_proj < 0),
        9: retina_mask & disc_mask,
    }
    occupied = np.zeros_like(retina_mask, dtype=bool)
    for zone_mask in masks.values():
        occupied |= zone_mask
    masks[10] = retina_mask & ~occupied
    return masks


def build_label_map(zone_masks: dict[int, np.ndarray]) -> np.ndarray:
    shape = next(iter(zone_masks.values())).shape
    label_map = np.zeros(shape, dtype=np.uint8)
    for zone in range(1, 11):
        label_map[zone_masks[zone]] = zone
    return label_map


def zone_masks_from_label_map(label_map: np.ndarray) -> dict[int, np.ndarray]:
    return {zone: label_map == zone for zone in range(1, 11)}


def make_qc_overlay(rgb: np.ndarray, zone_masks: dict[int, np.ndarray], geometry: Geometry | None) -> np.ndarray:
    overlay = rgb.copy()
    colors = {
        1: (255, 0, 0),
        2: (255, 128, 0),
        3: (255, 255, 0),
        4: (128, 255, 0),
        5: (0, 255, 0),
        6: (0, 255, 255),
        7: (0, 128, 255),
        8: (0, 0, 255),
        9: (255, 0, 255),
        10: (192, 192, 192),
    }
    for zone, color in colors.items():
        mask = zone_masks[zone]
        overlay[mask] = (0.65 * overlay[mask] + 0.35 * np.array(color)).astype(np.uint8)

    if geometry is not None:
        cx, cy = map(int, map(round, geometry.center_xy))
        disc_cx, disc_cy = map(int, map(round, geometry.disc_center_xy))
        cv2.circle(overlay, (cx, cy), int(round(geometry.inner_radius)), (255, 255, 255), 2)
        cv2.circle(overlay, (cx, cy), int(round(geometry.outer_radius)), (255, 255, 255), 2)
        cv2.circle(overlay, (disc_cx, disc_cy), int(round(geometry.disc_radius)), (255, 255, 255), 2)
        cv2.putText(overlay, geometry.eye, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def geometry_to_json(geometry: Geometry) -> dict[str, object]:
    return {
        "center_xy": [round(geometry.center_xy[0], 3), round(geometry.center_xy[1], 3)],
        "inner_radius": round(geometry.inner_radius, 3),
        "outer_radius": round(geometry.outer_radius, 3),
        "disc_center_xy": [round(geometry.disc_center_xy[0], 3), round(geometry.disc_center_xy[1], 3)],
        "disc_radius": round(geometry.disc_radius, 3),
        "eye": geometry.eye,
        "disc_axis_xy": [round(geometry.disc_axis_xy[0], 6), round(geometry.disc_axis_xy[1], 6)],
        "vertical_axis_xy": [round(geometry.vertical_axis_xy[0], 6), round(geometry.vertical_axis_xy[1], 6)],
    }


def process_image(
    path: Path,
    output_dir: Path,
    input_root: Path,
    retina_threshold: int,
    yellow_s_threshold: int,
    yellow_v_threshold: int,
) -> Path:
    rgb = load_rgb(path)
    retina_mask = build_retina_mask(rgb, threshold=retina_threshold)
    yellow_mask = detect_yellow_overlay(rgb, sat_threshold=yellow_s_threshold, val_threshold=yellow_v_threshold)

    try:
        label_map, contour_count = contour_label_map_from_rgb(rgb)
        zone_masks = zone_masks_from_label_map(label_map)
        geometry = None
        extraction_method = f"contours_{contour_count}"
    except Exception as contour_exc:
        circles = hough_circle_candidates(yellow_mask)
        outer_circle, inner_circle = choose_concentric_pair(circles, rgb.shape[:2])
        center_xy = ((float(outer_circle[0] + inner_circle[0]) / 2.0), (float(outer_circle[1] + inner_circle[1]) / 2.0))
        inner_radius, outer_radius = refine_radii(
            yellow_mask,
            center_xy=center_xy,
            outer_radius_guess=float(outer_circle[2]),
            inner_radius_guess=float(inner_circle[2]),
        )
        disc_center_xy, disc_radius = detect_disc_circle(yellow_mask, center_xy=center_xy, inner_radius=inner_radius, outer_radius=outer_radius)
        axis_a, axis_b = detect_axes(yellow_mask, center_xy=center_xy, inner_radius=inner_radius, outer_radius=outer_radius)
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
        label_map = build_label_map(zone_masks)
        extraction_method = f"geometry_fallback: {contour_exc}"

    qc_overlay = make_qc_overlay(rgb, zone_masks, geometry)

    image_output_dir = build_output_dir_for_image(path, output_dir, input_root)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    for zone in range(1, 11):
        save_mask(zone_masks[zone], image_output_dir / f"zone_{zone:02d}.png")
    Image.fromarray(label_map, mode="L").save(image_output_dir / "label_map.png")
    Image.fromarray(qc_overlay).save(image_output_dir / "qc_overlay.png")
    save_mask(retina_mask, image_output_dir / "retina_mask.png")
    save_mask(yellow_mask, image_output_dir / "yellow_overlay_mask.png")
    with open(image_output_dir / "geometry.json", "w", encoding="utf-8") as f:
        payload = geometry_to_json(geometry) if geometry is not None else {"method": extraction_method}
        json.dump(payload, f, indent=2)
    return image_output_dir


def iter_input_paths(pattern: str) -> Iterable[Path]:
    for match in sorted(glob.glob(str(Path(pattern).expanduser()), recursive=True)):
        path = Path(match)
        if path.is_file():
            yield path.resolve()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = infer_input_root(args.input_glob)

    paths = list(iter_input_paths(args.input_glob))
    if not paths:
        raise SystemExit(f"No images matched pattern: {args.input_glob}")

    summary: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        try:
            image_output_dir = process_image(
                path,
                output_dir=output_dir,
                input_root=input_root,
                retina_threshold=args.min_retina_threshold,
                yellow_s_threshold=args.yellow_s_threshold,
                yellow_v_threshold=args.yellow_v_threshold,
            )
            summary.append({"image": str(path), "status": "ok", "output_dir": str(image_output_dir)})
            print(f"[{index}/{len(paths)}] OK  {path} -> {image_output_dir}")
        except Exception as exc:  # pragma: no cover - batch scripts should continue
            summary.append({"image": str(path), "status": "error", "error": str(exc)})
            print(f"[{index}/{len(paths)}] ERR {path}: {exc}")

        if args.qc_limit > 0 and index >= args.qc_limit:
            break

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
