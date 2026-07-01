"""Export weak road-marking/detail masks for street-view and satellite images.

The street branch intentionally avoids ground-truth street RGB color cues. It
uses grayscale structure + street road semantics to locate possible markings.
The satellite branch uses satellite RGB + satellite impervious/road semantics
to provide color-visible marking evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

DETAIL_PALETTE = np.array(
    [
        [0, 0, 0],         # 0 none
        [255, 255, 255],   # 1 impervious_edge / road edge
        [255, 230, 0],     # 2 linear_marking
        [255, 140, 0],     # 3 transverse_marking
        [255, 80, 220],    # 4 road_symbol_marking
    ],
    dtype=np.uint8,
)


def iter_image_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir(), key=lambda p: p.stem):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_label(path: Path, size_hw: tuple[int, int]) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise ValueError(f"Failed to read label map: {path}")
    h, w = size_hw
    if label.shape[:2] != (h, w):
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    return label.astype(np.uint8)


def colorize_detail(mask: np.ndarray) -> np.ndarray:
    return DETAIL_PALETTE[np.clip(mask, 0, len(DETAIL_PALETTE) - 1)]


def _classify_marking_components(marking_mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    detail = np.zeros((h, w), dtype=np.uint8)
    marking_u8 = (marking_mask > 0).astype(np.uint8)
    if marking_u8.max() == 0:
        return detail

    marking_u8 = cv2.morphologyEx(marking_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(marking_u8, connectivity=8)
    image_area = float(h * w)
    for label_id in range(1, num_labels):
        x, y, bw, bh, area = stats[label_id]
        if area < 3:
            continue
        short = max(1, min(bw, bh))
        long = max(bw, bh)
        elongation = long / short
        fill = area / max(1, bw * bh)
        component = labels == label_id

        if elongation >= 3.0 and area <= 0.014 * image_area:
            detail[component] = 2  # linear_marking
        elif bw >= 1.8 * bh and area <= 0.030 * image_area:
            detail[component] = 3  # transverse_marking
        elif area >= 8 and fill <= 0.82:
            detail[component] = 4  # road_symbol_marking
        else:
            detail[component] = 2
    return detail


def street_detail_mask(image_rgb_or_gray: np.ndarray, street_labels: np.ndarray, road_label: int) -> np.ndarray:
    """Street-side marking candidates from grayscale structure only."""
    if image_rgb_or_gray.ndim == 3:
        gray = cv2.cvtColor(image_rgb_or_gray, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb_or_gray
    h, w = gray.shape[:2]
    road = street_labels == road_label

    lower = np.zeros((h, w), dtype=bool)
    lower[int(h * 0.32):, :] = True

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 135) > 0
    edges = cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1).astype(bool)
    if road.any():
        bright_thr = max(110, int(np.percentile(gray[road], 70)))
    else:
        bright_thr = 125
    bright_or_contrast = (gray >= bright_thr) | (cv2.absdiff(gray, cv2.GaussianBlur(gray, (17, 17), 0)) > 18)
    candidate = road & lower & edges & bright_or_contrast

    detail = np.zeros((h, w), dtype=np.uint8)
    road_edge = road & edges
    detail[road_edge] = 1
    marking_detail = _classify_marking_components(candidate, (h, w))
    detail[marking_detail > 0] = marking_detail[marking_detail > 0]
    return detail


def satellite_detail_mask(satellite_rgb: np.ndarray, satellite_labels: np.ndarray, road_label: int) -> np.ndarray:
    """Satellite-side road marking evidence from RGB + road/impervious mask."""
    h, w = satellite_rgb.shape[:2]
    road = satellite_labels == road_label
    hsv = cv2.cvtColor(satellite_rgb, cv2.COLOR_RGB2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    r, g, b = satellite_rgb[..., 0], satellite_rgb[..., 1], satellite_rgb[..., 2]

    gray = cv2.cvtColor(satellite_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150) > 0
    yellow = (
        road
        & (hh >= 14)
        & (hh <= 42)
        & (ss >= 45)
        & (vv >= 90)
        & (r.astype(np.int16) > b.astype(np.int16) + 20)
        & (g.astype(np.int16) > b.astype(np.int16) + 15)
    )
    white = (
        road
        & (ss <= 45)
        & (vv >= 165)
        & ((np.maximum.reduce([r, g, b]).astype(np.int16) - np.minimum.reduce([r, g, b]).astype(np.int16)) <= 38)
    )
    marking = (yellow | white) & (edges | (vv > 160))

    detail = np.zeros((h, w), dtype=np.uint8)
    detail[road & edges] = 1
    marking_detail = _classify_marking_components(marking, (h, w))
    detail[marking_detail > 0] = marking_detail[marking_detail > 0]
    return detail


def find_label(label_dir: Path, image_stem: str) -> Path | None:
    candidates = [
        label_dir / f"{image_stem}.png",
        label_dir / f"{image_stem}_semantic.png",
        label_dir / f"{image_stem}_mask.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def export_split(args, split: str):
    split_dir = Path(args.root) / split
    street_img_dir = split_dir / args.street_image_dir
    street_sem_dir = split_dir / args.street_semantic_dir
    sat_img_dir = split_dir / args.satellite_image_dir
    sat_sem_dir = split_dir / args.satellite_semantic_dir

    street_out = split_dir / args.street_output_dir
    satellite_out = split_dir / args.satellite_output_dir
    street_vis = split_dir / f"{args.street_output_dir}_vis"
    satellite_vis = split_dir / f"{args.satellite_output_dir}_vis"
    for d in (street_out, satellite_out, street_vis, satellite_vis):
        d.mkdir(parents=True, exist_ok=True)

    street_count = 0
    for image_path in iter_image_files(street_img_dir):
        label_path = find_label(street_sem_dir, image_path.stem.replace("_gray", ""))
        if label_path is None:
            label_path = find_label(street_sem_dir, image_path.stem)
        if label_path is None:
            continue
        rgb = read_rgb(image_path)
        labels = read_label(label_path, rgb.shape[:2])
        detail = street_detail_mask(rgb, labels, args.street_road_label)
        cv2.imwrite(str(street_out / f"{image_path.stem.replace('_gray', '')}.png"), detail)
        cv2.imwrite(str(street_vis / f"{image_path.stem.replace('_gray', '')}.png"), cv2.cvtColor(colorize_detail(detail), cv2.COLOR_RGB2BGR))
        street_count += 1

    satellite_count = 0
    for image_path in iter_image_files(sat_img_dir):
        label_path = find_label(sat_sem_dir, image_path.stem)
        if label_path is None:
            continue
        rgb = read_rgb(image_path)
        labels = read_label(label_path, rgb.shape[:2])
        detail = satellite_detail_mask(rgb, labels, args.satellite_road_label)
        cv2.imwrite(str(satellite_out / f"{image_path.stem}.png"), detail)
        cv2.imwrite(str(satellite_vis / f"{image_path.stem}.png"), cv2.cvtColor(colorize_detail(detail), cv2.COLOR_RGB2BGR))
        satellite_count += 1

    print(f"[Detail v16] split={split} street={street_count} satellite={satellite_count}")


def main():
    parser = argparse.ArgumentParser(description="Export weak road marking/detail masks for v16.")
    parser.add_argument("--root", default="C:/Users/31133/Desktop/dataset1/CVUSA_processed_split")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--street_image_dir", default="ground_gray")
    parser.add_argument("--street_semantic_dir", default="street_dino_semantic_v12")
    parser.add_argument("--satellite_image_dir", default="overhead_satellite")
    parser.add_argument("--satellite_semantic_dir", default="overhead_satellite_deeplab_v13")
    parser.add_argument("--street_output_dir", default="street_road_detail_v16")
    parser.add_argument("--satellite_output_dir", default="overhead_satellite_road_detail_v16")
    parser.add_argument("--street_road_label", type=int, default=0, help="DINO-v12 street road label is 0.")
    parser.add_argument("--satellite_road_label", type=int, default=0, help="DeepLab/v16 satellite impervious/road label is 0.")
    args = parser.parse_args()

    for split in args.splits:
        export_split(args, split)


if __name__ == "__main__":
    main()
