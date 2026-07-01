"""Import existing CVUSA/NEOS satellite segmentation maps into dataset1.

Input masks are color-coded aerial masks with Potsdam/NEOS-like colors:
white roads/impervious, blue buildings, cyan low vegetation, green trees,
yellow cars, red clutter/background. The output is a single-channel ID map using
this project's unified 7-class schema:
0 sky (unused for satellite), 1 roads/impervious, 2 building, 3 low vegetation,
4 tree, 5 car, 6 clutter/background.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# RGB palette for input color maps -> project labels.
INPUT_PALETTE = np.array([
    [255, 255, 255],  # roads / impervious
    [0, 0, 255],      # building (RGB blue)
    [0, 255, 255],    # low vegetation
    [0, 255, 0],      # tree
    [255, 255, 0],    # car
    [255, 0, 0],      # clutter/background
], dtype=np.float32)
PROJECT_IDS = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)


def color_to_label(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.float32)
    d2 = ((flat[:, None, :] - INPUT_PALETTE[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1)
    return PROJECT_IDS[nearest].reshape(rgb.shape[:2])


def convert_one(src: Path, dst: Path, size=(256, 256)):
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read {src}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = color_to_label(rgb)
    if label.shape[:2] != size:
        label = cv2.resize(label, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), label)


def main():
    parser = argparse.ArgumentParser(description="Import NEOS-style CVUSA satellite segmaps.")
    parser.add_argument("--segmap_root", default="C:/Users/31133/Desktop/dataset2/Dataset_CVUSA/segmap")
    parser.add_argument("--dataset_root", default="C:/Users/31133/Desktop/dataset1/CVUSA_processed_split")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--output_dirname", default="overhead_satellite_neos_semantic")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    segmap_root = Path(args.segmap_root)
    dataset_root = Path(args.dataset_root)
    missing = []
    converted = 0

    for split in args.splits:
        sat_dir = dataset_root / split / "overhead_satellite"
        out_dir = dataset_root / split / args.output_dirname
        sat_paths = sorted(sat_dir.glob("*.png")) + sorted(sat_dir.glob("*.jpg"))
        if args.limit is not None:
            sat_paths = sat_paths[: args.limit]
        print(f"[{split}] importing {len(sat_paths)} satellite segmaps -> {out_dir}")
        for sat_path in tqdm(sat_paths):
            stem = sat_path.stem
            src = segmap_root / f"output{stem}.png"
            if not src.exists():
                missing.append(stem)
                continue
            dst = out_dir / f"{stem}.png"
            if dst.exists() and not args.overwrite:
                continue
            convert_one(src, dst)
            converted += 1

    print(f"Converted: {converted}")
    print(f"Missing: {len(missing)}")
    if missing[:20]:
        print("First missing:", ", ".join(missing[:20]))


if __name__ == "__main__":
    main()
