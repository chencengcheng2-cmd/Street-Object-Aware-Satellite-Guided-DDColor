"""Create colored visualization copies of semantic ID masks.

This does not modify the original training masks. Original masks keep class IDs
0-5; visualization masks use RGB colors for human inspection.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PALETTES = {
    "old": np.array([
        [80, 170, 255],   # 0 sky
        [128, 128, 128],  # 1 road
        [40, 180, 70],    # 2 vegetation
        [190, 150, 110],  # 3 building
        [40, 90, 220],    # 4 water
        [220, 220, 220],  # 5 other
    ], dtype=np.uint8),
    "neos": np.array([
        [80, 170, 255],   # 0 Sky
        [255, 255, 255],  # 1 Impervious surfaces / Roads
        [0, 90, 255],     # 2 Building
        [0, 255, 255],    # 3 Low vegetation
        [0, 170, 0],      # 4 Tree
        [255, 230, 0],    # 5 Car
        [255, 0, 0],      # 6 Clutter / Background
    ], dtype=np.uint8),
}


def colorize(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    mask = np.clip(mask, 0, len(palette) - 1)
    return palette[mask]


def main():
    parser = argparse.ArgumentParser(description="Colorize semantic ID masks for visualization.")
    parser.add_argument("--dataset_root", default="C:/Users/31133/Desktop/dataset1/CVUSA_processed_split")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--semantic_dirs", nargs="+", default=["street_semantic", "overhead_satellite_semantic"])
    parser.add_argument("--suffix", default="_color")
    parser.add_argument("--palette", choices=sorted(PALETTES), default="neos")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    palette = PALETTES[args.palette]
    for split in args.splits:
        split_root = dataset_root / split
        for dirname in args.semantic_dirs:
            src_dir = split_root / dirname
            if not src_dir.exists():
                print(f"[Skip] missing {src_dir}")
                continue
            dst_dir = split_root / f"{dirname}{args.suffix}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            paths = sorted(src_dir.glob("*.png"))
            if args.limit is not None:
                paths = paths[: args.limit]
            print(f"[{split}] {dirname}: {len(paths)} masks -> {dst_dir}")
            for path in tqdm(paths):
                out_path = dst_dir / path.name
                if out_path.exists() and not args.overwrite:
                    continue
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                colored = colorize(mask, palette)
                cv2.imwrite(str(out_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
