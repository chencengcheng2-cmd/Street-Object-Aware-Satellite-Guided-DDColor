"""Generate NEOS-style semantic labels for CVUSA with SegFormer.

Class IDs follow the NEOS/CVUSA aerial semantic scheme requested for this project:
0 Sky
1 Impervious surfaces / Roads
2 Building
3 Low vegetation
4 Tree
5 Car
6 Clutter / Background

This script uses a SegFormer checkpoint as a practical local pseudo-labeler. If
real NEOS weights are available later, replace only this generator; the training
pipeline can keep the same class IDs and directories.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

NEOS_CLASSES = {
    "sky": 0,
    "impervious_roads": 1,
    "building": 2,
    "low_vegetation": 3,
    "tree": 4,
    "car": 5,
    "clutter_background": 6,
}

# ADE20K/SegFormer labels are mapped by keywords. This is not a replacement for
# real NEOS domain adaptation, but it gives the project the same target schema.
KEYWORDS = {
    NEOS_CLASSES["sky"]: ["sky"],
    NEOS_CLASSES["impervious_roads"]: [
        "road", "sidewalk", "path", "runway", "bridge", "floor", "earth", "dirt",
        "parking", "plaza", "track", "street", "pavement", "concrete", "asphalt"
    ],
    NEOS_CLASSES["building"]: [
        "building", "house", "skyscraper", "wall", "roof", "tower", "fence", "grandstand"
    ],
    NEOS_CLASSES["low_vegetation"]: [
        "grass", "field", "plant", "flower", "farm", "land", "terrain"
    ],
    NEOS_CLASSES["tree"]: ["tree", "palm", "forest", "bush"],
    NEOS_CLASSES["car"]: ["car", "truck", "bus", "van", "vehicle", "automobile"],
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_id_mapping(id2label: dict[int, str]) -> np.ndarray:
    max_id = max(int(k) for k in id2label.keys())
    mapping = np.full(max_id + 1, NEOS_CLASSES["clutter_background"], dtype=np.uint8)
    for raw_id, name in id2label.items():
        raw_id = int(raw_id)
        lower = name.lower()
        for target_id, words in KEYWORDS.items():
            if any(word in lower for word in words):
                mapping[raw_id] = target_id
                break
    return mapping


def iter_images(root: Path):
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def apply_neos_rules(rgb: np.ndarray, labels: np.ndarray, is_satellite: bool) -> np.ndarray:
    """Lightweight post-processing for common CVUSA errors.

    The rules target the exact failures observed in the previous masks: roads
    and dry/low vegetation being assigned to background.
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    lum = rgb_f.mean(axis=2)
    sat = rgb_f.max(axis=2) - rgb_f.min(axis=2)
    h, w = labels.shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]

    clutter = labels == NEOS_CLASSES["clutter_background"]

    if is_satellite:
        labels[labels == NEOS_CLASSES["sky"]] = NEOS_CLASSES["clutter_background"]

    # Impervious / hard-surface ground includes roads, parking lots, plazas,
    # concrete, asphalt and dirt roads. Roofs are not included here; building
    # labels are protected from this reassignment.
    road_like = clutter & (sat < 0.18) & (lum > 0.22) & (lum < 0.88)
    dark_asphalt_like = clutter & (sat < 0.18) & (lum > 0.10) & (lum <= 0.35)
    dirt_road_like = (
        clutter
        & (r > b * 1.05)
        & (g > b * 0.95)
        & (lum > 0.16)
        & (lum < 0.70)
        & (sat > 0.06)
        & (sat < 0.32)
        & (g < r * 1.08)
    )
    if not is_satellite:
        ground_zone = yy > 0.35
        road_like &= ground_zone
        dark_asphalt_like &= ground_zone
        dirt_road_like &= ground_zone
    labels[road_like | dark_asphalt_like | dirt_road_like] = NEOS_CLASSES["impervious_roads"]

    # Low vegetation includes grass, crops, dry grass and yellow/brown ground cover.
    green_lowveg = clutter & (g > r * 0.95) & (g >= b * 0.92) & (lum > 0.15)
    dry_lowveg = clutter & (r > b * 1.05) & (g > b * 1.03) & (lum > 0.18) & (sat > 0.08)
    if not is_satellite:
        green_lowveg &= yy > 0.25
        dry_lowveg &= yy > 0.35
    labels[green_lowveg | dry_lowveg] = NEOS_CLASSES["low_vegetation"]

    if is_satellite:
        # CVUSA aerial roads and hard surfaces are often pale grey/white.
        # ADE-style SegFormer can confuse them with low vegetation. Reassign only
        # low-saturation, high-luminance pixels where green is not dominant.
        lowveg_to_impervious = (
            (labels == NEOS_CLASSES["low_vegetation"])
            & (sat < 0.16)
            & (lum > 0.32)
            & (g < r * 1.05)
            & (g < b * 1.10)
        )
        lowveg_dark_asphalt = (
            (labels == NEOS_CLASSES["low_vegetation"])
            & (sat < 0.14)
            & (lum > 0.12)
            & (lum < 0.42)
            & (g < r * 1.06)
            & (g < b * 1.10)
        )
        labels[lowveg_to_impervious | lowveg_dark_asphalt] = NEOS_CLASSES["impervious_roads"]

    # Strong green vertical/large regions are usually trees, especially in street views.
    tree_like = (labels == NEOS_CLASSES["low_vegetation"]) & (g > r * 1.08) & (g > b * 1.05) & (sat > 0.10)
    if not is_satellite:
        tree_like &= yy < 0.85
    labels[tree_like] = NEOS_CLASSES["tree"]

    return labels


def main():
    parser = argparse.ArgumentParser(description="Generate NEOS-style SegFormer semantic label maps.")
    parser.add_argument("--dataset_root", default="C:/Users/31133/Desktop/dataset1/CVUSA_processed_split")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--street_input", default="ground_rgb")
    parser.add_argument("--satellite_input", default="overhead_satellite")
    parser.add_argument("--street_output", default="street_neos_semantic")
    parser.add_argument("--satellite_output", default="overhead_satellite_neos_semantic")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    except ImportError as exc:
        raise SystemExit("Install transformers first: pip install transformers==4.57.1") from exc

    processor = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(args.device).eval()
    mapping = build_id_mapping(model.config.id2label)
    dataset_root = Path(args.dataset_root)

    for split in args.splits:
        split_root = dataset_root / split
        jobs = [
            (split_root / args.street_input, split_root / args.street_output, False),
            (split_root / args.satellite_input, split_root / args.satellite_output, True),
        ]
        for input_dir, output_dir, is_satellite in jobs:
            if not input_dir.exists():
                print(f"[Skip] missing input dir: {input_dir}")
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            paths = list(iter_images(input_dir))
            if args.limit is not None:
                paths = paths[: args.limit]
            print(f"[{split}] {input_dir.name}: {len(paths)} images -> {output_dir}")
            for path in tqdm(paths):
                out_path = output_dir / f"{path.stem}.png"
                if out_path.exists() and not args.overwrite:
                    continue
                image = read_rgb(path)
                orig_h, orig_w = image.shape[:2]
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    logits = model(**inputs).logits
                    logits = torch.nn.functional.interpolate(
                        logits,
                        size=(orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    raw = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
                raw = np.clip(raw, 0, len(mapping) - 1)
                labels = mapping[raw].copy()
                labels = apply_neos_rules(image, labels, is_satellite=is_satellite)
                cv2.imwrite(str(out_path), labels.astype(np.uint8))


if __name__ == "__main__":
    main()
