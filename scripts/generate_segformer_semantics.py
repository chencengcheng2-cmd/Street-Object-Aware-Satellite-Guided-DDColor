"""Generate project semantic masks with SegFormer.

Output label IDs:
0 sky/no-match, 1 road, 2 vegetation, 3 building, 4 water, 5 other.

The script uses an ADE20K SegFormer checkpoint by default. It is good enough for
street-view masks and a practical baseline for satellite masks; satellite masks
can later be replaced by a remote-sensing SegFormer without changing training.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_CLASSES = {
    "sky": 0,
    "road": 1,
    "vegetation": 2,
    "building": 3,
    "water": 4,
    "other": 5,
}

KEYWORDS = {
    PROJECT_CLASSES["sky"]: ["sky"],
    PROJECT_CLASSES["road"]: ["road", "runway", "path", "sidewalk", "bridge"],
    PROJECT_CLASSES["vegetation"]: ["tree", "grass", "plant", "field", "earth", "mountain", "hill", "flower"],
    PROJECT_CLASSES["building"]: ["building", "house", "skyscraper", "wall", "fence", "roof", "tower"],
    PROJECT_CLASSES["water"]: ["water", "sea", "river", "lake", "pool", "waterfall"],
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_id_mapping(id2label: dict[int, str]) -> np.ndarray:
    max_id = max(int(k) for k in id2label.keys())
    mapping = np.full(max_id + 1, PROJECT_CLASSES["other"], dtype=np.uint8)
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


def main():
    parser = argparse.ArgumentParser(description="Generate SegFormer semantic label maps for CVUSA splits.")
    parser.add_argument("--dataset_root", default="C:/Users/31133/Desktop/dataset1/CVUSA_processed_split")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--street_input", default="ground_rgb")
    parser.add_argument("--satellite_input", default="overhead_satellite")
    parser.add_argument("--street_output", default="street_semantic")
    parser.add_argument("--satellite_output", default="overhead_satellite_semantic")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional max images per input directory for testing.")
    args = parser.parse_args()

    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    except ImportError as exc:
        raise SystemExit(
            "transformers is required for SegFormer masks. Install with: pip install transformers>=4.40"
        ) from exc

    processor = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(args.device).eval()
    mapping = build_id_mapping(model.config.id2label)
    dataset_root = Path(args.dataset_root)

    for split in args.splits:
        split_root = dataset_root / split
        jobs = [
            (split_root / args.street_input, split_root / args.street_output),
            (split_root / args.satellite_input, split_root / args.satellite_output),
        ]
        for input_dir, output_dir in jobs:
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
                project_labels = mapping[raw]
                cv2.imwrite(str(out_path), project_labels)


if __name__ == "__main__":
    main()
