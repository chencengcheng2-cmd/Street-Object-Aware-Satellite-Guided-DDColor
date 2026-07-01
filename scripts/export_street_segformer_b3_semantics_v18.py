"""Export street-view semantic masks with a Cityscapes SegFormer-B3 teacher.

The exported labels follow the existing street DINO convention used by the
colorization code:
0 road/pavement
1 building
2 low vegetation / terrain
3 tree / high vegetation
4 car / vehicle
5 other
6 sky

This script is a teacher stage. The next stage trains a DINOv3 street head on
these masks, matching the satellite-side "teacher masks -> DINOv3 refinement"
workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

PALETTE = np.array(
    [
        [130, 130, 130],  # road
        [0, 90, 255],     # building
        [0, 220, 220],    # low vegetation
        [20, 170, 40],    # tree / vegetation
        [255, 230, 0],    # car / vehicle
        [220, 40, 40],    # other
        [80, 170, 255],   # sky
    ],
    dtype=np.uint8,
)


# Cityscapes 19-class id -> project 7-class id.
CITYSCAPES_TO_PROJECT = np.array(
    [
        0,  # road -> road
        0,  # sidewalk -> road/pavement
        1,  # building -> building
        1,  # wall -> building/built structure
        1,  # fence -> building/built structure
        5,  # pole -> other
        5,  # traffic light -> other
        5,  # traffic sign -> other
        3,  # vegetation -> tree/high vegetation
        2,  # terrain -> low vegetation/ground
        6,  # sky -> sky
        5,  # person -> other
        4,  # rider -> vehicle-related foreground
        4,  # car -> car
        4,  # truck -> car/vehicle
        4,  # bus -> car/vehicle
        4,  # train -> car/vehicle
        4,  # motorcycle -> car/vehicle
        4,  # bicycle -> car/vehicle
    ],
    dtype=np.uint8,
)


def iter_image_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir(), key=lambda p: p.stem):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]


class StreetPatchDataset(Dataset):
    def __init__(self, root: Path, split: str, image_dir: str):
        self.image_dir = root / split / image_dir
        if not self.image_dir.exists():
            raise FileNotFoundError(self.image_dir)
        self.items = list(iter_image_files(self.image_dir))
        if not self.items:
            raise RuntimeError(f"No street images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path = self.items[idx]
        image = read_rgb(path)
        sample_id = path.stem[:-5] if path.stem.endswith("_gray") else path.stem
        return {"id": sample_id, "image": image}


def collate(batch: List[dict]):
    return {
        "id": [item["id"] for item in batch],
        "image": [Image.fromarray(item["image"]) for item in batch],
        "shape": [item["image"].shape[:2] for item in batch],
    }


def export_split(args, split: str, processor, model, device: torch.device):
    dataset = StreetPatchDataset(Path(args.root), split, args.image_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    out_dir = Path(args.root) / split / args.output_dir
    vis_dir = Path(args.root) / split / f"{args.output_dir}_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    print(f"[SegFormer-B3] split={split} images={len(dataset)} -> {out_dir}", flush=True)
    for batch_idx, batch in enumerate(loader):
        inputs = processor(images=batch["image"], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=args.use_amp and device.type == "cuda"):
            logits = model(**inputs).logits
        for idx, sample_id in enumerate(batch["id"]):
            h, w = batch["shape"][idx]
            resized = torch.nn.functional.interpolate(
                logits[idx:idx + 1],
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            raw = resized.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)
            raw = np.clip(raw, 0, len(CITYSCAPES_TO_PROJECT) - 1)
            project = CITYSCAPES_TO_PROJECT[raw]
            cv2.imwrite(str(out_dir / f"{sample_id}.png"), project)
            cv2.imwrite(str(vis_dir / f"{sample_id}.png"), cv2.cvtColor(colorize(project), cv2.COLOR_RGB2BGR))
            exported += 1
        if batch_idx % args.log_interval == 0:
            print(f"[SegFormer-B3] {split}: {exported}/{len(dataset)}", flush=True)
    print(f"[SegFormer-B3] split={split} done count={exported}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="C:/Users/31133/Desktop/dataset1/CVUSA_processed_split")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--image_dir", default="ground_rgb")
    parser.add_argument("--output_dir", default="street_segformer_b3_semantic_v18")
    parser.add_argument("--model_name", default="nvidia/segformer-b3-finetuned-cityscapes-1024-1024")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--use_amp", action="store_true")
    args = parser.parse_args()

    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SegFormer-B3] loading {args.model_name} on {device}", flush=True)
    processor = SegformerImageProcessor.from_pretrained(args.model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_name).to(device).eval()
    for split in args.splits:
        export_split(args, split, processor, model, device)


if __name__ == "__main__":
    main()
