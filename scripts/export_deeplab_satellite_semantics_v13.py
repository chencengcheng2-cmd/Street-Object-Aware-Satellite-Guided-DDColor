"""Export DeepLabV3 satellite semantic labels for CVUSA splits.

The output labels are NEOS-compatible ids:
0 sky, 1 road, 2 building, 3 grass/low vegetation, 4 tree, 5 car, 6 other.
For satellite images, "other" is removed by replacing it with the strongest
non-other DeepLabV3 class before mapping to NEOS ids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50


DEEPLAB_TO_NEOS = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)


class SegmentationModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            return out["out"]
        return out


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint.get("classes", ["road", "building", "grass", "tree", "car", "other"])
    model_inner = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=len(classes),
        aux_loss=False,
    )
    model = SegmentationModelWrapper(model_inner).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, list(classes)


def preprocess(rgb: np.ndarray, image_size: int) -> torch.Tensor:
    pil = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    resized = TF.resize(pil, [image_size, image_size], interpolation=TF.InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor


@torch.inference_mode()
def predict_label(model, classes, rgb: np.ndarray, image_size: int, device: torch.device) -> np.ndarray:
    h, w = rgb.shape[:2]
    tensor = preprocess(rgb, image_size).unsqueeze(0).to(device)
    logits = model(tensor)[0]
    pred = logits.argmax(dim=0)
    if "other" in classes and len(classes) > 1:
        other_idx = classes.index("other")
        non_other_logits = logits.clone()
        non_other_logits[other_idx] = -1e9
        second_choice = non_other_logits.argmax(dim=0)
        pred = torch.where(pred == other_idx, second_choice, pred)
    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
    mapped = DEEPLAB_TO_NEOS[np.clip(pred_np, 0, len(DEEPLAB_TO_NEOS) - 1)]
    if mapped.shape[:2] != (h, w):
        mapped = cv2.resize(mapped, (w, h), interpolation=cv2.INTER_NEAREST)
    return mapped.astype(np.uint8)


def iter_images(directory: Path):
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        yield from directory.glob(suffix)


def export_split(model, classes, dataset_root: Path, split: str, out_dirname: str, image_size: int, device: torch.device):
    in_dir = dataset_root / split / "overhead_satellite"
    out_dir = dataset_root / split / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted({p for p in iter_images(in_dir)})
    print(f"[DeepLab export] split={split} images={len(paths)} out={out_dir}", flush=True)
    for idx, path in enumerate(paths, start=1):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[DeepLab export] skip unreadable: {path}", flush=True)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        labels = predict_label(model, classes, rgb, image_size, device)
        if labels.shape[:2] != (256, 256):
            labels = cv2.resize(labels, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / f"{path.stem}.png"), labels)
        if idx % 100 == 0 or idx == len(paths):
            print(f"[DeepLab export] {split}: {idx}/{len(paths)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="C:/Users/31133/Desktop/dataset1/CVUSA_processed_split")
    parser.add_argument("--checkpoint", default="checkpoints/satellite_segmentation_v13/best.pth")
    parser.add_argument("--out_dirname", default="overhead_satellite_deeplab_v13")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(Path(args.checkpoint), device)
    print(f"[DeepLab export] device={device} classes={classes}", flush=True)
    for split in args.splits:
        export_split(model, classes, Path(args.dataset_root), split, args.out_dirname, args.image_size, device)


if __name__ == "__main__":
    main()
