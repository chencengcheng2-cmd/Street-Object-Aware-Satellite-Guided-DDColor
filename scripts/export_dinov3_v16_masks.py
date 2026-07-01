"""Export v16 DINOv3 masks for satellite and street-view branches."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_satellite_dinov3_multihead_segmentation_v16 import DinoV3MultiHeadSegmentation
from scripts.train_satellite_multihead_segmentation_v16 import MATERIAL_PALETTE, SEMANTIC_PALETTE as SAT_SEM_PALETTE, DETAIL_PALETTE as SAT_DETAIL_PALETTE
from scripts.train_street_dinov3_semantic_detail_v16 import StreetDinoV3Model, SEMANTIC_PALETTE as STREET_SEM_PALETTE, DETAIL_PALETTE as STREET_DETAIL_PALETTE


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_image_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir(), key=lambda p: p.stem):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_rgb_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = TF.resize(image, [image_size, image_size], interpolation=TF.InterpolationMode.BILINEAR)
    t = TF.to_tensor(image)
    return TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def colorize(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[np.clip(mask, 0, len(palette) - 1)]


class ImageExportDataset(Dataset):
    def __init__(self, directory: Path, image_size: int, strip_gray_suffix: bool):
        self.items = list(iter_image_files(directory))
        self.image_size = image_size
        self.strip_gray_suffix = strip_gray_suffix
        if not self.items:
            raise FileNotFoundError(f"No images found in {directory}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path = self.items[idx]
        stem = path.stem
        if self.strip_gray_suffix and stem.endswith("_gray"):
            stem = stem[:-5]
        return {
            "id": stem,
            "image": load_rgb_tensor(path, self.image_size),
        }


def collate(batch):
    return {
        "id": [x["id"] for x in batch],
        "image": torch.stack([x["image"] for x in batch]),
    }


def wait_for_checkpoint(path: Path, timeout_minutes: float):
    start = time.time()
    while not path.exists():
        if time.time() - start > timeout_minutes * 60:
            raise TimeoutError(f"Timed out waiting for checkpoint: {path}")
        print(f"[Export] Waiting for checkpoint: {path}", flush=True)
        time.sleep(30)


def load_satellite_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    config = load_yaml(config_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DinoV3MultiHeadSegmentation(
        backbone_name=config["model"].get("backbone", "vit_small_patch16_dinov3"),
        semantic_classes=len(config["dataset"]["semantic_classes"]),
        material_classes=len(config["dataset"]["material_classes"]),
        detail_classes=len(config["dataset"]["detail_classes"]),
        pretrained=False,
        freeze_backbone=True,
        unfreeze_last_blocks=0,
        decoder_channels=int(config["model"].get("decoder_channels", 192)),
        dropout=float(config["model"].get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, int(config["dataset"]["image_size"])


def load_street_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    config = load_yaml(config_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = StreetDinoV3Model(config["model"], len(config["dataset"]["semantic_classes"]), len(config["dataset"]["detail_classes"]))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, int(config["dataset"]["image_size"])


@torch.no_grad()
def export_satellite(args, device: torch.device):
    ckpt = Path(args.satellite_checkpoint)
    wait_for_checkpoint(ckpt, args.wait_minutes)
    model, image_size = load_satellite_model(Path(args.satellite_config), ckpt, device)
    root = Path(args.root)
    for split in args.splits:
        dataset = ImageExportDataset(root / split / args.satellite_image_dir, image_size, strip_gray_suffix=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate, pin_memory=device.type == "cuda")
        out_sem = root / split / args.satellite_semantic_output
        out_mat = root / split / args.satellite_material_output
        out_det = root / split / args.satellite_detail_output
        vis_sem = root / split / f"{args.satellite_semantic_output}_vis"
        vis_mat = root / split / f"{args.satellite_material_output}_vis"
        vis_det = root / split / f"{args.satellite_detail_output}_vis"
        for d in (out_sem, out_mat, out_det, vis_sem, vis_mat, vis_det):
            d.mkdir(parents=True, exist_ok=True)
        count = 0
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=args.use_amp and device.type == "cuda"):
                out = model(images)
            sem = out["semantic"].argmax(dim=1).cpu().numpy().astype(np.uint8)
            mat = out["material"].argmax(dim=1).cpu().numpy().astype(np.uint8)
            det = out["detail"].argmax(dim=1).cpu().numpy().astype(np.uint8)
            for image_id, s, m, d in zip(batch["id"], sem, mat, det):
                cv2.imwrite(str(out_sem / f"{image_id}.png"), s)
                cv2.imwrite(str(out_mat / f"{image_id}.png"), m)
                cv2.imwrite(str(out_det / f"{image_id}.png"), d)
                cv2.imwrite(str(vis_sem / f"{image_id}.png"), cv2.cvtColor(colorize(s, SAT_SEM_PALETTE), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(vis_mat / f"{image_id}.png"), cv2.cvtColor(colorize(m, MATERIAL_PALETTE), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(vis_det / f"{image_id}.png"), cv2.cvtColor(colorize(d, SAT_DETAIL_PALETTE), cv2.COLOR_RGB2BGR))
                count += 1
            if count % 500 == 0:
                print(f"[Export] satellite split={split} count={count}", flush=True)
        print(f"[Export] satellite split={split} done count={count}", flush=True)


@torch.no_grad()
def export_street(args, device: torch.device):
    ckpt = Path(args.street_checkpoint)
    wait_for_checkpoint(ckpt, args.wait_minutes)
    model, image_size = load_street_model(Path(args.street_config), ckpt, device)
    root = Path(args.root)
    for split in args.splits:
        dataset = ImageExportDataset(root / split / args.street_image_dir, image_size, strip_gray_suffix=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate, pin_memory=device.type == "cuda")
        out_sem = root / split / args.street_semantic_output
        out_det = root / split / args.street_detail_output
        vis_sem = root / split / f"{args.street_semantic_output}_vis"
        vis_det = root / split / f"{args.street_detail_output}_vis"
        for d in (out_sem, out_det, vis_sem, vis_det):
            d.mkdir(parents=True, exist_ok=True)
        count = 0
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=args.use_amp and device.type == "cuda"):
                out = model(images)
            sem = out["semantic"].argmax(dim=1).cpu().numpy().astype(np.uint8)
            det = out["detail"].argmax(dim=1).cpu().numpy().astype(np.uint8)
            for image_id, s, d in zip(batch["id"], sem, det):
                cv2.imwrite(str(out_sem / f"{image_id}.png"), s)
                cv2.imwrite(str(out_det / f"{image_id}.png"), d)
                cv2.imwrite(str(vis_sem / f"{image_id}.png"), cv2.cvtColor(colorize(s, STREET_SEM_PALETTE), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(vis_det / f"{image_id}.png"), cv2.cvtColor(colorize(d, STREET_DETAIL_PALETTE), cv2.COLOR_RGB2BGR))
                count += 1
            if count % 500 == 0:
                print(f"[Export] street split={split} count={count}", flush=True)
        print(f"[Export] street split={split} done count={count}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/czc/dataset1_v14/CVUSA_processed_split")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--branch", choices=["satellite", "street", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--wait_minutes", type=float, default=180)

    parser.add_argument("--satellite_config", default="configs/satellite_dinov3_multihead_segmentation_v16_server.yaml")
    parser.add_argument("--satellite_checkpoint", default="checkpoints/satellite_dinov3_multihead_segmentation_v16_server/best.pth")
    parser.add_argument("--satellite_image_dir", default="overhead_satellite")
    parser.add_argument("--satellite_semantic_output", default="overhead_satellite_dinov3_semantic_v16")
    parser.add_argument("--satellite_material_output", default="overhead_satellite_dinov3_material_v16")
    parser.add_argument("--satellite_detail_output", default="overhead_satellite_dinov3_detail_v16")

    parser.add_argument("--street_config", default="configs/street_dinov3_semantic_detail_v16_server.yaml")
    parser.add_argument("--street_checkpoint", default="checkpoints/street_dinov3_semantic_detail_v16_server/best.pth")
    parser.add_argument("--street_image_dir", default="ground_gray")
    parser.add_argument("--street_semantic_output", default="street_dinov3_semantic_v16")
    parser.add_argument("--street_detail_output", default="street_dinov3_detail_v16")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Export] device={device}", flush=True)
    if args.branch in ("satellite", "both"):
        export_satellite(args, device)
    if args.branch in ("street", "both"):
        export_street(args, device)
    print("[Export] done", flush=True)


if __name__ == "__main__":
    main()
