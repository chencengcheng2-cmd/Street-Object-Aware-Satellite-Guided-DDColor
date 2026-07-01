"""Train v12 DINO semantic distillation heads.

Scheme A:
    existing street / satellite / polar semantic masks are teacher labels.
    DINO backbones are frozen. Only lightweight semantic heads are trained.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dino_semantic_distillation import DinoSemanticDistillationModel
from src.utils import load_config, set_seed


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

NEOS_PALETTE = np.array(
    [
        [135, 206, 235],  # 0 sky / unknown
        [255, 255, 255],  # 1 road / hard surface
        [0, 0, 255],      # 2 building / object
        [0, 255, 255],    # 3 low vegetation
        [0, 128, 0],      # 4 tree / vegetation
        [255, 255, 0],    # 5 car / object
        [255, 0, 0],      # 6 clutter / other
    ],
    dtype=np.uint8,
)


def iter_image_files(directory: Path):
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_shared_index(directory: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for path in iter_image_files(directory):
        stem = path.stem
        parts = stem.rsplit("_", 1)
        key = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
        files.setdefault(key, path)
    return files


def rgb_mask_to_labels(rgb: np.ndarray) -> np.ndarray:
    """Map an RGB semantic visualization to nearest NEOS class IDs."""
    rgb = rgb.astype(np.int16)
    palette = NEOS_PALETTE.astype(np.int16)
    dist = ((rgb[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=-1)
    return dist.argmin(axis=-1).astype(np.int64)


def load_rgb(path: Path, size: Tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def load_label(path: Path, size: Tuple[int, int], from_rgb_palette: bool = False) -> np.ndarray:
    if from_rgb_palette:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load RGB label image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
        return rgb_mask_to_labels(image)
    label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise ValueError(f"Failed to load label image: {path}")
    return cv2.resize(label, size, interpolation=cv2.INTER_NEAREST).astype(np.int64)


class DinoSemanticTeacherDataset(Dataset):
    """Dataset with RGB inputs and existing teacher semantic masks."""

    def __init__(
        self,
        dataset_root: str,
        split: str,
        street_semantic_dirname: str,
        satellite_semantic_dirname: str,
        image_size: Tuple[int, int] = (256, 256),
        polar_size: Tuple[int, int] = (512, 256),
        max_samples: Optional[int] = None,
    ):
        self.root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.polar_size = polar_size
        split_dir = self.root / split

        self.rgb_dir = split_dir / "ground_rgb"
        self.sat_dir = split_dir / "overhead_satellite"
        self.polar_dir = split_dir / "overhead_polar"
        self.street_sem_dir = split_dir / street_semantic_dirname
        self.sat_sem_dir = split_dir / satellite_semantic_dirname
        self.polar_sem_dir = split_dir / "overhead_polar_seg"

        required = [
            self.rgb_dir,
            self.sat_dir,
            self.polar_dir,
            self.street_sem_dir,
            self.sat_sem_dir,
            self.polar_sem_dir,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required directories: {missing}")

        self.samples = self._build_index()
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]
        print(f"[DINO v12] Loaded {len(self.samples)} {split} samples.")

    def _build_index(self) -> List[Dict[str, Path]]:
        rgb_files = {path.stem: path for path in iter_image_files(self.rgb_dir)}
        street_sem_files = {path.stem: path for path in iter_image_files(self.street_sem_dir)}
        sat_files = build_shared_index(self.sat_dir)
        polar_files = build_shared_index(self.polar_dir)
        sat_sem_files = build_shared_index(self.sat_sem_dir)
        polar_sem_files = {path.stem: path for path in iter_image_files(self.polar_sem_dir)}

        samples: List[Dict[str, Path]] = []
        for file_id, rgb_path in rgb_files.items():
            parts = file_id.rsplit("_", 1)
            if len(parts) != 2:
                continue
            panorama_id = parts[0]
            street_sem = street_sem_files.get(file_id)
            sat = sat_files.get(panorama_id)
            polar = polar_files.get(panorama_id)
            sat_sem = sat_sem_files.get(panorama_id)
            polar_sem = polar_sem_files.get(file_id) or polar_sem_files.get(panorama_id)
            if not all([street_sem, sat, polar, sat_sem, polar_sem]):
                continue
            samples.append(
                {
                    "file_id": file_id,
                    "rgb": rgb_path,
                    "satellite": sat,
                    "polar": polar,
                    "street_semantic": street_sem,
                    "satellite_semantic": sat_sem,
                    "polar_semantic": polar_sem,
                }
            )
        samples.sort(key=lambda item: item["file_id"])
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        street_rgb = load_rgb(sample["rgb"], self.image_size)
        satellite_rgb = load_rgb(sample["satellite"], self.image_size)
        polar_rgb = load_rgb(sample["polar"], self.polar_size)
        street_label = load_label(sample["street_semantic"], self.image_size)
        satellite_label = load_label(sample["satellite_semantic"], self.image_size)
        polar_label = load_label(sample["polar_semantic"], self.polar_size, from_rgb_palette=True)

        return {
            "file_id": sample["file_id"],
            "street_rgb": torch.from_numpy(street_rgb).permute(2, 0, 1).float() / 255.0,
            "satellite_rgb": torch.from_numpy(satellite_rgb).permute(2, 0, 1).float() / 255.0,
            "polar_rgb": torch.from_numpy(polar_rgb).permute(2, 0, 1).float() / 255.0,
            "street_label": torch.from_numpy(street_label).long(),
            "satellite_label": torch.from_numpy(satellite_label).long(),
            "polar_label": torch.from_numpy(polar_label).long(),
        }


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    valid = (labels >= 0) & (labels < num_classes)
    correct = ((pred == labels) & valid).sum().item()
    total = valid.sum().item()
    ious = []
    for cls in range(num_classes):
        pred_c = (pred == cls) & valid
        label_c = (labels == cls) & valid
        union = (pred_c | label_c).sum().item()
        if union == 0:
            continue
        inter = (pred_c & label_c).sum().item()
        ious.append(inter / max(union, 1))
    return {
        "pixel_acc": correct / max(total, 1),
        "miou": float(np.mean(ious)) if ious else 0.0,
    }


def merge_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {"pixel_acc": 0.0, "miou": 0.0}
    keys = metric_list[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in metric_list])) for key in keys}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def run_epoch(
    model: DinoSemanticDistillationModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    num_classes: int,
    weights: Dict[str, float],
    use_amp: bool,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    losses = []
    metrics = []
    scaler = torch.amp.GradScaler("cuda", enabled=train and use_amp and device.type == "cuda")

    for batch in loader:
        street_rgb = batch["street_rgb"].to(device, non_blocking=True)
        satellite_rgb = batch["satellite_rgb"].to(device, non_blocking=True)
        polar_rgb = batch["polar_rgb"].to(device, non_blocking=True)
        street_label = batch["street_label"].to(device, non_blocking=True)
        satellite_label = batch["satellite_label"].to(device, non_blocking=True)
        polar_label = batch["polar_label"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            output = model(street_rgb=street_rgb, satellite_rgb=satellite_rgb, polar_rgb=polar_rgb)
            loss_street = F.cross_entropy(output["street_logits"], street_label)
            loss_sat = F.cross_entropy(output["satellite_logits"], satellite_label)
            loss_polar = F.cross_entropy(output["polar_logits"], polar_label)
            loss = (
                weights["street"] * loss_street
                + weights["satellite"] * loss_sat
                + weights["polar"] * loss_polar
            )

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            part_metrics = []
            part_metrics.append(compute_metrics(output["street_logits"], street_label, num_classes))
            part_metrics.append(compute_metrics(output["satellite_logits"], satellite_label, num_classes))
            part_metrics.append(compute_metrics(output["polar_logits"], polar_label, num_classes))
            metrics.append(merge_metrics(part_metrics))

    merged = merge_metrics(metrics)
    merged["loss"] = float(np.mean(losses)) if losses else 0.0
    return merged


def main():
    parser = argparse.ArgumentParser(description="Train DINO semantic distillation heads for v12.")
    parser.add_argument("--config", default="configs/dino_semantic_distill_v12.yaml")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dino_pretrained", choices=["true", "false"], default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("training", {}).get("seed", 42)))

    exp_name = args.exp_name or config.get("experiment", {}).get("name", "dino_semantic_distill_v12")
    checkpoint_dir = REPO_ROOT / config.get("paths", {}).get("checkpoint_base_dir", "checkpoints") / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DINO v12] Device: {device}")

    dataset_cfg = config["dataset"]
    semantic_cfg = config.get("semantic", {})
    model_cfg = config["model"]
    train_cfg = config["training"]
    if args.dino_pretrained is not None:
        model_cfg["dino_pretrained"] = args.dino_pretrained == "true"

    image_size = tuple(model_cfg.get("image_size", [256, 256]))
    polar_size_hw = tuple(model_cfg.get("polar_size", [256, 512]))
    polar_size = (polar_size_hw[1], polar_size_hw[0])
    image_size_cv = (image_size[1], image_size[0])

    train_set = DinoSemanticTeacherDataset(
        dataset_cfg["root"],
        "train",
        semantic_cfg.get("street_dirname", "street_neos_semantic"),
        semantic_cfg.get("satellite_dirname", "overhead_satellite_neos_semantic"),
        image_size=image_size_cv,
        polar_size=polar_size,
        max_samples=args.max_train_samples or train_cfg.get("max_train_samples"),
    )
    val_set = DinoSemanticTeacherDataset(
        dataset_cfg["root"],
        "val",
        semantic_cfg.get("street_dirname", "street_neos_semantic"),
        semantic_cfg.get("satellite_dirname", "overhead_satellite_neos_semantic"),
        image_size=image_size_cv,
        polar_size=polar_size,
        max_samples=args.max_val_samples or train_cfg.get("max_val_samples"),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )

    model = DinoSemanticDistillationModel(
        num_classes=int(model_cfg.get("num_classes", 7)),
        dino_model_name=model_cfg.get("dino_model_name", "vit_small_patch16_224.dino"),
        dino_pretrained=bool(model_cfg.get("dino_pretrained", True)),
        feature_channels=int(model_cfg.get("feature_channels", 128)),
        head_hidden_channels=int(model_cfg.get("head_hidden_channels", 128)),
        share_overhead_backbone=bool(model_cfg.get("share_overhead_backbone", True)),
        freeze_dino=bool(model_cfg.get("freeze_dino", True)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    start_epoch = 0
    best_miou = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_miou = float(checkpoint.get("metrics", {}).get("val_miou", -1.0))

    weights = config.get("loss", {})
    loss_weights = {
        "street": float(weights.get("street_weight", 1.0)),
        "satellite": float(weights.get("satellite_weight", 1.0)),
        "polar": float(weights.get("polar_weight", 1.0)),
    }

    use_amp = bool(train_cfg.get("use_amp", True))
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 20))
    history = []

    for epoch in range(start_epoch, epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, device, model.num_classes, loss_weights, use_amp)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, device, model.num_classes, loss_weights, use_amp=False)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_miou": train_metrics["miou"],
            "train_pixel_acc": train_metrics["pixel_acc"],
            "val_loss": val_metrics["loss"],
            "val_miou": val_metrics["miou"],
            "val_pixel_acc": val_metrics["pixel_acc"],
        }
        history.append(row)
        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"train_loss={row['train_loss']:.4f} train_miou={row['train_miou']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_miou={row['val_miou']:.4f} "
            f"val_acc={row['val_pixel_acc']:.4f}",
            flush=True,
        )

        save_checkpoint(checkpoint_dir / "latest.pth", model, optimizer, epoch, row, config)
        if row["val_miou"] > best_miou:
            best_miou = row["val_miou"]
            save_checkpoint(checkpoint_dir / "best.pth", model, optimizer, epoch, row, config)

        with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"[DINO v12] Done. Best val mIoU: {best_miou:.4f}")
    print(f"[DINO v12] Best checkpoint: {checkpoint_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
