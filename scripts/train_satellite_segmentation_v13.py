"""Train a satellite semantic segmentation model from LabelMe annotations.

The script is designed for the user-labeled dataset under:
    C:/Users/31133/Desktop/segdataset

It does not modify the original dataset. LabelMe polygons are rasterized on the
fly and resized to the configured training size.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


PALETTE = np.array(
    [
        [255, 255, 255],  # road
        [0, 90, 255],     # building
        [0, 220, 220],    # grass
        [20, 170, 40],    # tree
        [255, 230, 0],    # car
        [220, 40, 40],    # other
    ],
    dtype=np.uint8,
)


@dataclass
class Sample:
    image_path: Path
    json_path: Path
    sample_id: str


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_labelme(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="gbk"))


def resolve_image_path(json_path: Path, data: dict) -> Path | None:
    same_stem = json_path.with_suffix(".png")
    if same_stem.exists():
        return same_stem

    image_path = data.get("imagePath")
    if image_path:
        candidate = (json_path.parent / image_path).resolve()
        if candidate.exists():
            return candidate

    root = json_path.parents[1] if len(json_path.parents) > 1 else json_path.parent
    matches = list(root.rglob(json_path.stem + ".png"))
    if matches:
        return matches[0]
    return None


def collect_samples(root: Path) -> List[Sample]:
    samples: List[Sample] = []
    missing = []
    for json_path in sorted(root.rglob("*.json")):
        data = read_labelme(json_path)
        image_path = resolve_image_path(json_path, data)
        if image_path is None:
            missing.append(str(json_path))
            continue
        samples.append(Sample(image_path=image_path, json_path=json_path, sample_id=json_path.stem))
    if missing:
        print(f"[Data] Warning: {len(missing)} JSON files have no matching image. First: {missing[:3]}")
    return samples


def split_samples(samples: List[Sample], val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    samples = samples[:]
    rng.shuffle(samples)
    n = len(samples)
    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    val = samples[:n_val]
    test = samples[n_val:n_val + n_test]
    train = samples[n_val + n_test:]
    return train, val, test


def rasterize_labelme(
    data: dict,
    class_to_id: Dict[str, int],
    label_aliases: Dict[str, str],
    default_label: str,
) -> np.ndarray:
    width = int(data.get("imageWidth", 0))
    height = int(data.get("imageHeight", 0))
    if width <= 0 or height <= 0:
        raise ValueError("LabelMe JSON must contain imageWidth and imageHeight")

    default_id = class_to_id[default_label]
    mask = Image.new("L", (width, height), default_id)
    draw = ImageDraw.Draw(mask)
    for shape in data.get("shapes", []):
        raw_label = str(shape.get("label", "")).strip()
        label = label_aliases.get(raw_label, raw_label)
        if label not in class_to_id:
            label = default_label
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        polygon = [(float(x), float(y)) for x, y in points]
        draw.polygon(polygon, fill=class_to_id[label])
    return np.array(mask, dtype=np.uint8)


class SatelliteSegDataset(Dataset):
    def __init__(self, samples: List[Sample], config: dict, split: str):
        self.samples = samples
        self.image_size = int(config["dataset"]["image_size"])
        self.classes = list(config["dataset"]["classes"])
        self.class_to_id = {name: i for i, name in enumerate(self.classes)}
        self.label_aliases = dict(config["dataset"].get("label_aliases", {}))
        self.default_label = config["dataset"].get("default_label", "other")
        self.split = split

    def __len__(self):
        return len(self.samples)

    def _load_pair(self, sample: Sample) -> Tuple[Image.Image, Image.Image]:
        img = Image.open(sample.image_path).convert("RGB")
        data = read_labelme(sample.json_path)
        mask_np = rasterize_labelme(data, self.class_to_id, self.label_aliases, self.default_label)
        mask = Image.fromarray(mask_np, mode="L")
        return img, mask

    def _augment(self, img: Image.Image, mask: Image.Image):
        if self.split != "train":
            return img, mask
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        k = random.randint(0, 3)
        if k:
            angle = 90 * k
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        if random.random() < 0.8:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            saturation = random.uniform(0.85, 1.15)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)
        return img, mask

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img, mask = self._load_pair(sample)
        img, mask = self._augment(img, mask)
        img = TF.resize(img, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return {
            "image": img_t,
            "mask": mask_t,
            "sample_id": sample.sample_id,
        }


class SegmentationModelWrapper(nn.Module):
    """Normalize torchvision segmentation model outputs to a logits tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            return out["out"]
        return out


def build_deeplabv3_model(num_classes: int, pretrained_encoder: bool) -> nn.Module:
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained_encoder else None
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
        aux_loss=False,
    )
    return SegmentationModelWrapper(model)


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, target):
        probs = logits.softmax(dim=1)
        target_oh = F.one_hot(target.clamp(0, self.num_classes - 1), self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * target_oh).sum(dims)
        denom = probs.sum(dims) + target_oh.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1 - dice.mean()


def compute_class_weights(loader: DataLoader, num_classes: int, device: torch.device):
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for batch in loader:
        mask = batch["mask"].view(-1)
        counts += torch.bincount(mask, minlength=num_classes).double()
    freq = counts / counts.sum().clamp_min(1)
    weights = 1.0 / torch.log(1.02 + freq)
    weights = weights / weights.mean()
    print("[Data] Class pixel counts:", counts.long().tolist())
    print("[Data] Class weights:", [round(float(x), 3) for x in weights])
    return weights.float().to(device)


@torch.no_grad()
def evaluate(model, loader, ce_loss, dice_loss, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        logits = model(x)
        loss = ce_loss(logits, y) + dice_loss(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        idx = (y.view(-1) * num_classes + pred.view(-1)).long()
        conf += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    tp = conf.diag().float()
    union = conf.sum(1).float() + conf.sum(0).float() - tp
    iou = tp / union.clamp_min(1)
    acc = tp.sum() / conf.sum().clamp_min(1)
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "miou": float(iou.mean().item()),
        "acc": float(acc.item()),
        "iou": [float(v.item()) for v in iou],
    }


def denormalize(img_t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_t.device).view(3, 1, 1)
    img = (img_t * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    return PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]


@torch.no_grad()
def save_visuals(model, loader, device, out_dir: Path, epoch: int, class_names: List[str]):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(loader))
    x = batch["image"].to(device)
    y = batch["mask"].to(device)
    pred = model(x).argmax(dim=1)
    n = min(6, x.size(0))
    tiles = []
    for i in range(n):
        img = denormalize(x[i])
        gt = colorize_mask(y[i].cpu().numpy())
        pr = colorize_mask(pred[i].cpu().numpy())
        tile = np.concatenate([img, gt, pr], axis=1)
        cv2.putText(tile, "RGB | GT | Pred", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(tile)
    grid = np.concatenate(tiles, axis=0)
    cv2.imwrite(str(out_dir / f"epoch_{epoch:03d}_preview.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def save_checkpoint(path: Path, model, optimizer, epoch, metrics, config, class_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "classes": class_names,
            "palette": PALETTE.tolist(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/satellite_segmentation_v13.yaml")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    exp_name = args.exp_name or config.get("experiment", {}).get("name", "satellite_segmentation_v13")
    root = Path(config["dataset"]["root"])
    class_names = list(config["dataset"]["classes"])
    num_classes = len(class_names)
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    if exp_name != "satellite_segmentation_v13":
        checkpoint_dir = checkpoint_dir.parent / exp_name
        output_dir = output_dir.parent / exp_name
    visual_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["dataset"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    if device.type == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    samples = collect_samples(root)
    if len(samples) < 10:
        raise RuntimeError(f"Too few samples found in {root}: {len(samples)}")
    train_samples, val_samples, test_samples = split_samples(
        samples,
        float(config["dataset"].get("val_ratio", 0.1)),
        float(config["dataset"].get("test_ratio", 0.1)),
        seed,
    )
    print(f"[Data] samples total={len(samples)} train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    print(f"[Data] classes={class_names}")

    split_path = output_dir / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "train": [s.sample_id for s in train_samples],
                "val": [s.sample_id for s in val_samples],
                "test": [s.sample_id for s in test_samples],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.smoke:
        train_samples = train_samples[:8]
        val_samples = val_samples[:4]
        config["training"]["epochs"] = 1
        config["training"]["batch_size"] = 2

    train_ds = SatelliteSegDataset(train_samples, config, split="train")
    val_ds = SatelliteSegDataset(val_samples, config, split="val")
    test_ds = SatelliteSegDataset(test_samples, config, split="test")
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    pretrained_encoder = bool(config["training"].get("pretrained_encoder", True))
    print(f"[Train] Architecture: deeplabv3_resnet50 pretrained_encoder={pretrained_encoder}")
    model = build_deeplabv3_model(num_classes=num_classes, pretrained_encoder=pretrained_encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"]))
    epochs = int(config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(config["training"].get("min_lr", 5e-6)))
    class_weights = compute_class_weights(train_loader, num_classes, device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(num_classes)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config["training"].get("use_amp", True)))

    best_miou = -1.0
    log_interval = int(config["training"].get("log_interval", 10))
    save_interval = int(config["training"].get("save_interval", 5))
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda" and bool(config["training"].get("use_amp", True))):
                logits = model(x)
                loss = ce_loss(logits, y) + dice_loss(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * x.size(0)
            if step % log_interval == 0:
                print(f"[Train] epoch={epoch+1}/{epochs} step={step}/{len(train_loader)} loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.6f}", flush=True)
        scheduler.step()
        train_loss = running / max(1, len(train_loader.dataset))
        val_metrics = evaluate(model, val_loader, ce_loss, dice_loss, device, num_classes)
        elapsed = time.time() - t0
        iou_text = " ".join(f"{name}:{iou:.3f}" for name, iou in zip(class_names, val_metrics["iou"]))
        print(
            f"[Val] epoch={epoch+1}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mIoU={val_metrics['miou']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} time={elapsed:.1f}s {iou_text}",
            flush=True,
        )

        latest = checkpoint_dir / "latest.pth"
        save_checkpoint(latest, model, optimizer, epoch, val_metrics, config, class_names)
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            save_checkpoint(checkpoint_dir / "best.pth", model, optimizer, epoch, val_metrics, config, class_names)
            save_visuals(model, val_loader, device, visual_dir, epoch + 1, class_names)
            print(f"[Checkpoint] New best mIoU={best_miou:.4f} -> {checkpoint_dir / 'best.pth'}", flush=True)
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth", model, optimizer, epoch, val_metrics, config, class_names)

    best_ckpt = torch.load(checkpoint_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, ce_loss, dice_loss, device, num_classes)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({"best_val_miou": best_miou, "test": test_metrics, "classes": class_names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_visuals(model, test_loader, device, visual_dir, 999, class_names)
    print(f"[Done] best_val_mIoU={best_miou:.4f}")
    print(f"[Done] test_mIoU={test_metrics['miou']:.4f} test_acc={test_metrics['acc']:.4f}")
    print(f"[Done] best_checkpoint={checkpoint_dir / 'best.pth'}")
    print(f"[Done] metrics={metrics_path}")


if __name__ == "__main__":
    main()
