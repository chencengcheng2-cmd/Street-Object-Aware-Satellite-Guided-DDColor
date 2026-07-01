"""Train DINOv3 street-view semantic/detail heads for v16.

The semantic target uses existing street_dino_semantic_v12 pseudo labels.
The detail target is generated from grayscale structure inside road regions.
No ground-truth RGB street color is used here.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

SEMANTIC_PALETTE = np.array(
    [
        [130, 130, 130],  # road
        [0, 90, 255],     # building
        [0, 220, 220],    # low vegetation
        [20, 170, 40],    # tree
        [255, 230, 0],    # car
        [220, 40, 40],    # other
        [80, 170, 255],   # sky
    ],
    dtype=np.uint8,
)

DETAIL_PALETTE = np.array(
    [
        [0, 0, 0],         # none
        [255, 255, 255],   # road_edge
        [255, 230, 0],     # linear_marking
        [255, 140, 0],     # transverse_marking
        [255, 80, 220],    # road_symbol_marking
    ],
    dtype=np.uint8,
)


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    sample_id: str


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_image_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir(), key=lambda p: p.stem):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def semantic_id_from_gray_stem(stem: str) -> str:
    return stem[:-5] if stem.endswith("_gray") else stem


def collect_samples(root: Path, split: str, image_dir: str, semantic_dir: str) -> List[Sample]:
    img_dir = root / split / image_dir
    sem_dir = root / split / semantic_dir
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)
    if not sem_dir.exists():
        raise FileNotFoundError(sem_dir)
    samples: List[Sample] = []
    for image_path in iter_image_files(img_dir):
        sem_id = semantic_id_from_gray_stem(image_path.stem)
        label_path = sem_dir / f"{sem_id}.png"
        if not label_path.exists():
            continue
        samples.append(Sample(image_path=image_path, label_path=label_path, sample_id=sem_id))
    if not samples:
        raise RuntimeError(f"No matched street samples for split={split}")
    return samples


def classify_marking_components(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    detail = np.zeros((h, w), dtype=np.uint8)
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.max() == 0:
        return detail
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    image_area = float(h * w)
    for label_id in range(1, num_labels):
        x, y, bw, bh, area = stats[label_id]
        if area < 4:
            continue
        short = max(1, min(bw, bh))
        long = max(bw, bh)
        elongation = long / short
        fill = area / max(1, bw * bh)
        component = labels == label_id
        if elongation >= 3.0 and area <= 0.018 * image_area:
            detail[component] = 2
        elif bw >= 1.8 * bh and area <= 0.035 * image_area:
            detail[component] = 3
        elif area >= 10 and fill <= 0.82:
            detail[component] = 4
        else:
            detail[component] = 2
    return detail


def make_detail_label(gray: np.ndarray, semantic: np.ndarray, road_label: int = 0) -> np.ndarray:
    h, w = gray.shape[:2]
    road = semantic == road_label
    lower = np.zeros((h, w), dtype=bool)
    lower[int(h * 0.32):, :] = True
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 135) > 0
    edges = cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1).astype(bool)
    if road.any():
        bright_thr = max(110, int(np.percentile(gray[road], 70)))
    else:
        bright_thr = 125
    local = cv2.absdiff(gray, cv2.GaussianBlur(gray, (17, 17), 0))
    candidate = road & lower & edges & ((gray >= bright_thr) | (local > 18))
    detail = np.zeros((h, w), dtype=np.uint8)
    detail[road & edges] = 1
    marks = classify_marking_components(candidate)
    detail[marks > 0] = marks[marks > 0]
    return detail


class StreetDinoDataset(Dataset):
    def __init__(self, samples: List[Sample], config: dict, split: str):
        self.samples = samples
        self.image_size = int(config["dataset"]["image_size"])
        self.split = split

    def __len__(self):
        return len(self.samples)

    def _augment(self, img: Image.Image, semantic: Image.Image):
        if self.split != "train":
            return img, semantic
        if random.random() < 0.5:
            img = TF.hflip(img)
            semantic = TF.hflip(semantic)
        if random.random() < 0.6:
            img = TF.adjust_brightness(img, random.uniform(0.88, 1.12))
            img = TF.adjust_contrast(img, random.uniform(0.88, 1.12))
        return img, semantic

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample.image_path).convert("RGB")
        semantic = Image.open(sample.label_path).convert("L")
        img, semantic = self._augment(img, semantic)
        img = TF.resize(img, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
        semantic = TF.resize(semantic, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
        img_np = np.array(img, dtype=np.uint8)
        sem_np = np.array(semantic, dtype=np.uint8)
        gray_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        detail_np = make_detail_label(gray_np, sem_np, road_label=0)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return {
            "image": img_t,
            "semantic": torch.from_numpy(sem_np.astype(np.int64)).clamp(0, 6),
            "detail": torch.from_numpy(detail_np.astype(np.int64)).clamp(0, 4),
            "sample_id": sample.sample_id,
        }


class ConvSegHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

    def forward(self, x, size):
        return F.interpolate(self.net(x), size=size, mode="bilinear", align_corners=False)


class StreetDinoV3Model(nn.Module):
    def __init__(self, cfg: dict, num_semantic: int, num_detail: int):
        super().__init__()
        self.backbone = timm.create_model(cfg.get("backbone", "vit_small_patch16_dinov3"), pretrained=bool(cfg.get("pretrained", True)), num_classes=0)
        self.embed_dim = int(getattr(self.backbone, "embed_dim"))
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        self.patch_size = int(self.backbone.patch_embed.patch_size[0])
        if bool(cfg.get("freeze_backbone", True)):
            for p in self.backbone.parameters():
                p.requires_grad = False
        unfreeze_last_blocks = int(cfg.get("unfreeze_last_blocks", 0))
        if unfreeze_last_blocks > 0 and hasattr(self.backbone, "blocks"):
            for block in self.backbone.blocks[-unfreeze_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
        hidden = int(cfg.get("decoder_channels", 192))
        dropout = float(cfg.get("dropout", 0.1))
        self.semantic_head = ConvSegHead(self.embed_dim, hidden, num_semantic, dropout)
        self.detail_head = ConvSegHead(self.embed_dim, hidden, num_detail, dropout)

    def _tokens_to_map(self, tokens, input_hw):
        h, w = input_hw
        gh, gw = h // self.patch_size, w // self.patch_size
        expected = gh * gw
        patch_tokens = tokens[:, self.num_prefix_tokens:, :]
        if patch_tokens.shape[1] != expected:
            patch_tokens = tokens[:, -expected:, :]
        return patch_tokens.transpose(1, 2).reshape(tokens.shape[0], self.embed_dim, gh, gw)

    def forward(self, x):
        input_hw = (x.shape[-2], x.shape[-1])
        tokens = self.backbone.forward_features(x)
        if isinstance(tokens, dict):
            tokens = tokens.get("x_norm_patchtokens", None) or tokens.get("x", None)
            if tokens is None:
                raise RuntimeError("Unsupported DINOv3 feature dict output")
        feat = self._tokens_to_map(tokens, input_hw)
        return {
            "semantic": self.semantic_head(feat, input_hw),
            "detail": self.detail_head(feat, input_hw),
        }


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, target):
        probs = logits.softmax(dim=1)
        target_oh = F.one_hot(target.clamp(0, self.num_classes - 1), self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * target_oh).sum(dims)
        denom = probs.sum(dims) + target_oh.sum(dims)
        return 1 - ((2 * inter + self.eps) / (denom + self.eps)).mean()


def compute_weights(loader, key: str, num_classes: int, device):
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for batch in loader:
        counts += torch.bincount(batch[key].view(-1), minlength=num_classes).double()
    freq = counts / counts.sum().clamp_min(1)
    weights = 1.0 / torch.log(1.02 + freq)
    weights = weights / weights.mean()
    print(f"[Data] {key} pixel counts:", counts.long().tolist())
    print(f"[Data] {key} class weights:", [round(float(x), 3) for x in weights])
    return weights.float().to(device)


@torch.no_grad()
def evaluate(model, loader, losses, loss_weights, device, class_counts):
    model.eval()
    total = 0.0
    confs = {k: torch.zeros((n, n), dtype=torch.int64, device=device) for k, n in class_counts.items()}
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        targets = {k: batch[k].to(device, non_blocking=True) for k in class_counts}
        logits = model(x)
        loss = 0.0
        for key, target in targets.items():
            loss = loss + loss_weights[key] * (losses[key]["ce"](logits[key], target) + losses[key]["dice"](logits[key], target))
            pred = logits[key].argmax(dim=1)
            idx = (target.view(-1) * class_counts[key] + pred.view(-1)).long()
            confs[key] += torch.bincount(idx, minlength=class_counts[key] ** 2).reshape(class_counts[key], class_counts[key])
        total += float(loss.item()) * x.size(0)
    metrics = {"loss": total / max(1, len(loader.dataset))}
    for key, conf in confs.items():
        tp = conf.diag().float()
        union = conf.sum(1).float() + conf.sum(0).float() - tp
        iou = tp / union.clamp_min(1)
        metrics[f"{key}_miou"] = float(iou.mean().item())
        metrics[f"{key}_iou"] = [float(v.item()) for v in iou]
    metrics["score"] = metrics["semantic_miou"] + 0.25 * metrics["detail_miou"]
    return metrics


def colorize(mask, palette):
    return palette[np.clip(mask, 0, len(palette) - 1)]


@torch.no_grad()
def save_visuals(model, loader, device, out_dir: Path, epoch: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    batch = next(iter(loader))
    x = batch["image"].to(device)
    out = model(x)
    sem_pred = out["semantic"].argmax(dim=1).cpu().numpy()
    det_pred = out["detail"].argmax(dim=1).cpu().numpy()
    sem_gt = batch["semantic"].numpy()
    det_gt = batch["detail"].numpy()
    n = min(4, x.size(0))
    tiles = []
    for i in range(n):
        img = x[i].detach().cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_np = ((img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        row1 = np.concatenate([img_np, colorize(sem_gt[i], SEMANTIC_PALETTE), colorize(sem_pred[i], SEMANTIC_PALETTE)], axis=1)
        row2 = np.concatenate([img_np, colorize(det_gt[i], DETAIL_PALETTE), colorize(det_pred[i], DETAIL_PALETTE)], axis=1)
        tiles.append(np.concatenate([row1, row2], axis=0))
    grid = np.concatenate(tiles, axis=0)
    cv2.imwrite(str(out_dir / f"epoch_{epoch:03d}_preview.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def save_checkpoint(path, model, optimizer, epoch, metrics, config, class_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "classes": class_names,
            "palettes": {"semantic": SEMANTIC_PALETTE.tolist(), "detail": DETAIL_PALETTE.tolist()},
        },
        path,
    )


def build_optimizer(model, config):
    lr = float(config["training"]["lr"])
    backbone_lr = float(config["training"].get("backbone_lr", lr * 0.05))
    wd = float(config["training"]["weight_decay"])
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if name.startswith("backbone.") else head).append(p)
    groups = []
    if head:
        groups.append({"params": head, "lr": lr})
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/street_dinov3_semantic_detail_v16_server.yaml")
    parser.add_argument("--exp_name", default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    exp_name = args.exp_name or config["experiment"]["name"]
    root = Path(config["dataset"]["root"])
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    if exp_name != config["experiment"]["name"]:
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

    train_samples = collect_samples(root, config["dataset"]["train_split"], config["dataset"]["image_dir"], config["dataset"]["semantic_dir"])
    val_samples = collect_samples(root, config["dataset"]["val_split"], config["dataset"]["image_dir"], config["dataset"]["semantic_dir"])
    print(f"[Data] train={len(train_samples)} val={len(val_samples)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    if device.type == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    train_ds = StreetDinoDataset(train_samples, config, "train")
    val_ds = StreetDinoDataset(val_samples, config, "val")
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")

    model = StreetDinoV3Model(config["model"], len(config["dataset"]["semantic_classes"]), len(config["dataset"]["detail_classes"])).to(device)
    print(f"[Train] parameters trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M total={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    class_counts = {"semantic": len(config["dataset"]["semantic_classes"]), "detail": len(config["dataset"]["detail_classes"])}
    losses = {
        "semantic": {"ce": nn.CrossEntropyLoss(weight=compute_weights(train_loader, "semantic", class_counts["semantic"], device)), "dice": DiceLoss(class_counts["semantic"])},
        "detail": {"ce": nn.CrossEntropyLoss(weight=compute_weights(train_loader, "detail", class_counts["detail"], device)), "dice": DiceLoss(class_counts["detail"])},
    }
    lw_cfg = config["training"]["loss_weights"]
    loss_weights = {"semantic": float(lw_cfg["semantic"]), "detail": float(lw_cfg["detail"])}
    optimizer = build_optimizer(model, config)
    epochs = int(config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(config["training"]["min_lr"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config["training"]["use_amp"]))

    best_score = -1.0
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            targets = {k: batch[k].to(device, non_blocking=True) for k in class_counts}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda" and bool(config["training"]["use_amp"])):
                logits = model(x)
                parts = {}
                loss = 0.0
                for key, target in targets.items():
                    part = losses[key]["ce"](logits[key], target) + losses[key]["dice"](logits[key], target)
                    parts[key] = part
                    loss = loss + loss_weights[key] * part
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss epoch={epoch} step={step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * x.size(0)
            if step % int(config["training"]["log_interval"]) == 0:
                lrs = ",".join(f"{g['lr']:.6f}" for g in optimizer.param_groups)
                print(f"[Train] epoch={epoch+1}/{epochs} step={step}/{len(train_loader)} loss={loss.item():.4f} sem={parts['semantic'].item():.4f} det={parts['detail'].item():.4f} lr={lrs}", flush=True)
        scheduler.step()
        metrics = evaluate(model, val_loader, losses, loss_weights, device, class_counts)
        print(f"[Val] epoch={epoch+1}/{epochs} train_loss={running/max(1,len(train_loader.dataset)):.4f} val_loss={metrics['loss']:.4f} score={metrics['score']:.4f} sem_mIoU={metrics['semantic_miou']:.4f} det_mIoU={metrics['detail_miou']:.4f} time={time.time()-t0:.1f}s", flush=True)
        save_checkpoint(checkpoint_dir / "latest.pth", model, optimizer, epoch, metrics, config, {"semantic": config["dataset"]["semantic_classes"], "detail": config["dataset"]["detail_classes"]})
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            save_checkpoint(checkpoint_dir / "best.pth", model, optimizer, epoch, metrics, config, {"semantic": config["dataset"]["semantic_classes"], "detail": config["dataset"]["detail_classes"]})
            save_visuals(model, val_loader, device, visual_dir, epoch + 1)
            print(f"[Checkpoint] New best score={best_score:.4f} -> {checkpoint_dir / 'best.pth'}", flush=True)
        if (epoch + 1) % int(config["training"]["save_interval"]) == 0:
            save_checkpoint(checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth", model, optimizer, epoch, metrics, config, {"semantic": config["dataset"]["semantic_classes"], "detail": config["dataset"]["detail_classes"]})

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"best_val_score": best_score}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] best_val_score={best_score:.4f}")
    print(f"[Done] best_checkpoint={checkpoint_dir / 'best.pth'}")
    print(f"[Done] metrics={metrics_path}")


if __name__ == "__main__":
    main()
