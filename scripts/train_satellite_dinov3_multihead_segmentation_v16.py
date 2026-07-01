"""Train DINOv3 + multi-head satellite segmentation for v16.

This stage uses the 400 manually labeled satellite images as strong supervision
for land-cover semantics and weak RGB/edge labels for material/detail heads.
The default mode freezes DINOv3 and trains only the segmentation heads, which is
safer for the small labeled set.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_satellite_multihead_segmentation_v16 import (  # noqa: E402
    DETAIL_PALETTE,
    MATERIAL_PALETTE,
    SEMANTIC_PALETTE,
    DiceLoss,
    SatelliteMultiHeadSegDataset,
    collect_samples,
    compute_class_weights,
    evaluate,
    save_visuals,
    split_samples,
)


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

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        logits = self.net(x)
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


class DinoV3MultiHeadSegmentation(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        semantic_classes: int,
        material_classes: int,
        detail_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_last_blocks: int = 0,
        decoder_channels: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = int(getattr(self.backbone, "embed_dim"))
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        self.patch_size = int(self.backbone.patch_embed.patch_size[0])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if unfreeze_last_blocks > 0 and hasattr(self.backbone, "blocks"):
            for block in self.backbone.blocks[-int(unfreeze_last_blocks):]:
                for param in block.parameters():
                    param.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True

        self.semantic_head = ConvSegHead(self.embed_dim, decoder_channels, semantic_classes, dropout)
        self.material_head = ConvSegHead(self.embed_dim, decoder_channels, material_classes, dropout)
        self.detail_head = ConvSegHead(self.embed_dim, decoder_channels, detail_classes, dropout)

    def _tokens_to_map(self, tokens: torch.Tensor, input_hw: tuple[int, int]) -> torch.Tensor:
        patch_tokens = tokens[:, self.num_prefix_tokens:, :]
        h, w = input_hw
        gh = h // self.patch_size
        gw = w // self.patch_size
        expected = gh * gw
        if patch_tokens.shape[1] != expected:
            # DINOv3/timm can expose extra prefix tokens depending on variant.
            patch_tokens = tokens[:, -expected:, :]
        return patch_tokens.transpose(1, 2).reshape(tokens.shape[0], self.embed_dim, gh, gw)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_hw = (x.shape[-2], x.shape[-1])
        tokens = self.backbone.forward_features(x)
        if isinstance(tokens, dict):
            tokens = tokens.get("x_norm_patchtokens", None) or tokens.get("x", None)
            if tokens is None:
                raise RuntimeError("Unsupported DINOv3 feature dict output")
        feat = self._tokens_to_map(tokens, input_hw)
        return {
            "semantic": self.semantic_head(feat, input_hw),
            "material": self.material_head(feat, input_hw),
            "detail": self.detail_head(feat, input_hw),
        }


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
            "palettes": {
                "semantic": SEMANTIC_PALETTE.tolist(),
                "material": MATERIAL_PALETTE.tolist(),
                "detail": DETAIL_PALETTE.tolist(),
            },
        },
        path,
    )


def build_optimizer(model: DinoV3MultiHeadSegmentation, config: dict):
    lr = float(config["training"]["lr"])
    backbone_lr = float(config["training"].get("backbone_lr", lr * 0.05))
    weight_decay = float(config["training"]["weight_decay"])
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/satellite_dinov3_multihead_segmentation_v16.yaml")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    exp_name = args.exp_name or config.get("experiment", {}).get("name", "satellite_dinov3_multihead_segmentation_v16")
    root = Path(config["dataset"]["root"])

    semantic_names = list(config["dataset"]["semantic_classes"])
    material_names = list(config["dataset"]["material_classes"])
    detail_names = list(config["dataset"]["detail_classes"])
    class_names = {"semantic": semantic_names, "material": material_names, "detail": detail_names}
    class_counts = {k: len(v) for k, v in class_names.items()}

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    if exp_name != config.get("experiment", {}).get("name", "satellite_dinov3_multihead_segmentation_v16"):
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
    print(f"[Data] semantic={semantic_names}")
    print(f"[Data] material={material_names}")
    print(f"[Data] detail={detail_names}")

    (output_dir / "splits.json").write_text(
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
        test_samples = test_samples[:4]
        config["training"]["epochs"] = 1
        config["training"]["batch_size"] = 2
        config["training"]["num_workers"] = 0
        config["model"]["pretrained"] = False

    train_ds = SatelliteMultiHeadSegDataset(train_samples, config, split="train")
    val_ds = SatelliteMultiHeadSegDataset(val_samples, config, split="val")
    test_ds = SatelliteMultiHeadSegDataset(test_samples, config, split="test")

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model_cfg = config["model"]
    print(
        f"[Train] Architecture: {model_cfg.get('backbone')} "
        f"pretrained={model_cfg.get('pretrained', True)} freeze_backbone={model_cfg.get('freeze_backbone', True)} "
        f"unfreeze_last_blocks={model_cfg.get('unfreeze_last_blocks', 0)}"
    )
    model = DinoV3MultiHeadSegmentation(
        backbone_name=model_cfg.get("backbone", "vit_small_patch16_dinov3"),
        semantic_classes=class_counts["semantic"],
        material_classes=class_counts["material"],
        detail_classes=class_counts["detail"],
        pretrained=bool(model_cfg.get("pretrained", True)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        unfreeze_last_blocks=int(model_cfg.get("unfreeze_last_blocks", 0)),
        decoder_channels=int(model_cfg.get("decoder_channels", 192)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Train] parameters trainable={trainable/1e6:.2f}M total={total/1e6:.2f}M")

    optimizer = build_optimizer(model, config)
    epochs = int(config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(config["training"].get("min_lr", 5e-6)),
    )

    losses = {}
    for key, num_classes in class_counts.items():
        weights = compute_class_weights(train_loader, key, num_classes, device)
        losses[key] = {
            "ce": nn.CrossEntropyLoss(weight=weights),
            "dice": DiceLoss(num_classes),
        }

    lw = dict(config["training"].get("loss_weights", {}))
    loss_weights = {
        "semantic": float(lw.get("semantic", 1.0)),
        "material": float(lw.get("material", 0.35)),
        "detail": float(lw.get("detail", 0.30)),
    }
    print(f"[Train] loss_weights={loss_weights}")

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config["training"].get("use_amp", True)))
    best_score = -1.0
    log_interval = int(config["training"].get("log_interval", 10))
    save_interval = int(config["training"].get("save_interval", 5))

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            targets = {k: batch[k].to(device, non_blocking=True) for k in class_counts}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda" and bool(config["training"].get("use_amp", True))):
                logits = model(x)
                loss = 0.0
                parts = {}
                for key, target in targets.items():
                    part = losses[key]["ce"](logits[key], target) + losses[key]["dice"](logits[key], target)
                    parts[key] = part
                    loss = loss + loss_weights[key] * part
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * x.size(0)
            if step % log_interval == 0:
                lrs = ",".join(f"{g['lr']:.6f}" for g in optimizer.param_groups)
                print(
                    f"[Train] epoch={epoch+1}/{epochs} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} sem={parts['semantic'].item():.4f} "
                    f"mat={parts['material'].item():.4f} det={parts['detail'].item():.4f} lr={lrs}",
                    flush=True,
                )

        scheduler.step()
        train_loss = running / max(1, len(train_loader.dataset))
        val_metrics = evaluate(model, val_loader, losses, loss_weights, device, class_counts)
        elapsed = time.time() - t0
        print(
            f"[Val] epoch={epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"score={val_metrics['score']:.4f} sem_mIoU={val_metrics['semantic_miou']:.4f} "
            f"mat_mIoU={val_metrics['material_miou']:.4f} det_mIoU={val_metrics['detail_miou']:.4f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        save_checkpoint(checkpoint_dir / "latest.pth", model, optimizer, epoch, val_metrics, config, class_names)
        if val_metrics["score"] > best_score:
            best_score = val_metrics["score"]
            save_checkpoint(checkpoint_dir / "best.pth", model, optimizer, epoch, val_metrics, config, class_names)
            save_visuals(model, val_loader, device, visual_dir, epoch + 1)
            print(f"[Checkpoint] New best score={best_score:.4f} -> {checkpoint_dir / 'best.pth'}", flush=True)
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth", model, optimizer, epoch, val_metrics, config, class_names)

    best_ckpt = torch.load(checkpoint_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, losses, loss_weights, device, class_counts)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({"best_val_score": best_score, "test": test_metrics, "classes": class_names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_visuals(model, test_loader, device, visual_dir, 999)
    print(f"[Done] best_val_score={best_score:.4f}")
    print(
        f"[Done] test_sem_mIoU={test_metrics['semantic_miou']:.4f} "
        f"test_mat_mIoU={test_metrics['material_miou']:.4f} "
        f"test_det_mIoU={test_metrics['detail_miou']:.4f}"
    )
    print(f"[Done] best_checkpoint={checkpoint_dir / 'best.pth'}")
    print(f"[Done] metrics={metrics_path}")


if __name__ == "__main__":
    main()
