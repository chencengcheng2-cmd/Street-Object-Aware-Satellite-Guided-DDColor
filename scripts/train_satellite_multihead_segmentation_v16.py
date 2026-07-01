"""Train a multi-head satellite segmentation model from LabelMe annotations.

Heads:
    semantic: supervised by manual LabelMe polygons.
    material: weak labels generated from RGB + semantic mask.
    detail: weak labels generated from edges + semantic mask + RGB cues.

This keeps the 400 manually labeled satellite images useful without requiring
immediate fine-grained relabeling for every road/grass/land subtype.
"""

from __future__ import annotations

import argparse
import json
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
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


SEMANTIC_PALETTE = np.array(
    [
        [255, 255, 255],  # impervious_surface
        [0, 90, 255],     # building
        [0, 220, 220],    # low_vegetation
        [20, 170, 40],    # tree
        [170, 120, 60],   # bare_land
        [30, 120, 255],   # water
        [80, 80, 80],     # shadow
        [220, 40, 40],    # other
    ],
    dtype=np.uint8,
)

MATERIAL_PALETTE = np.array(
    [
        [55, 55, 55],      # impervious_dark
        [185, 185, 175],   # impervious_light
        [180, 135, 75],    # impervious_warm
        [255, 235, 60],    # linear_marking
        [45, 210, 70],     # low_vegetation_healthy
        [205, 170, 75],    # low_vegetation_dry
        [145, 115, 55],    # shrubland
        [10, 110, 30],     # tree_dense
        [85, 190, 80],     # tree_sparse
        [125, 110, 55],    # tree_dry
        [150, 95, 55],     # bare_soil
        [220, 190, 120],   # sand
        [210, 210, 210],   # roof_bright
        [100, 110, 125],   # roof_dark
        [175, 120, 80],    # roof_warm
        [30, 120, 255],    # water
        [45, 45, 45],      # shadow
        [220, 40, 40],     # other
    ],
    dtype=np.uint8,
)

DETAIL_PALETTE = np.array(
    [
        [0, 0, 0],         # none
        [255, 255, 255],   # impervious_edge
        [255, 230, 0],     # linear_marking
        [255, 140, 0],     # transverse_marking
        [255, 80, 220],    # road_symbol_marking
        [0, 255, 120],     # vegetation_boundary
        [0, 120, 255],     # building_edge
        [30, 120, 255],    # water_boundary
        [140, 140, 140],   # shadow_edge
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


def _class_id(names: List[str], name: str) -> int:
    return names.index(name)


def apply_weak_water_labels(
    rgb: np.ndarray,
    semantic: np.ndarray,
    semantic_names: List[str],
    weak_config: dict,
) -> np.ndarray:
    """Promote high-confidence blue/cyan other regions to water.

    This is intentionally conservative: it only changes pixels that are still
    labeled as other, so manual road/building/grass/tree polygons are never
    overwritten by color heuristics.
    """

    water_cfg = dict(weak_config.get("water", {}))
    if not bool(water_cfg.get("enabled", False)):
        return semantic
    if "water" not in semantic_names or "other" not in semantic_names:
        return semantic

    other_id = _class_id(semantic_names, "other")
    water_id = _class_id(semantic_names, "water")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.int16)
    s = hsv[..., 1].astype(np.int16)
    v = hsv[..., 2].astype(np.int16)

    h_min = int(water_cfg.get("hue_min", 85))
    h_max = int(water_cfg.get("hue_max", 125))
    s_min = int(water_cfg.get("saturation_min", 35))
    v_min = int(water_cfg.get("value_min", 45))
    v_max = int(water_cfg.get("value_max", 210))
    min_area = int(water_cfg.get("min_component_area", 120))

    water_candidate = (
        (semantic == other_id)
        & (h >= h_min)
        & (h <= h_max)
        & (s >= s_min)
        & (v >= v_min)
        & (v <= v_max)
    ).astype(np.uint8)

    # Remove tiny blue objects/noise such as small roofs or compression artifacts.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(water_candidate, connectivity=8)
    keep = np.zeros_like(water_candidate, dtype=bool)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == label_id

    out = semantic.copy()
    out[keep] = water_id
    return out


def make_auxiliary_labels(
    rgb: np.ndarray,
    semantic: np.ndarray,
    semantic_names: List[str],
    material_names: List[str],
    detail_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create weak material/detail labels from image colors and semantics."""

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.int16)
    s = hsv[..., 1].astype(np.int16)
    v = hsv[..., 2].astype(np.int16)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    material = np.full(semantic.shape, _class_id(material_names, "other"), dtype=np.uint8)
    detail = np.full(semantic.shape, _class_id(detail_names, "none"), dtype=np.uint8)

    impervious = semantic == _class_id(semantic_names, "impervious_surface")
    building = semantic == _class_id(semantic_names, "building")
    low_vegetation = semantic == _class_id(semantic_names, "low_vegetation")
    tree = semantic == _class_id(semantic_names, "tree")
    bare_land = semantic == _class_id(semantic_names, "bare_land")
    water_sem = semantic == _class_id(semantic_names, "water")
    shadow_sem = semantic == _class_id(semantic_names, "shadow")
    other = semantic == _class_id(semantic_names, "other")

    dark = v < 55
    very_dark = v < 75
    yellow = (h >= 14) & (h <= 42) & (s > 45) & (v > 90)
    white = (s < 45) & (v > 165)
    green = (h >= 35) & (h <= 95) & (s > 35) & (v > 55)
    dark_green = (h >= 35) & (h <= 95) & (s > 35) & (v > 55) & (v < 105)
    light_green = (h >= 35) & (h <= 95) & (s > 35) & (v >= 130)
    dry_yellow_brown = (h >= 10) & (h <= 35) & (s > 35) & (v > 65)
    blue_water = (h >= 90) & (h <= 125) & (s > 35) & (v > 45)

    def assign_road_marking_details(marking_mask: np.ndarray):
        """Split road-surface markings by component geometry.

        Weak taxonomy:
        - linear_marking: long thin lane/edge lines.
        - transverse_marking: short horizontal/striped markings such as crosswalks.
        - road_symbol_marking: compact/medium road symbols, arrows, text-like blobs.
        """

        marking_u8 = (marking_mask > 0).astype(np.uint8)
        if marking_u8.max() == 0:
            return
        marking_u8 = cv2.morphologyEx(marking_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(marking_u8, connectivity=8)
        for label_id in range(1, num_labels):
            x, y, bw, bh, area = stats[label_id]
            if area < 3:
                continue
            short = max(1, min(bw, bh))
            long = max(bw, bh)
            elongation = long / short
            fill = area / max(1, bw * bh)
            component = labels == label_id

            if elongation >= 3.0 and area <= 0.012 * semantic.size:
                detail[component] = _class_id(detail_names, "linear_marking")
            elif bw >= 1.8 * bh and area <= 0.025 * semantic.size:
                detail[component] = _class_id(detail_names, "transverse_marking")
            elif area >= 8 and fill <= 0.80:
                detail[component] = _class_id(detail_names, "road_symbol_marking")
            else:
                detail[component] = _class_id(detail_names, "linear_marking")

    material[impervious] = _class_id(material_names, "impervious_dark")
    material[impervious & very_dark] = _class_id(material_names, "impervious_dark")
    material[impervious & (v > 145) & (s < 70)] = _class_id(material_names, "impervious_light")
    material[impervious & dry_yellow_brown & (v < 175)] = _class_id(material_names, "impervious_warm")
    material[impervious & ((yellow & (v >= 150)) | white)] = _class_id(material_names, "linear_marking")

    material[low_vegetation] = _class_id(material_names, "low_vegetation_dry")
    material[low_vegetation & green] = _class_id(material_names, "low_vegetation_healthy")
    material[low_vegetation & dry_yellow_brown] = _class_id(material_names, "low_vegetation_dry")
    material[low_vegetation & dry_yellow_brown & (s > 55)] = _class_id(material_names, "shrubland")

    material[tree] = _class_id(material_names, "tree_dense")
    material[tree & dark_green] = _class_id(material_names, "tree_dense")
    material[tree & light_green] = _class_id(material_names, "tree_sparse")
    material[tree & dry_yellow_brown] = _class_id(material_names, "tree_dry")

    material[bare_land] = _class_id(material_names, "bare_soil")
    material[bare_land & (v > 130) & (s < 120)] = _class_id(material_names, "sand")

    material[building] = _class_id(material_names, "roof_dark")
    material[building & (v > 150)] = _class_id(material_names, "roof_bright")
    material[building & dry_yellow_brown] = _class_id(material_names, "roof_warm")

    material[water_sem | (other & blue_water)] = _class_id(material_names, "water")
    material[shadow_sem | dark] = _class_id(material_names, "shadow")
    material[other & dry_yellow_brown] = _class_id(material_names, "bare_soil")

    edges = cv2.Canny(gray, 60, 150) > 0
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1).astype(bool)
    boundary = np.zeros_like(edges)
    boundary[:, 1:] |= semantic[:, 1:] != semantic[:, :-1]
    boundary[1:, :] |= semantic[1:, :] != semantic[:-1, :]
    boundary = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1).astype(bool)

    detail[impervious & edges] = _class_id(detail_names, "impervious_edge")
    road_marking = impervious & (yellow | white) & (edges | (v > 150))
    assign_road_marking_details(road_marking)
    detail[(low_vegetation | tree) & boundary] = _class_id(detail_names, "vegetation_boundary")
    detail[building & boundary] = _class_id(detail_names, "building_edge")
    detail[water_sem & boundary] = _class_id(detail_names, "water_boundary")
    detail[(shadow_sem | dark) & edges] = _class_id(detail_names, "shadow_edge")
    assign_road_marking_details(road_marking)

    return material, detail


class SatelliteMultiHeadSegDataset(Dataset):
    def __init__(self, samples: List[Sample], config: dict, split: str):
        self.samples = samples
        self.image_size = int(config["dataset"]["image_size"])
        self.semantic_names = list(config["dataset"]["semantic_classes"])
        self.material_names = list(config["dataset"]["material_classes"])
        self.detail_names = list(config["dataset"]["detail_classes"])
        self.class_to_id = {name: i for i, name in enumerate(self.semantic_names)}
        self.label_aliases = dict(config["dataset"].get("label_aliases", {}))
        self.default_label = config["dataset"].get("default_label", "other")
        self.weak_config = dict(config["dataset"].get("weak_labels", {}))
        self.split = split

    def __len__(self):
        return len(self.samples)

    def _load_pair(self, sample: Sample) -> Tuple[Image.Image, Image.Image]:
        img = Image.open(sample.image_path).convert("RGB")
        data = read_labelme(sample.json_path)
        semantic_np = rasterize_labelme(data, self.class_to_id, self.label_aliases, self.default_label)
        semantic = Image.fromarray(semantic_np, mode="L")
        return img, semantic

    def _augment(self, img: Image.Image, semantic: Image.Image):
        if self.split != "train":
            return img, semantic
        if random.random() < 0.5:
            img = TF.hflip(img)
            semantic = TF.hflip(semantic)
        if random.random() < 0.5:
            img = TF.vflip(img)
            semantic = TF.vflip(semantic)
        k = random.randint(0, 3)
        if k:
            angle = 90 * k
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            semantic = TF.rotate(semantic, angle, interpolation=TF.InterpolationMode.NEAREST)
        if random.random() < 0.8:
            img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
            img = TF.adjust_contrast(img, random.uniform(0.85, 1.15))
            img = TF.adjust_saturation(img, random.uniform(0.85, 1.15))
        return img, semantic

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img, semantic = self._load_pair(sample)
        img, semantic = self._augment(img, semantic)
        img = TF.resize(img, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
        semantic = TF.resize(semantic, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        rgb_np = np.array(img, dtype=np.uint8)
        semantic_np = np.array(semantic, dtype=np.uint8)
        semantic_np = apply_weak_water_labels(rgb_np, semantic_np, self.semantic_names, self.weak_config)
        material_np, detail_np = make_auxiliary_labels(
            rgb_np,
            semantic_np,
            self.semantic_names,
            self.material_names,
            self.detail_names,
        )

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return {
            "image": img_t,
            "semantic": torch.from_numpy(semantic_np.astype(np.int64)),
            "material": torch.from_numpy(material_np.astype(np.int64)),
            "detail": torch.from_numpy(detail_np.astype(np.int64)),
            "sample_id": sample.sample_id,
        }


class MultiHeadDeepLabV3(nn.Module):
    def __init__(self, semantic_classes: int, material_classes: int, detail_classes: int, pretrained_encoder: bool):
        super().__init__()
        weights_backbone = ResNet50_Weights.DEFAULT if pretrained_encoder else None
        base = deeplabv3_resnet50(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=semantic_classes,
            aux_loss=False,
        )
        self.backbone = base.backbone
        self.semantic_head = DeepLabHead(2048, semantic_classes)
        self.material_head = DeepLabHead(2048, material_classes)
        self.detail_head = DeepLabHead(2048, detail_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)["out"]
        return {
            "semantic": F.interpolate(self.semantic_head(features), size=input_shape, mode="bilinear", align_corners=False),
            "material": F.interpolate(self.material_head(features), size=input_shape, mode="bilinear", align_corners=False),
            "detail": F.interpolate(self.detail_head(features), size=input_shape, mode="bilinear", align_corners=False),
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
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1 - dice.mean()


def compute_class_weights(loader: DataLoader, key: str, num_classes: int, device: torch.device):
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for batch in loader:
        mask = batch[key].view(-1)
        counts += torch.bincount(mask, minlength=num_classes).double()
    freq = counts / counts.sum().clamp_min(1)
    weights = 1.0 / torch.log(1.02 + freq)
    weights = weights / weights.mean()
    print(f"[Data] {key} pixel counts:", counts.long().tolist())
    print(f"[Data] {key} class weights:", [round(float(x), 3) for x in weights])
    return weights.float().to(device)


def _confusion_iou(pred, target, num_classes: int, device: torch.device):
    idx = (target.view(-1) * num_classes + pred.view(-1)).long()
    conf = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes).to(device)
    tp = conf.diag().float()
    union = conf.sum(1).float() + conf.sum(0).float() - tp
    iou = tp / union.clamp_min(1)
    acc = tp.sum() / conf.sum().clamp_min(1)
    return iou, acc


@torch.no_grad()
def evaluate(model, loader, losses, weights, device, class_counts: Dict[str, int]):
    model.eval()
    total_loss = 0.0
    confs = {k: torch.zeros((n, n), dtype=torch.int64, device=device) for k, n in class_counts.items()}
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        targets = {k: batch[k].to(device, non_blocking=True) for k in class_counts}
        logits = model(x)
        loss = 0.0
        for key, target in targets.items():
            loss = loss + weights[key] * (losses[key]["ce"](logits[key], target) + losses[key]["dice"](logits[key], target))
            pred = logits[key].argmax(dim=1)
            idx = (target.view(-1) * class_counts[key] + pred.view(-1)).long()
            confs[key] += torch.bincount(idx, minlength=class_counts[key] ** 2).reshape(class_counts[key], class_counts[key])
        total_loss += float(loss.item()) * x.size(0)

    metrics = {"loss": total_loss / max(1, len(loader.dataset))}
    for key, conf in confs.items():
        tp = conf.diag().float()
        union = conf.sum(1).float() + conf.sum(0).float() - tp
        iou = tp / union.clamp_min(1)
        acc = tp.sum() / conf.sum().clamp_min(1)
        metrics[f"{key}_miou"] = float(iou.mean().item())
        metrics[f"{key}_acc"] = float(acc.item())
        metrics[f"{key}_iou"] = [float(v.item()) for v in iou]
    metrics["score"] = metrics["semantic_miou"] + 0.25 * metrics["material_miou"] + 0.15 * metrics["detail_miou"]
    return metrics


def denormalize(img_t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_t.device).view(3, 1, 1)
    img = (img_t * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def colorize(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[np.clip(mask, 0, len(palette) - 1)]


@torch.no_grad()
def save_visuals(model, loader, device, out_dir: Path, epoch: int):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(loader))
    x = batch["image"].to(device)
    out = model(x)
    sem_pred = out["semantic"].argmax(dim=1).cpu().numpy()
    mat_pred = out["material"].argmax(dim=1).cpu().numpy()
    det_pred = out["detail"].argmax(dim=1).cpu().numpy()
    sem_gt = batch["semantic"].numpy()
    mat_gt = batch["material"].numpy()
    det_gt = batch["detail"].numpy()
    n = min(4, x.size(0))
    tiles = []
    for i in range(n):
        img = denormalize(x[i])
        row1 = np.concatenate([img, colorize(sem_gt[i], SEMANTIC_PALETTE), colorize(sem_pred[i], SEMANTIC_PALETTE)], axis=1)
        row2 = np.concatenate([img, colorize(mat_gt[i], MATERIAL_PALETTE), colorize(mat_pred[i], MATERIAL_PALETTE)], axis=1)
        row3 = np.concatenate([img, colorize(det_gt[i], DETAIL_PALETTE), colorize(det_pred[i], DETAIL_PALETTE)], axis=1)
        tile = np.concatenate([row1, row2, row3], axis=0)
        cv2.putText(tile, "RGB | weak/GT | pred   rows: semantic/material/detail", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
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
            "palettes": {
                "semantic": SEMANTIC_PALETTE.tolist(),
                "material": MATERIAL_PALETTE.tolist(),
                "detail": DETAIL_PALETTE.tolist(),
            },
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/satellite_multihead_segmentation_v16.yaml")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    exp_name = args.exp_name or config.get("experiment", {}).get("name", "satellite_multihead_segmentation_v16")
    root = Path(config["dataset"]["root"])

    semantic_names = list(config["dataset"]["semantic_classes"])
    material_names = list(config["dataset"]["material_classes"])
    detail_names = list(config["dataset"]["detail_classes"])
    class_names = {"semantic": semantic_names, "material": material_names, "detail": detail_names}
    class_counts = {k: len(v) for k, v in class_names.items()}

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    if exp_name != config.get("experiment", {}).get("name", "satellite_multihead_segmentation_v16"):
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

    train_ds = SatelliteMultiHeadSegDataset(train_samples, config, split="train")
    val_ds = SatelliteMultiHeadSegDataset(val_samples, config, split="val")
    test_ds = SatelliteMultiHeadSegDataset(test_samples, config, split="test")

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    pretrained_encoder = bool(config["training"].get("pretrained_encoder", True))
    print(f"[Train] Architecture: multihead_deeplabv3_resnet50 pretrained_encoder={pretrained_encoder}")
    model = MultiHeadDeepLabV3(
        semantic_classes=class_counts["semantic"],
        material_classes=class_counts["material"],
        detail_classes=class_counts["detail"],
        pretrained_encoder=pretrained_encoder,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"]))
    epochs = int(config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(config["training"].get("min_lr", 5e-6)))

    losses = {}
    for key, num_classes in class_counts.items():
        weights = compute_class_weights(train_loader, key, num_classes, device)
        losses[key] = {
            "ce": nn.CrossEntropyLoss(weight=weights),
            "dice": DiceLoss(num_classes),
        }

    loss_weights = dict(config["training"].get("loss_weights", {}))
    loss_weights = {
        "semantic": float(loss_weights.get("semantic", 1.0)),
        "material": float(loss_weights.get("material", 0.35)),
        "detail": float(loss_weights.get("detail", 0.25)),
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * x.size(0)
            if step % log_interval == 0:
                print(
                    f"[Train] epoch={epoch+1}/{epochs} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} sem={parts['semantic'].item():.4f} "
                    f"mat={parts['material'].item():.4f} det={parts['detail'].item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}",
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
