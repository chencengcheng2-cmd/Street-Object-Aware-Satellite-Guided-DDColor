"""Export v12 DINO semantic predictions as label-map images.

The colorization pipeline reads precomputed semantic masks from the CVUSA split
directories. This script bridges the v12 DINO semantic distillation checkpoint
to that pipeline by writing predicted street and satellite label maps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dino_semantic_distillation import DinoSemanticDistillationModel
from src.utils import load_config


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def iter_image_files(directory: Path) -> Iterable[Path]:
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


def load_rgb(path: Path, size_wh: Tuple[int, int]) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size_wh, interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


class StreetPatchExportDataset(Dataset):
    def __init__(self, root: Path, split: str, size_wh: Tuple[int, int]):
        self.rgb_dir = root / split / "ground_rgb"
        self.size_wh = size_wh
        self.items = sorted(iter_image_files(self.rgb_dir), key=lambda path: path.stem)
        if not self.items:
            raise FileNotFoundError(f"No street images found in {self.rgb_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        path = self.items[idx]
        return {"id": path.stem, "image": load_rgb(path, self.size_wh)}


class SatelliteExportDataset(Dataset):
    def __init__(self, root: Path, split: str, size_wh: Tuple[int, int]):
        self.size_wh = size_wh
        self.items = sorted(build_shared_index(root / split / "overhead_satellite").items())
        if not self.items:
            raise FileNotFoundError(f"No satellite images found in {root / split / 'overhead_satellite'}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        image_id, path = self.items[idx]
        return {"id": image_id, "image": load_rgb(path, self.size_wh)}


def collate(batch: List[Dict]) -> Dict:
    return {
        "id": [item["id"] for item in batch],
        "image": torch.stack([item["image"] for item in batch]),
    }


def load_model(checkpoint_path: Path, config: Dict, device: torch.device) -> DinoSemanticDistillationModel:
    model_cfg = config["model"]
    model = DinoSemanticDistillationModel(
        num_classes=int(model_cfg.get("num_classes", 7)),
        dino_model_name=model_cfg.get("dino_model_name", "vit_small_patch16_224.dino"),
        dino_pretrained=bool(model_cfg.get("dino_pretrained", True)),
        feature_channels=int(model_cfg.get("feature_channels", 128)),
        head_hidden_channels=int(model_cfg.get("head_hidden_channels", 128)),
        share_overhead_backbone=bool(model_cfg.get("share_overhead_backbone", True)),
        freeze_dino=bool(model_cfg.get("freeze_dino", True)),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def export_loader(
    model: DinoSemanticDistillationModel,
    loader: DataLoader,
    output_dir: Path,
    branch: str,
    device: torch.device,
    use_amp: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                if branch == "street":
                    logits = model(street_rgb=images)["street_logits"]
                elif branch == "satellite":
                    logits = model(satellite_rgb=images)["satellite_logits"]
                else:
                    raise ValueError(f"Unsupported branch: {branch}")
            labels = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            for image_id, label in zip(batch["id"], labels):
                cv2.imwrite(str(output_dir / f"{image_id}.png"), label)
                count += 1
            if count % 500 == 0:
                print(f"[DINO v12 export] {branch}: exported {count} masks", flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(description="Export v12 DINO semantic masks for colorization training.")
    parser.add_argument("--config", default="configs/dino_semantic_distill_v12.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/dino_semantic_distill_v12/best.pth")
    parser.add_argument("--street_dirname", default="street_dino_semantic_v12")
    parser.add_argument("--satellite_dirname", default="overhead_satellite_dino_semantic_v12")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_amp", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["dataset"]["root"])
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model_cfg = config["model"]
    image_size_hw = tuple(model_cfg.get("image_size", [256, 256]))
    size_wh = (int(image_size_hw[1]), int(image_size_hw[0]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, config, device)

    total = 0
    for split in args.splits:
        split_dir = root / split
        street_dataset = StreetPatchExportDataset(root, split, size_wh)
        satellite_dataset = SatelliteExportDataset(root, split, size_wh)
        loader_kwargs = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "collate_fn": collate,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
        }
        street_loader = DataLoader(street_dataset, **loader_kwargs)
        satellite_loader = DataLoader(satellite_dataset, **loader_kwargs)
        street_out = split_dir / args.street_dirname
        satellite_out = split_dir / args.satellite_dirname
        print(f"[DINO v12 export] Split={split} street_out={street_out}", flush=True)
        total += export_loader(model, street_loader, street_out, "street", device, args.use_amp)
        print(f"[DINO v12 export] Split={split} satellite_out={satellite_out}", flush=True)
        total += export_loader(model, satellite_loader, satellite_out, "satellite", device, args.use_amp)

    print(f"[DINO v12 export] Done. Exported {total} masks.", flush=True)


if __name__ == "__main__":
    main()
