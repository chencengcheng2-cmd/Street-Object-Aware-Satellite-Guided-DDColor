"""
Utility functions for the project.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file."""
    import yaml
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_checkpoint_dir(exp_name: str, base_dir: str = "checkpoints") -> Path:
    """Get checkpoint directory for experiment."""
    return Path(base_dir) / exp_name


def get_output_dir(exp_name: str, base_dir: str = "outputs") -> Path:
    """Get output directory for experiment."""
    return Path(base_dir) / exp_name


def create_timestamp_dir(base_dir: str) -> Path:
    """Create a directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / timestamp


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    metrics,
    path: str,
    is_best: bool = False,
):
    """Save training checkpoint."""
    import torch
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(checkpoint, path)

    if is_best:
        best_path = str(Path(path).parent / "best.pth")
        torch.save(checkpoint, best_path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    """Load training checkpoint."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    load_matching_state_dict(model, checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as exc:
            print(f"[Checkpoint] Skipping optimizer state: {exc}")

    if scheduler and checkpoint.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except ValueError as exc:
            print(f"[Checkpoint] Skipping scheduler state: {exc}")

    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})


def load_matching_state_dict(model, state_dict: Dict[str, Any]):
    """Load only parameters that exist in the current model with the same shape."""
    current = model.state_dict()
    compatible = {}
    skipped = []

    for key, value in state_dict.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"[Checkpoint] Skipped {len(skipped)} incompatible parameter(s).")
    if missing:
        print(f"[Checkpoint] Missing {len(missing)} parameter(s) initialized from current model.")
    if unexpected:
        print(f"[Checkpoint] Unexpected {len(unexpected)} parameter(s).")
    return {
        "loaded": list(compatible.keys()),
        "skipped": skipped,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def count_trainable_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model):
    """Count total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def get_device():
    """Get available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_tensor(arr: np.ndarray) -> "torch.Tensor":
    """Convert numpy array to tensor with proper shape."""
    import torch
    if len(arr.shape) == 2:
        arr = arr[:, :, None]
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


def to_numpy(tensor: "torch.Tensor") -> np.ndarray:
    """Convert tensor to numpy array."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.detach().numpy().transpose(1, 2, 0)


def tensor_to_img(tensor: "torch.Tensor", denormalize: bool = False) -> np.ndarray:
    """Convert tensor to uint8 image."""
    img = to_numpy(tensor)

    if denormalize:
        img = img * 255.0

    img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    return img


def img_to_tensor(img: np.ndarray, normalize: bool = False) -> "torch.Tensor":
    """Convert uint8 image to tensor."""
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    tensor = to_tensor(img)
    return tensor


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale using OpenCV."""
    if len(rgb.shape) == 3:
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.dtype == np.uint8 else \
               cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return rgb


class AverageMeter:
    """Compute and store average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class Progbar:
    """Simple progress bar."""

    def __init__(self, target, width=30, verbose=1):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.state = {"index": 0}

    def update(self, current, values=None):
        self.state["index"] = current

        if self.verbose == 1:
            pct = current / self.target
            filled = int(self.width * pct)
            bar = "=" * filled + " " * (self.width - filled)
            print(f"\r[{bar}] {pct*100:.1f}% ({current}/{self.target})", end="")

            if values:
                vals = ", ".join([f"{k}: {v:.4f}" for k, v in values.items()])
                print(f" - {vals}", end="")

            if current >= self.target:
                print()

    def add(self, n, values=None):
        self.update(self.state["index"] + n, values)
