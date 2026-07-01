"""Frozen DDColor adapter returning official-pipeline RGB colorizations."""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDColorWrapper(nn.Module):
    """Load the local official DDColor model and keep it permanently frozen."""

    def __init__(
        self,
        model_path: str,
        model_code_path: str = None,
        input_size: int = 256,
        model_size: str = "tiny",
        device: str = "auto",
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.ddcolor = self._load_ddcolor_model(model_path, model_code_path, model_size)
        for parameter in self.ddcolor.parameters():
            parameter.requires_grad = False
        self.ddcolor.eval()

    def _load_ddcolor_model(self, model_path: str, model_code_path: str, model_size: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"DDColor weights not found: {model_path}")
        if model_code_path and model_code_path not in sys.path:
            sys.path.insert(0, model_code_path)
        try:
            from ddcolor import DDColor, build_ddcolor_model
        except ImportError as exc:
            raise ImportError(f"Could not import local DDColor code from {model_code_path}") from exc
        return build_ddcolor_model(
            DDColor,
            model_path=model_path,
            input_size=self.input_size,
            model_size=model_size,
            device=self.device,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.ddcolor.eval()
        return self

    def forward(self, gray_rgb: torch.Tensor) -> torch.Tensor:
        return self.colorize(gray_rgb)

    @torch.inference_mode()
    def colorize(self, gray_rgb: torch.Tensor) -> torch.Tensor:
        """Apply DDColor using its official Lab reconstruction, batched on GPU."""
        if gray_rgb.ndim != 4 or gray_rgb.shape[1] != 3:
            raise ValueError(f"Expected Bx3xHxW grayscale RGB tensor, got {tuple(gray_rgb.shape)}")

        input_l = []
        network_inputs = []
        height, width = gray_rgb.shape[-2:]
        images = gray_rgb.detach().float().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
        for image_rgb in images:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            orig_l = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, :1]
            resized = cv2.resize(image_bgr, (self.input_size, self.input_size))
            img_l = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)[:, :, :1]
            gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
            network_inputs.append(cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB).transpose(2, 0, 1))
            input_l.append(orig_l)

        tensor_input = torch.from_numpy(np.stack(network_inputs)).float().to(self.device)
        output_ab = self.ddcolor(tensor_input)
        output_ab = F.interpolate(output_ab, size=(height, width), mode="bilinear", align_corners=False)
        output_ab = output_ab.float().cpu().permute(0, 2, 3, 1).numpy()

        results = []
        for lightness, chroma in zip(input_l, output_ab):
            output_lab = np.concatenate((lightness, chroma), axis=-1)
            output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
            results.append(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
        return torch.from_numpy(np.stack(results)).float().clamp(0, 1).to(self.device)

    def get_num_parameters(self) -> tuple:
        total = sum(p.numel() for p in self.ddcolor.parameters())
        frozen = sum(p.numel() for p in self.ddcolor.parameters() if not p.requires_grad)
        return total, frozen

    def verify_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.ddcolor.parameters()) and not self.ddcolor.training
