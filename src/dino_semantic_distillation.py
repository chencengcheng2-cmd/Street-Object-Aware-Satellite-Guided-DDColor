"""DINO semantic distillation modules for v12.

This module implements scheme A:
existing semantic masks act as teacher labels, while frozen DINO features feed
small trainable segmentation heads. DINO itself is used as a feature extractor,
not as a native semantic segmentation model.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenDINOFeatureBackbone(nn.Module):
    """Frozen DINO feature extractor with a lightweight CNN fallback."""

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224.dino",
        pretrained: bool = True,
        out_channels: int = 128,
        input_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.model_name = model_name
        self.out_channels = out_channels
        self.input_size = input_size
        self.uses_timm = False

        try:
            import timm

            try:
                self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            except Exception as exc:
                print(f"[DINO v12] Warning: pretrained DINO unavailable ({exc}). Using random DINO weights.")
                self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
            self.uses_timm = True
            in_features = int(getattr(self.backbone, "num_features", 384))
            self.project = nn.Sequential(
                nn.Conv2d(in_features, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        except Exception as exc:
            print(f"[DINO v12] Warning: failed to create timm DINO ({exc}). Using CNN fallback.")
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            self.project = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
        """Convert ViT token tensor B,N,C to B,C,H,W."""
        token_count = tokens.shape[1]
        grid = int((token_count - 1) ** 0.5)
        if grid * grid == token_count - 1:
            tokens = tokens[:, 1:, :]
            token_count = tokens.shape[1]
        grid = int(token_count ** 0.5)
        if grid * grid != token_count:
            raise ValueError(f"Cannot reshape {token_count} DINO tokens into a square feature map.")
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid, grid)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        original_size = image.shape[-2:]
        image_in = F.interpolate(image, size=self.input_size, mode="bilinear", align_corners=False)
        with torch.no_grad():
            raw = self.backbone.forward_features(image_in) if self.uses_timm else self.backbone(image_in)

        if self.uses_timm:
            if isinstance(raw, dict):
                raw = raw.get("x_norm_patchtokens", raw.get("last_hidden_state", raw.get("tokens")))
            if raw is None:
                raise RuntimeError("DINO backbone returned an unsupported feature dictionary.")
            feat = self._tokens_to_feature_map(raw) if raw.dim() == 3 else raw
            feat = self.project(feat)
        else:
            feat = self.project(raw)

        return F.interpolate(feat, size=original_size, mode="bilinear", align_corners=False)


class DinoSemanticHead(nn.Module):
    """Small trainable segmentation head on top of frozen DINO features."""

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        num_classes: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class DinoSemanticDistillationModel(nn.Module):
    """Street, satellite, and polar semantic heads distilled from teacher masks."""

    def __init__(
        self,
        num_classes: int = 7,
        dino_model_name: str = "vit_small_patch16_224.dino",
        dino_pretrained: bool = True,
        feature_channels: int = 128,
        head_hidden_channels: int = 128,
        share_overhead_backbone: bool = True,
        freeze_dino: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.share_overhead_backbone = share_overhead_backbone

        self.street_backbone = FrozenDINOFeatureBackbone(
            model_name=dino_model_name,
            pretrained=dino_pretrained,
            out_channels=feature_channels,
        )
        self.overhead_backbone = FrozenDINOFeatureBackbone(
            model_name=dino_model_name,
            pretrained=dino_pretrained,
            out_channels=feature_channels,
        )
        self.polar_backbone = self.overhead_backbone if share_overhead_backbone else FrozenDINOFeatureBackbone(
            model_name=dino_model_name,
            pretrained=dino_pretrained,
            out_channels=feature_channels,
        )

        self.street_head = DinoSemanticHead(feature_channels, head_hidden_channels, num_classes)
        self.satellite_head = DinoSemanticHead(feature_channels, head_hidden_channels, num_classes)
        self.polar_head = DinoSemanticHead(feature_channels, head_hidden_channels, num_classes)

        if freeze_dino:
            self.freeze_backbones()

    def freeze_backbones(self):
        for module in [self.street_backbone, self.overhead_backbone, self.polar_backbone]:
            for param in module.parameters():
                param.requires_grad = False
        for module in [self.street_head, self.satellite_head, self.polar_head]:
            for param in module.parameters():
                param.requires_grad = True

    def forward(
        self,
        street_rgb: Optional[torch.Tensor] = None,
        satellite_rgb: Optional[torch.Tensor] = None,
        polar_rgb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        if street_rgb is not None:
            outputs["street_logits"] = self.street_head(self.street_backbone(street_rgb))
        if satellite_rgb is not None:
            outputs["satellite_logits"] = self.satellite_head(self.overhead_backbone(satellite_rgb))
        if polar_rgb is not None:
            outputs["polar_logits"] = self.polar_head(self.polar_backbone(polar_rgb))
        return outputs

    def predict_labels(self, **kwargs) -> Dict[str, torch.Tensor]:
        logits = self.forward(**kwargs)
        return {key.replace("_logits", "_labels"): value.argmax(dim=1) for key, value in logits.items()}
