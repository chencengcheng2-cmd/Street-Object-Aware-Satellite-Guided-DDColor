"""Semantic context utilities for mask-guided color correction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSatelliteStatsEncoder(nn.Module):
    """Encode satellite semantic ratios and per-class colors into a context vector.

    Class order: 0 sky, 1 impervious/roads, 2 building, 3 low vegetation, 4 tree, 5 car, 6 clutter/background. Satellite sky is expected to be empty.
    """

    def __init__(self, context_dim: int = 512, num_classes: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        stats_dim = num_classes * 4  # ratio + RGB mean per class
        self.mlp = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, context_dim),
        )

    def to_one_hot(self, semantic: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if semantic.dim() == 3:
            semantic = semantic.unsqueeze(1)
        if semantic.size(1) == 1:
            labels = semantic.long().clamp(0, self.num_classes - 1)
            one_hot = F.one_hot(labels.squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        else:
            one_hot = semantic[:, : self.num_classes].float()
        if one_hot.shape[-2:] != (h, w):
            one_hot = F.interpolate(one_hot, size=(h, w), mode="nearest")
        denom = one_hot.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return one_hot / denom

    def fallback_street_semantic(self, gray_rgb: torch.Tensor, base_rgb: torch.Tensor) -> torch.Tensor:
        b, _, h, w = base_rgb.shape
        y_grid = torch.linspace(0, 1, h, device=base_rgb.device, dtype=base_rgb.dtype).view(1, 1, h, 1)
        r, g, bl = base_rgb[:, 0:1], base_rgb[:, 1:2], base_rgb[:, 2:3]
        lum = base_rgb.mean(dim=1, keepdim=True)
        sat = base_rgb.max(dim=1, keepdim=True).values - base_rgb.min(dim=1, keepdim=True).values

        sky = (y_grid < 0.42) & ((bl > r * 1.03) | (lum > 0.62))
        impervious = (y_grid > 0.35) & (sat < 0.20) & (lum > 0.18) & (lum < 0.88) & (~sky)
        tree = (g > r * 1.08) & (g > bl * 1.05) & (sat > 0.10) & (y_grid < 0.90)
        low_veg = (g > r * 0.96) & (g >= bl * 0.92) & (lum > 0.15) & (y_grid > 0.25) & (~tree)
        dry_low_veg = (r > bl * 1.05) & (g > bl * 1.03) & (sat > 0.08) & (lum > 0.18) & (y_grid > 0.35)
        low_veg = low_veg | dry_low_veg
        building = (y_grid < 0.85) & (~sky) & (~impervious) & (~tree) & (~low_veg) & (lum > 0.18)
        car = torch.zeros_like(lum, dtype=torch.bool)

        masks = torch.cat([sky, impervious, building, low_veg, tree, car], dim=1).float()
        clutter = (1.0 - masks.max(dim=1, keepdim=True).values).clamp(0, 1)
        return torch.cat([masks, clutter], dim=1)

    def fallback_satellite_semantic(self, satellite_img: torch.Tensor) -> torch.Tensor:
        r, g, bl = satellite_img[:, 0:1], satellite_img[:, 1:2], satellite_img[:, 2:3]
        lum = satellite_img.mean(dim=1, keepdim=True)
        sat = satellite_img.max(dim=1, keepdim=True).values - satellite_img.min(dim=1, keepdim=True).values

        sky = torch.zeros_like(lum, dtype=torch.bool)
        impervious = (sat < 0.18) & (lum > 0.20) & (lum < 0.90)
        tree = (g > r * 1.08) & (g > bl * 1.05) & (sat > 0.10)
        low_veg = ((g > r * 0.96) & (g >= bl * 0.92) & (lum > 0.15) | ((r > bl * 1.05) & (g > bl * 1.03) & (sat > 0.08))) & (~tree)
        building = (~impervious) & (~tree) & (~low_veg) & (lum > 0.22)
        car = torch.zeros_like(lum, dtype=torch.bool)

        masks = torch.cat([sky, impervious, building, low_veg, tree, car], dim=1).float()
        clutter = (1.0 - masks.max(dim=1, keepdim=True).values).clamp(0, 1)
        return torch.cat([masks, clutter], dim=1)

    def street_one_hot(self, gray_rgb: torch.Tensor, base_rgb: torch.Tensor, semantic: torch.Tensor = None) -> torch.Tensor:
        if semantic is None:
            return self.fallback_street_semantic(gray_rgb, base_rgb)
        return self.to_one_hot(semantic, base_rgb.shape[-2], base_rgb.shape[-1])

    def satellite_one_hot(self, satellite_img: torch.Tensor, semantic: torch.Tensor = None) -> torch.Tensor:
        if semantic is None:
            return self.fallback_satellite_semantic(satellite_img)
        return self.to_one_hot(semantic, satellite_img.shape[-2], satellite_img.shape[-1])

    def forward(self, satellite_img: torch.Tensor, satellite_semantic: torch.Tensor = None) -> dict:
        semantic = self.satellite_one_hot(satellite_img, satellite_semantic)
        b, _, h, w = satellite_img.shape
        area = semantic.flatten(2).mean(dim=2)
        denom = semantic.flatten(2).sum(dim=2).clamp_min(1e-6)
        rgb_sum = torch.einsum("bchw,bkhw->bkc", satellite_img, semantic)
        mean_rgb = rgb_sum / denom.unsqueeze(-1)
        stats = torch.cat([area.unsqueeze(-1), mean_rgb], dim=-1).flatten(1)
        context = self.mlp(stats)
        return {
            "context": context,
            "stats": stats,
            "class_area": area,
            "class_mean_rgb": mean_rgb,
            "satellite_semantic": semantic,
        }
