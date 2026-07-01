"""Street-object-aware correction module for v11.

This module follows the v11 design from the project report:
- Frozen DDColor provides a stable base colorization.
- Polar satellite RGB plus satellite/polar semantic cues form a scene prior.
- Street-side learned masks, detail attention, and frozen DINO features decide
  where the base result should be changed.
- The module predicts residual corrections instead of regenerating the image.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenDINOStreetFeatures(nn.Module):
    """Frozen DINO feature extractor with a safe lightweight fallback."""

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224.dino",
        pretrained: bool = True,
        out_channels: int = 96,
    ):
        super().__init__()
        self.uses_timm = False
        self.model_name = model_name

        try:
            import timm

            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.uses_timm = True
            in_features = int(getattr(self.backbone, "num_features", 384))
            self.project = nn.Sequential(
                nn.Conv2d(in_features, out_channels, kernel_size=1),
                nn.ReLU(inplace=True),
            )
        except Exception as exc:
            print(f"[v11] Warning: failed to create DINO backbone ({exc}). Using frozen CNN fallback.")
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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_224 = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            backbone_out = self.backbone.forward_features(image_224) if self.uses_timm else self.backbone(image_224)

        if self.uses_timm:
            tokens = backbone_out
            if isinstance(tokens, dict):
                tokens = tokens.get("x_norm_patchtokens", tokens.get("last_hidden_state"))
            if tokens.dim() == 3:
                # Drop cls token when present and reshape patch tokens to a feature map.
                token_count = tokens.shape[1]
                grid = int((token_count - 1) ** 0.5)
                if grid * grid == token_count - 1:
                    tokens = tokens[:, 1:, :]
                else:
                    grid = int(token_count ** 0.5)
                feat = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid, grid)
            else:
                feat = tokens
            feat = self.project(feat)
        else:
            feat = self.project(backbone_out)
        return F.interpolate(feat, size=image.shape[-2:], mode="bilinear", align_corners=False)


class StreetObjectAwareCorrectionModule(nn.Module):
    """v11 residual correction with polar prior and street-object awareness."""

    def __init__(
        self,
        context_dim: int = 512,
        hidden_dim: int = 96,
        num_semantic_classes: int = 7,
        num_street_masks: int = 8,
        residual_scale: float = 0.22,
        detail_scale: float = 0.18,
        dino_model_name: str = "vit_small_patch16_224.dino",
        dino_pretrained: bool = True,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_semantic_classes = num_semantic_classes
        self.num_street_masks = num_street_masks
        self.residual_scale = residual_scale
        self.detail_scale = detail_scale

        prior_channels = 3 + 3 + num_semantic_classes
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(prior_channels, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 160, 3, stride=2, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.prior_projection = nn.Sequential(
            nn.Linear(160, context_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(context_dim, context_dim),
        )
        self.stats_projection = nn.Sequential(
            nn.Linear(num_semantic_classes + 9, context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim, context_dim),
        )

        street_in_channels = 6 + num_semantic_classes  # base RGB + gray + edge + saturation + street DINO semantics
        self.street_stem = nn.Sequential(
            nn.Conv2d(street_in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.street_mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_street_masks, 1),
        )
        self.object_attention = nn.Sequential(
            nn.Conv2d(hidden_dim + num_street_masks + num_semantic_classes + 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

        self.dino = FrozenDINOStreetFeatures(
            model_name=dino_model_name,
            pretrained=dino_pretrained,
            out_channels=hidden_dim,
        )
        self.fuse_street = nn.Sequential(
            nn.Conv2d(hidden_dim * 2 + num_street_masks + num_semantic_classes + 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.film = nn.Sequential(
            nn.Linear(context_dim, hidden_dim * 2),
        )
        self.global_residual = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, 3, padding=1),
            nn.Tanh(),
        )
        self.global_gate = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.detail_residual = nn.Sequential(
            nn.Conv2d(hidden_dim + 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, 3, padding=1),
            nn.Tanh(),
        )
        self.preservation_gate = nn.Sequential(
            nn.Conv2d(3 + 3 + 1 + 1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

        # Start close to DDColor and let training learn stronger edits.
        nn.init.constant_(self.global_residual[-2].weight, 0.0)
        nn.init.constant_(self.global_residual[-2].bias, 0.0)
        nn.init.constant_(self.detail_residual[-2].weight, 0.0)
        nn.init.constant_(self.detail_residual[-2].bias, 0.0)

    @staticmethod
    def _edge_map(gray_rgb: torch.Tensor) -> torch.Tensor:
        gray = gray_rgb.mean(dim=1, keepdim=True)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        edge = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
        return edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)

    @staticmethod
    def _saturation(rgb: torch.Tensor) -> torch.Tensor:
        max_rgb = rgb.max(dim=1, keepdim=True).values
        min_rgb = rgb.min(dim=1, keepdim=True).values
        return (max_rgb - min_rgb) / (max_rgb + 1e-6)

    def _semantic_one_hot(
        self,
        semantic: Optional[torch.Tensor],
        size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if semantic is None:
            return torch.zeros(1, self.num_semantic_classes, *size, device=device, dtype=dtype)
        semantic = semantic.to(device=device, dtype=torch.long).clamp(0, self.num_semantic_classes - 1)
        one_hot = F.one_hot(semantic, num_classes=self.num_semantic_classes).permute(0, 3, 1, 2).float()
        one_hot = one_hot.to(dtype=dtype)
        if one_hot.shape[-2:] != size:
            one_hot = F.interpolate(one_hot, size=size, mode="nearest")
        return one_hot

    def _prior_context(
        self,
        polar_img: torch.Tensor,
        polar_seg_rgb: Optional[torch.Tensor],
        satellite_semantic: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, _, h, w = polar_img.shape
        if polar_seg_rgb is None:
            polar_seg_rgb = torch.zeros_like(polar_img)
        else:
            polar_seg_rgb = polar_seg_rgb.to(device=polar_img.device, dtype=polar_img.dtype)
            if polar_seg_rgb.shape[-2:] != (h, w):
                polar_seg_rgb = F.interpolate(polar_seg_rgb, size=(h, w), mode="bilinear", align_corners=False)

        sem = self._semantic_one_hot(
            satellite_semantic,
            size=(h, w),
            device=polar_img.device,
            dtype=polar_img.dtype,
        )
        if sem.shape[0] == 1 and b > 1:
            sem = sem.expand(b, -1, -1, -1)

        prior_in = torch.cat([polar_img, polar_seg_rgb, sem], dim=1)
        cnn_context = self.prior_encoder(prior_in).flatten(1)
        cnn_context = self.prior_projection(cnn_context)

        class_hist = sem.mean(dim=(2, 3))
        rgb_mean = polar_img.mean(dim=(2, 3))
        rgb_std = polar_img.std(dim=(2, 3))
        seg_mean = polar_seg_rgb.mean(dim=(2, 3))
        stats = torch.cat([class_hist, rgb_mean, rgb_std, seg_mean], dim=1)
        stats_context = self.stats_projection(stats)
        return cnn_context + stats_context

    def _street_semantic_context(
        self,
        street_semantic: Optional[torch.Tensor],
        size: tuple[int, int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sem = self._semantic_one_hot(
            street_semantic,
            size=size,
            device=device,
            dtype=dtype,
        )
        if sem.shape[0] == 1 and batch_size > 1:
            sem = sem.expand(batch_size, -1, -1, -1)
        return sem

    def forward(
        self,
        base_rgb: torch.Tensor,
        gray_rgb: torch.Tensor,
        polar_img: torch.Tensor,
        polar_seg_rgb: Optional[torch.Tensor] = None,
        satellite_semantic: Optional[torch.Tensor] = None,
        street_semantic: Optional[torch.Tensor] = None,
        patch_idx: Optional[torch.Tensor] = None,
    ) -> dict:
        if polar_img is None:
            polar_img = torch.zeros(
                base_rgb.size(0), 3, 256, 512, device=base_rgb.device, dtype=base_rgb.dtype
            )

        prior_context = self._prior_context(polar_img, polar_seg_rgb, satellite_semantic)
        edge = self._edge_map(gray_rgb)
        saturation = self._saturation(base_rgb)
        gray = gray_rgb.mean(dim=1, keepdim=True)
        street_sem = self._street_semantic_context(
            street_semantic,
            size=base_rgb.shape[-2:],
            batch_size=base_rgb.shape[0],
            device=base_rgb.device,
            dtype=base_rgb.dtype,
        )
        street_input = torch.cat([base_rgb, gray, edge, saturation, street_sem], dim=1)

        street_feat = self.street_stem(street_input)
        street_logits = self.street_mask_head(street_feat)
        street_masks = F.softmax(street_logits, dim=1)
        object_attention = self.object_attention(torch.cat([street_feat, street_masks, street_sem, edge, saturation], dim=1))

        dino_feat = self.dino(base_rgb)
        fused = self.fuse_street(torch.cat([street_feat, dino_feat, street_masks, street_sem, edge, object_attention], dim=1))

        gamma_beta = self.film(prior_context)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(gamma.shape[0], -1, 1, 1)
        beta = beta.view(beta.shape[0], -1, 1, 1)
        modulated = fused * (1.0 + gamma) + beta

        global_gate = self.global_gate(torch.cat([modulated, saturation], dim=1))
        global_delta = self.global_residual(modulated) * self.residual_scale * global_gate
        detail_delta = self.detail_residual(torch.cat([fused, edge, object_attention], dim=1))
        detail_delta = detail_delta * self.detail_scale * object_attention

        candidate = torch.clamp(base_rgb + global_delta + detail_delta, 0.0, 1.0)
        preserve_gate = self.preservation_gate(torch.cat([base_rgb, candidate, saturation, object_attention], dim=1))
        final_rgb = torch.clamp(preserve_gate * base_rgb + (1.0 - preserve_gate) * candidate, 0.0, 1.0)
        delta = final_rgb - base_rgb

        return {
            "final_rgb": final_rgb,
            "delta_color": delta,
            "prior_context": prior_context,
            "street_masks": street_masks,
            "street_semantic": street_sem,
            "object_attention": object_attention,
            "global_delta": global_delta,
            "detail_delta": detail_delta,
            "preservation_gate": preserve_gate,
        }
