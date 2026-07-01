"""Satellite-color-bottleneck correction module for v14.

The module intentionally separates "where" from "what color":
- Street semantics and street edge maps decide where a correction is allowed.
- Satellite RGB plus satellite semantics provide class-wise color prototypes.
- Non-sky corrections are driven toward the satellite color prior instead of
  being generated directly from grayscale street appearance.
- Sky is treated as a street-only exception because it has no satellite match.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SatelliteColorBottleneckCorrectionModule(nn.Module):
    """Mask-separated correction with satellite-derived color prototypes."""

    def __init__(
        self,
        num_semantic_classes: int = 7,
        hidden_dim: int = 64,
        residual_scale: float = 0.22,
        detail_scale: float = 0.12,
        satellite_prior_strength: float = 0.65,
        use_street_gray_edges: bool = False,
        use_street_gray_modulation: bool = True,
        use_gray_satellite_token_selection: bool = True,
        use_satellite_chroma_token_selection: bool = False,
        token_selection_patch_size: int = 16,
        token_selection_dim: int = 32,
        street_semantic_source: str = "dino_v12",
        satellite_semantic_source: str = "neos",
    ):
        super().__init__()
        self.num_semantic_classes = num_semantic_classes
        self.hidden_dim = hidden_dim
        self.residual_scale = residual_scale
        self.detail_scale = detail_scale
        self.satellite_prior_strength = satellite_prior_strength
        self.use_street_gray_edges = use_street_gray_edges
        self.use_street_gray_modulation = use_street_gray_modulation
        self.use_gray_satellite_token_selection = use_gray_satellite_token_selection
        self.use_satellite_chroma_token_selection = use_satellite_chroma_token_selection
        self.token_selection_patch_size = token_selection_patch_size
        self.street_semantic_source = street_semantic_source
        self.satellite_semantic_source = satellite_semantic_source
        # Unified semantic ids: 0 sky, 1 road, 2 building, 3 grass, 4 tree,
        # 5 car, 6 other. Cars are view-specific transient objects in CVUSA;
        # satellite car masks are not useful for street-view color guidance.
        self.sky_class_id = 0
        self.car_class_id = 5

        # Gate input:
        # - base luminance and semantic masks say where this street pixel belongs.
        # - satellite prior RGB says which color source is available.
        # - optional grayscale luma/contrast only modulates gate strength.
        #
        # Detail color input intentionally excludes grayscale luma/contrast. This
        # prevents the detail residual head from directly learning
        # gray-street -> real-street color mappings.
        detail_base_channels = 1 + 3 + 1 + 1 + num_semantic_classes + 1
        gate_channels = detail_base_channels + 2
        self.non_sky_gate = nn.Sequential(
            nn.Conv2d(gate_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

        detail_channels = detail_base_channels + 1
        self.detail_delta = nn.Sequential(
            nn.Conv2d(detail_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, 3, padding=1),
            nn.Tanh(),
        )

        # Sky is allowed to use the DDColor result because satellite views do
        # not contain sky. Keep this branch small so it remains a refinement.
        sky_channels = 3 + 1 + 1
        self.sky_refine = nn.Sequential(
            nn.Conv2d(sky_channels, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 3, 3, padding=1),
            nn.Tanh(),
        )

        # Grayscale is allowed to choose which same-semantic satellite token to
        # read from. The selected value is still satellite RGB, not a color
        # generated from street grayscale.
        self.street_token_query = nn.Sequential(
            nn.Conv2d(num_semantic_classes + 2, token_selection_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_selection_dim, token_selection_dim, 1),
        )
        self.satellite_token_key = nn.Sequential(
            nn.Conv2d(num_semantic_classes + 2, token_selection_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_selection_dim, token_selection_dim, 1),
        )
        if not self.use_gray_satellite_token_selection:
            for module in (self.street_token_query, self.satellite_token_key):
                for param in module.parameters():
                    param.requires_grad = False

        nn.init.constant_(self.detail_delta[-2].weight, 0.0)
        nn.init.constant_(self.detail_delta[-2].bias, 0.0)
        nn.init.constant_(self.sky_refine[-2].weight, 0.0)
        nn.init.constant_(self.sky_refine[-2].bias, 0.0)

    @staticmethod
    def _luma(rgb: torch.Tensor) -> torch.Tensor:
        weights = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (rgb * weights).sum(dim=1, keepdim=True)

    @staticmethod
    def _edge_map(image: torch.Tensor) -> torch.Tensor:
        gray = image.mean(dim=1, keepdim=True)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, 3, 3)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        edge = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
        return edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)

    @staticmethod
    def _local_contrast(gray_rgb: torch.Tensor) -> torch.Tensor:
        gray = gray_rgb.mean(dim=1, keepdim=True)
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        contrast = torch.abs(gray - local_mean)
        return contrast / (contrast.amax(dim=(2, 3), keepdim=True) + 1e-6)

    def _satellite_chroma_features(self, satellite_img: torch.Tensor) -> torch.Tensor:
        """Two-channel chroma descriptor for satellite token matching.

        We intentionally remove luminance so the satellite branch mostly
        contributes color tendency instead of overhead brightness.
        """
        luma = self._luma(satellite_img)
        r_chroma = satellite_img[:, 0:1] - luma
        b_chroma = satellite_img[:, 2:3] - luma
        chroma = torch.cat([r_chroma, b_chroma], dim=1)
        denom = chroma.flatten(2).abs().amax(dim=2).view(chroma.shape[0], 2, 1, 1).clamp_min(1e-6)
        return chroma / denom

    def _remap_street_semantic(self, semantic: torch.Tensor) -> torch.Tensor:
        """Map known DINO-v12 labels to unified NEOS-style ids.

        Unified order:
        0 sky, 1 road, 2 building, 3 grass, 4 tree, 5 car, 6 other.

        DINO-v12 order:
        0 road, 1 building, 2 grass, 3 tree, 4 car, 5 other, 6 sky.
        """
        semantic = semantic.long().clamp(0, self.num_semantic_classes - 1)
        if self.street_semantic_source.lower() not in {"dino_v12", "dino_v16"} or self.num_semantic_classes < 7:
            return semantic
        mapping = semantic.new_tensor([1, 2, 3, 4, 5, 6, 0])
        return mapping[semantic]

    def _remap_satellite_semantic(self, semantic: torch.Tensor) -> torch.Tensor:
        semantic = semantic.long()
        if self.satellite_semantic_source.lower() == "dinov3_v16" and self.num_semantic_classes >= 7:
            # v16 satellite DINOv3 heads use remote-sensing IDs:
            # 0 impervious, 1 building, 2 low vegetation, 3 tree,
            # 4 bare land, 5 water, 6 shadow, 7 other.
            # Convert to the unified street colorization IDs:
            # 0 sky, 1 road/impervious, 2 building, 3 grass/low vegetation,
            # 4 tree, 5 car, 6 other. Satellite has no sky/car.
            mapping = semantic.new_tensor([1, 2, 3, 4, 3, 6, 6, 6])
            return mapping[semantic.clamp(0, mapping.numel() - 1)]
        semantic = semantic.clamp(0, self.num_semantic_classes - 1)
        return semantic

    def _one_hot(
        self,
        semantic: Optional[torch.Tensor],
        size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        is_street: bool,
    ) -> torch.Tensor:
        if semantic is None:
            return torch.zeros(1, self.num_semantic_classes, *size, device=device, dtype=dtype)
        semantic = semantic.to(device=device, dtype=torch.long)
        semantic = self._remap_street_semantic(semantic) if is_street else self._remap_satellite_semantic(semantic)
        semantic = semantic.clamp(0, self.num_semantic_classes - 1)
        one_hot = F.one_hot(semantic, num_classes=self.num_semantic_classes).permute(0, 3, 1, 2).float()
        one_hot = one_hot.to(dtype=dtype)
        if one_hot.shape[-2:] != size:
            one_hot = F.interpolate(one_hot, size=size, mode="nearest")
        return one_hot

    @staticmethod
    def _semantic_boundary(one_hot: torch.Tensor) -> torch.Tensor:
        labels = one_hot.argmax(dim=1, keepdim=True).float()
        dx = F.pad(torch.abs(labels[:, :, :, 1:] - labels[:, :, :, :-1]), (0, 1, 0, 0))
        dy = F.pad(torch.abs(labels[:, :, 1:, :] - labels[:, :, :-1, :]), (0, 0, 0, 1))
        return ((dx + dy) > 0).float()

    def _replace_satellite_car_semantic(self, sat_one_hot: torch.Tensor) -> torch.Tensor:
        """Replace satellite car pixels with surrounding non-car semantics."""
        if self.num_semantic_classes <= self.car_class_id:
            return sat_one_hot
        car = sat_one_hot[:, self.car_class_id:self.car_class_id + 1]
        if torch.count_nonzero(car).item() == 0:
            return sat_one_hot

        static_ids = [
            idx
            for idx in range(self.num_semantic_classes)
            if idx not in (self.sky_class_id, self.car_class_id)
        ]
        if not static_ids:
            out = sat_one_hot.clone()
            out[:, self.car_class_id:self.car_class_id + 1] = 0
            return out

        static = sat_one_hot[:, static_ids]
        kernel = torch.ones(
            len(static_ids),
            1,
            7,
            7,
            device=sat_one_hot.device,
            dtype=sat_one_hot.dtype,
        )
        neighbor_scores = F.conv2d(static, kernel, padding=3, groups=len(static_ids))
        nearest_static = neighbor_scores.argmax(dim=1)
        replacement = F.one_hot(nearest_static, num_classes=len(static_ids)).permute(0, 3, 1, 2)
        replacement = replacement.to(dtype=sat_one_hot.dtype)

        out = sat_one_hot.clone()
        out[:, self.car_class_id:self.car_class_id + 1] = 0
        for local_idx, class_id in enumerate(static_ids):
            out[:, class_id:class_id + 1] = torch.where(
                car > 0.5,
                replacement[:, local_idx:local_idx + 1],
                out[:, class_id:class_id + 1],
            )
        return out / out.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _satellite_color_prototypes(
        self,
        satellite_img: torch.Tensor,
        satellite_semantic: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = satellite_img.shape
        sat_one_hot = self._one_hot(
            satellite_semantic,
            size=(h, w),
            device=satellite_img.device,
            dtype=satellite_img.dtype,
            is_street=False,
        )
        if sat_one_hot.shape[0] == 1 and b > 1:
            sat_one_hot = sat_one_hot.expand(b, -1, -1, -1)
        sat_one_hot = self._replace_satellite_car_semantic(sat_one_hot)

        # Force FP32 reductions under AMP. 512x512 masks can overflow FP16
        # during area summation, producing inf / inf -> NaN.
        with torch.autocast(device_type=satellite_img.device.type, enabled=False):
            weights = sat_one_hot.float()
            satellite_img_f = satellite_img.float()
            denom = weights.flatten(2).sum(dim=2).clamp_min(1e-6)
            prototypes = torch.einsum("bchw,bkhw->bkc", satellite_img_f, weights) / denom.unsqueeze(-1)

            global_mean = satellite_img_f.mean(dim=(2, 3))
            missing = denom <= 1e-5
            prototypes = torch.where(missing.unsqueeze(-1), global_mean.unsqueeze(1), prototypes)

            # There is no satellite sky. Use global mean as placeholder; sky branch
            # will override it with street-only refinement.
            if self.num_semantic_classes > 0:
                prototypes[:, 0, :] = global_mean
        return prototypes.to(dtype=satellite_img.dtype), sat_one_hot

    def _compose_prior(self, street_one_hot: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bkhw,bkc->bchw", street_one_hot, prototypes)

    def _token_semantic(self, one_hot: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        sem = F.adaptive_avg_pool2d(one_hot, out_size)
        return sem / sem.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _token_color_prior(
        self,
        satellite_img: torch.Tensor,
        satellite_one_hot: torch.Tensor,
        street_one_hot: torch.Tensor,
        gray_luma: torch.Tensor,
        gray_contrast: torch.Tensor,
        satellite_edge: torch.Tensor,
        fallback_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select satellite color tokens using street semantic + grayscale query."""
        b, _, h, w = street_one_hot.shape
        patch = max(int(self.token_selection_patch_size), 1)
        street_h = max(h // patch, 1)
        street_w = max(w // patch, 1)
        sat_h = max(satellite_img.shape[-2] // patch, 1)
        sat_w = max(satellite_img.shape[-1] // patch, 1)

        street_sem_t = self._token_semantic(street_one_hot, (street_h, street_w))
        sat_sem_t = self._token_semantic(satellite_one_hot, (sat_h, sat_w))
        street_gray_t = F.adaptive_avg_pool2d(gray_luma, (street_h, street_w))
        street_contrast_t = F.adaptive_avg_pool2d(gray_contrast, (street_h, street_w))
        sat_luma_t = F.adaptive_avg_pool2d(self._luma(satellite_img), (sat_h, sat_w))
        sat_edge_t = F.adaptive_avg_pool2d(satellite_edge, (sat_h, sat_w))
        sat_chroma_t = F.adaptive_avg_pool2d(self._satellite_chroma_features(satellite_img), (sat_h, sat_w))
        sat_rgb_t = F.adaptive_avg_pool2d(satellite_img, (sat_h, sat_w))

        q_in = torch.cat([street_sem_t, street_gray_t, street_contrast_t], dim=1)
        if self.use_satellite_chroma_token_selection:
            k_in = torch.cat([sat_sem_t, sat_chroma_t], dim=1)
        else:
            k_in = torch.cat([sat_sem_t, sat_luma_t, sat_edge_t], dim=1)
        q = self.street_token_query(q_in).flatten(2).transpose(1, 2)
        k = self.satellite_token_key(k_in).flatten(2)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=1)
        logits = torch.bmm(q, k) / (q.shape[-1] ** 0.5)

        street_sem_flat = street_sem_t.flatten(2).transpose(1, 2)
        sat_sem_flat = sat_sem_t.flatten(2)
        semantic_overlap = torch.bmm(street_sem_flat, sat_sem_flat)
        valid_same_semantic = semantic_overlap > 0.05

        # Sky has no satellite match. Its color prior is handled by sky_refine.
        if self.num_semantic_classes > 0:
            sky_prob = street_sem_flat[:, :, self.sky_class_id:self.sky_class_id + 1]
            valid_same_semantic = valid_same_semantic & (sky_prob < 0.5)
        logits = logits + 2.0 * semantic_overlap
        logits = logits.masked_fill(~valid_same_semantic, -1e4)

        no_valid = ~valid_same_semantic.any(dim=-1, keepdim=True)
        logits = torch.where(no_valid, torch.zeros_like(logits), logits)
        attn = torch.softmax(logits, dim=-1)

        sat_values = sat_rgb_t.flatten(2).transpose(1, 2)
        selected = torch.bmm(attn, sat_values)
        selected = selected.transpose(1, 2).reshape(b, 3, street_h, street_w)
        selected = F.interpolate(selected, size=(h, w), mode="bilinear", align_corners=False)

        no_valid_map = no_valid.float().transpose(1, 2).reshape(b, 1, street_h, street_w)
        no_valid_map = F.interpolate(no_valid_map, size=(h, w), mode="nearest")
        selected = torch.where(no_valid_map > 0.5, fallback_prior, selected)
        return selected, attn

    def forward(
        self,
        base_rgb: torch.Tensor,
        gray_rgb: torch.Tensor,
        satellite_img: torch.Tensor,
        street_semantic: Optional[torch.Tensor] = None,
        satellite_semantic: Optional[torch.Tensor] = None,
    ) -> dict:
        b, _, h, w = base_rgb.shape
        if satellite_img is None:
            satellite_img = torch.zeros(b, 3, 256, 256, device=base_rgb.device, dtype=base_rgb.dtype)

        street_one_hot = self._one_hot(
            street_semantic,
            size=(h, w),
            device=base_rgb.device,
            dtype=base_rgb.dtype,
            is_street=True,
        )
        if street_one_hot.shape[0] == 1 and b > 1:
            street_one_hot = street_one_hot.expand(b, -1, -1, -1)

        gray_edge = self._edge_map(gray_rgb)
        satellite_edge = self._edge_map(satellite_img)
        satellite_edge = F.interpolate(satellite_edge, size=(h, w), mode="bilinear", align_corners=False)
        semantic_boundary = self._semantic_boundary(street_one_hot)
        street_edge = gray_edge if self.use_street_gray_edges else semantic_boundary
        gray_luma = gray_rgb.mean(dim=1, keepdim=True)
        gray_contrast = self._local_contrast(gray_rgb)
        if not self.use_street_gray_modulation:
            gray_luma = torch.zeros_like(gray_luma)
            gray_contrast = torch.zeros_like(gray_contrast)

        prototypes, sat_one_hot = self._satellite_color_prototypes(satellite_img, satellite_semantic)
        class_prior_rgb = self._compose_prior(street_one_hot, prototypes)
        token_attention = None
        if self.use_gray_satellite_token_selection:
            satellite_prior_rgb, token_attention = self._token_color_prior(
                satellite_img,
                sat_one_hot,
                street_one_hot,
                gray_luma,
                gray_contrast,
                satellite_edge,
                class_prior_rgb,
            )
        else:
            satellite_prior_rgb = class_prior_rgb

        base_luma = self._luma(base_rgb)
        prior_luma = self._luma(satellite_prior_rgb)
        satellite_prior_chroma = satellite_prior_rgb - prior_luma
        satellite_luma_preserved = torch.clamp(base_luma + satellite_prior_chroma, 0.0, 1.0)

        sky_mask = (
            street_one_hot[:, self.sky_class_id:self.sky_class_id + 1]
            if self.num_semantic_classes > self.sky_class_id
            else torch.zeros_like(base_luma)
        )
        car_mask = (
            street_one_hot[:, self.car_class_id:self.car_class_id + 1]
            if self.num_semantic_classes > self.car_class_id
            else torch.zeros_like(base_luma)
        )
        street_only_mask = torch.clamp(sky_mask + car_mask, 0.0, 1.0)
        satellite_guided_mask = 1.0 - street_only_mask

        # The non-sky/car detail branch must not use street grayscale edges.
        # It receives only semantic boundaries, satellite color prior, satellite
        # edges and the satellite gate. This reduces direct gray-street -> color
        # learning outside the street-only sky/car branch.
        detail_edge = semantic_boundary
        detail_base_input = torch.cat(
            [base_luma, satellite_prior_rgb, detail_edge, satellite_edge, street_one_hot, semantic_boundary],
            dim=1,
        )
        gate_input = torch.cat([detail_base_input, gray_luma, gray_contrast], dim=1)
        satellite_gate = self.non_sky_gate(gate_input) * satellite_guided_mask
        satellite_gate = satellite_gate * self.satellite_prior_strength
        satellite_delta = (satellite_luma_preserved - base_rgb) * satellite_gate

        detail_input = torch.cat([detail_base_input, satellite_gate], dim=1)
        line_detail_delta = self.detail_delta(detail_input) * self.detail_scale
        line_detail_delta = line_detail_delta * semantic_boundary * satellite_guided_mask

        sky_delta = self.sky_refine(torch.cat([base_rgb, base_luma, street_edge], dim=1))
        sky_delta = sky_delta * self.detail_scale * street_only_mask

        delta = (satellite_delta + line_detail_delta + sky_delta) * self.residual_scale
        final_rgb = torch.clamp(base_rgb + delta, 0.0, 1.0)

        return {
            "final_rgb": final_rgb,
            "delta_color": final_rgb - base_rgb,
            "satellite_color_prototypes": prototypes,
            "satellite_color_prior": satellite_prior_rgb,
            "satellite_class_color_prior": class_prior_rgb,
            "satellite_token_attention": token_attention,
            "satellite_luma_preserved_prior": satellite_luma_preserved,
            "street_semantic": street_one_hot,
            "satellite_semantic": sat_one_hot,
            "street_edge": street_edge,
            "satellite_edge": satellite_edge,
            "gray_luma_gate_input": gray_luma,
            "gray_contrast_gate_input": gray_contrast,
            "semantic_boundary": semantic_boundary,
            "sky_mask": sky_mask,
            "car_mask": car_mask,
            "street_only_mask": street_only_mask,
            "non_sky_mask": satellite_guided_mask,
            "satellite_guided_mask": satellite_guided_mask,
            "satellite_gate": satellite_gate,
            "satellite_delta": satellite_delta,
            "line_detail_delta": line_detail_delta,
            "sky_delta": sky_delta,
        }
