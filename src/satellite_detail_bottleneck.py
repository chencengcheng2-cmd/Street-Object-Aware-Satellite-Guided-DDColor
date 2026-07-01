"""Satellite-detail bottleneck correction module for v15.

Compared with v14, this module keeps the same "street tells where, satellite
tells color" bottleneck, but adds an explicit road-detail branch:
- satellite road regions are scanned for yellow/white lane-color evidence;
- street road regions are scanned for thin line/detail candidates;
- lane/detail correction is only allowed where both sides provide evidence.

The goal is to make satellite changes affect the output more strongly while
still avoiding a free gray-street -> RGB mapping outside sky/car.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .satellite_color_bottleneck import SatelliteColorBottleneckCorrectionModule


class SatelliteDetailBottleneckCorrectionModule(SatelliteColorBottleneckCorrectionModule):
    """v15 correction module with explicit satellite lane/detail evidence."""

    def __init__(
        self,
        *args,
        lane_detail_strength: float = 0.45,
        satellite_dependency_boost: float = 1.35,
        lane_evidence_threshold: float = 0.002,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lane_detail_strength = lane_detail_strength
        self.satellite_dependency_boost = satellite_dependency_boost
        self.lane_evidence_threshold = lane_evidence_threshold

        lane_channels = 3 + 1 + 1 + 1 + self.num_semantic_classes
        self.lane_gate = nn.Sequential(
            nn.Conv2d(lane_channels, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, 1, 1),
            nn.Sigmoid(),
        )
        self.lane_delta = nn.Sequential(
            nn.Conv2d(lane_channels + 1, self.hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, 3, 3, padding=1),
            nn.Tanh(),
        )

        nn.init.constant_(self.lane_delta[-2].weight, 0.0)
        nn.init.constant_(self.lane_delta[-2].bias, 0.0)

    @staticmethod
    def _vertical_weight_like(mask: torch.Tensor) -> torch.Tensor:
        h = mask.shape[-2]
        y = torch.linspace(0.25, 1.0, h, device=mask.device, dtype=mask.dtype)
        return y.view(1, 1, h, 1)

    def _street_lane_candidate(
        self,
        base_rgb: torch.Tensor,
        street_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        """Road-internal thin/detail candidate map from street-side structure."""
        road = street_one_hot[:, 1:2] if self.num_semantic_classes > 1 else torch.zeros_like(base_rgb[:, :1])
        luma = self._luma(base_rgb)
        edge = self._edge_map(base_rgb)
        local = self._local_contrast(base_rgb)
        bright = torch.sigmoid((luma - 0.48) * 10.0)
        candidate = road * (0.55 * edge + 0.45 * local) * bright
        candidate = candidate * self._vertical_weight_like(candidate)
        return candidate.clamp(0.0, 1.0)

    def _satellite_lane_prior(
        self,
        satellite_img: torch.Tensor,
        sat_one_hot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return lane color prior, scalar evidence and visualization mask."""
        b, _, h, w = satellite_img.shape
        # Use FP32 for reductions under AMP. Large 512x512 masks overflow FP16
        # area sums and create NaNs in color prototypes.
        with torch.autocast(device_type=satellite_img.device.type, enabled=False):
            img_f = satellite_img.float()
            road = sat_one_hot[:, 1:2].float() if self.num_semantic_classes > 1 else torch.zeros(b, 1, h, w, device=satellite_img.device, dtype=torch.float32)
            r, g, bl = img_f[:, 0:1], img_f[:, 1:2], img_f[:, 2:3]
            max_rgb = img_f.max(dim=1, keepdim=True).values
            min_rgb = img_f.min(dim=1, keepdim=True).values

            yellow = (
                road
                * (r > 0.45).float()
                * (g > 0.35).float()
                * (bl < 0.40).float()
                * ((r - bl) > 0.12).float()
                * ((g - bl) > 0.08).float()
            )
            white = (
                road
                * (r > 0.62).float()
                * (g > 0.62).float()
                * (bl > 0.62).float()
                * ((max_rgb - min_rgb) < 0.14).float()
            )

            lane_mask = torch.clamp(yellow + white, 0.0, 1.0)
            road_area = road.flatten(2).sum(dim=2).clamp_min(1.0)
            lane_area = lane_mask.flatten(2).sum(dim=2)
            evidence = (lane_area / road_area).clamp(0.0, 1.0)
            evidence = torch.where(
                evidence > self.lane_evidence_threshold,
                evidence,
                torch.zeros_like(evidence),
            )

            lane_weight = lane_mask.flatten(2).sum(dim=2).clamp_min(1e-6)
            lane_color = torch.einsum("bchw,bkhw->bkc", img_f, lane_mask) / lane_weight.unsqueeze(-1)
            road_weight = road.flatten(2).sum(dim=2).clamp_min(1e-6)
            road_color = torch.einsum("bchw,bkhw->bkc", img_f, road) / road_weight.unsqueeze(-1)
            lane_color = torch.where((lane_area > 0).unsqueeze(-1), lane_color, road_color)

            lane_prior_rgb = lane_color[:, 0].view(b, 3, 1, 1).expand(-1, -1, h, w)
        return (
            lane_prior_rgb.to(dtype=satellite_img.dtype),
            evidence.view(b, 1, 1, 1).to(dtype=satellite_img.dtype),
            lane_mask.to(dtype=satellite_img.dtype),
        )

    def forward(
        self,
        base_rgb: torch.Tensor,
        gray_rgb: torch.Tensor,
        satellite_img: torch.Tensor,
        street_semantic: torch.Tensor | None = None,
        satellite_semantic: torch.Tensor | None = None,
    ) -> dict:
        b, _, h, w = base_rgb.shape
        street_one_hot = self._one_hot(
            street_semantic,
            size=(h, w),
            device=base_rgb.device,
            dtype=base_rgb.dtype,
            is_street=True,
        )
        if street_one_hot.shape[0] == 1 and b > 1:
            street_one_hot = street_one_hot.expand(b, -1, -1, -1)

        semantic_boundary = self._semantic_boundary(street_one_hot)
        satellite_edge = self._edge_map(satellite_img)
        if satellite_edge.shape[-2:] != (h, w):
            satellite_edge = F.interpolate(satellite_edge, size=(h, w), mode="bilinear", align_corners=False)

        gray_luma = gray_rgb.mean(dim=1, keepdim=True)
        gray_contrast = self._local_contrast(gray_rgb)
        if not self.use_street_gray_modulation:
            gray_luma = torch.zeros_like(gray_luma)
            gray_contrast = torch.zeros_like(gray_contrast)

        prototypes, sat_one_hot = self._satellite_color_prototypes(satellite_img, satellite_semantic)
        class_prior_rgb = self._compose_prior(street_one_hot, prototypes)
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
            satellite_prior_rgb, token_attention = class_prior_rgb, None

        base_luma = self._luma(base_rgb)
        prior_luma = self._luma(satellite_prior_rgb)
        satellite_prior_chroma = satellite_prior_rgb - prior_luma
        satellite_luma_preserved = torch.clamp(base_luma + satellite_prior_chroma, 0.0, 1.0)

        sky_mask = street_one_hot[:, self.sky_class_id:self.sky_class_id + 1]
        car_mask = street_one_hot[:, self.car_class_id:self.car_class_id + 1]
        street_only_mask = torch.clamp(sky_mask + car_mask, 0.0, 1.0)
        satellite_guided_mask = 1.0 - street_only_mask

        detail_edge = torch.clamp(semantic_boundary + 0.65 * satellite_edge, 0.0, 1.0)
        detail_base_input = torch.cat(
            [base_luma, satellite_prior_rgb, detail_edge, satellite_edge, street_one_hot, semantic_boundary],
            dim=1,
        )
        gate_input = torch.cat([detail_base_input, gray_luma, gray_contrast], dim=1)
        satellite_gate = self.non_sky_gate(gate_input) * satellite_guided_mask
        satellite_gate = satellite_gate * self.satellite_prior_strength * self.satellite_dependency_boost
        satellite_gate = satellite_gate.clamp(0.0, 1.0)
        satellite_delta = (satellite_luma_preserved - base_rgb) * satellite_gate

        detail_input = torch.cat([detail_base_input, satellite_gate], dim=1)
        line_detail_delta = self.detail_delta(detail_input) * self.detail_scale
        line_detail_delta = line_detail_delta * detail_edge * satellite_guided_mask

        sat_lane_prior, lane_evidence, sat_lane_mask = self._satellite_lane_prior(satellite_img, sat_one_hot)
        if sat_lane_prior.shape[-2:] != (h, w):
            sat_lane_prior = F.interpolate(sat_lane_prior, size=(h, w), mode="bilinear", align_corners=False)
        street_lane = self._street_lane_candidate(base_rgb, street_one_hot)
        lane_input = torch.cat([sat_lane_prior, street_lane, satellite_edge, semantic_boundary, street_one_hot], dim=1)
        lane_gate = self.lane_gate(lane_input) * street_lane * lane_evidence * satellite_guided_mask
        learned_lane_delta = self.lane_delta(torch.cat([lane_input, lane_gate], dim=1)) * lane_gate
        direct_lane_delta = (sat_lane_prior - base_rgb) * lane_gate
        lane_delta = (0.65 * direct_lane_delta + 0.35 * learned_lane_delta) * self.lane_detail_strength

        street_edge = self._edge_map(gray_rgb) if self.use_street_gray_edges else semantic_boundary
        sky_delta = self.sky_refine(torch.cat([base_rgb, base_luma, street_edge], dim=1))
        sky_delta = sky_delta * self.detail_scale * street_only_mask

        delta = (satellite_delta + line_detail_delta + lane_delta + sky_delta) * self.residual_scale
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
            "semantic_boundary": semantic_boundary,
            "sky_mask": sky_mask,
            "car_mask": car_mask,
            "street_only_mask": street_only_mask,
            "non_sky_mask": satellite_guided_mask,
            "satellite_guided_mask": satellite_guided_mask,
            "satellite_gate": satellite_gate,
            "satellite_delta": satellite_delta,
            "line_detail_delta": line_detail_delta,
            "lane_delta": lane_delta,
            "street_lane_candidate": street_lane,
            "satellite_lane_evidence": lane_evidence,
            "satellite_lane_mask": sat_lane_mask,
            "satellite_lane_prior": sat_lane_prior,
            "lane_gate": lane_gate,
            "sky_delta": sky_delta,
        }
