"""Sky-only gray-region to satellite-chroma matching correction for v20.

This module keeps only a sky mask from street semantics. Non-sky color is
constrained to come from satellite chroma-region prototypes selected by
street gray-region tokens.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkyOnlyGrayChromaMatchCorrectionModule(nn.Module):
    """Correction module for v20 gray-region/chroma-region correspondence."""

    def __init__(
        self,
        num_gray_regions: int = 8,
        num_chroma_regions: int = 8,
        hidden_dim: int = 64,
        residual_scale: float = 1.0,
        satellite_prior_strength: float = 0.85,
        gray_temperature: float = 0.035,
        chroma_temperature: float = 0.045,
        token_dim: int = 48,
        detail_scale: float = 0.10,
        street_semantic_source: str = "dino_v16",
    ):
        super().__init__()
        self.num_gray_regions = int(num_gray_regions)
        self.num_chroma_regions = int(num_chroma_regions)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.satellite_prior_strength = float(satellite_prior_strength)
        self.gray_temperature = float(gray_temperature)
        self.chroma_temperature = float(chroma_temperature)
        self.token_dim = int(token_dim)
        self.detail_scale = float(detail_scale)
        self.street_semantic_source = street_semantic_source

        # Unified order used by the colorization model:
        # 0 sky, 1 road, 2 building, 3 grass, 4 tree, 5 car, 6 other.
        self.sky_class_id = 0

        gray_centers = torch.linspace(
            0.5 / self.num_gray_regions,
            1.0 - 0.5 / self.num_gray_regions,
            self.num_gray_regions,
        )
        self.register_buffer("gray_centers", gray_centers.view(1, self.num_gray_regions, 1, 1))

        chroma_centers = torch.tensor(
            [
                [0.00, 0.00],    # neutral
                [-0.14, -0.02],  # green / vegetation-like
                [0.14, -0.10],   # yellow / dry-ground-like
                [0.12, 0.08],    # warm brown/red
                [-0.08, 0.12],   # blue/cyan
                [-0.18, -0.14],  # deep green
                [0.18, -0.02],   # bright warm
                [-0.04, -0.16],  # yellow-green
            ],
            dtype=torch.float32,
        )
        if self.num_chroma_regions != chroma_centers.shape[0]:
            # Fallback to deterministic centers on a circle when K changes.
            theta = torch.linspace(0, 2 * torch.pi, self.num_chroma_regions + 1)[:-1]
            chroma_centers = torch.stack([0.16 * torch.cos(theta), 0.16 * torch.sin(theta)], dim=1)
        self.register_buffer("chroma_centers", chroma_centers.view(1, self.num_chroma_regions, 2))

        self.gray_query = nn.Sequential(
            nn.Linear(3, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )
        self.chroma_key = nn.Sequential(
            nn.Linear(6, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )

        # Gate predicts how strongly the satellite chroma prior should replace
        # DDColor chroma. It does not receive continuous street gray directly;
        # it receives only gray-region probabilities.
        gate_channels = 1 + 3 + self.num_gray_regions + 1
        self.non_sky_gate = nn.Sequential(
            nn.Conv2d(gate_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

        detail_channels = 1 + 3 + self.num_gray_regions + 1
        self.detail_delta = nn.Sequential(
            nn.Conv2d(detail_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, 3, padding=1),
            nn.Tanh(),
        )

        # Sky is the only street-only branch.
        self.sky_refine = nn.Sequential(
            nn.Conv2d(5, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 3, 3, padding=1),
            nn.Tanh(),
        )

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

    def _remap_street_semantic(self, semantic: torch.Tensor) -> torch.Tensor:
        semantic = semantic.long()
        if self.street_semantic_source.lower() in {"dino_v12", "dino_v16"}:
            # DINO order: 0 road, 1 building, 2 grass, 3 tree, 4 car, 5 other, 6 sky.
            mapping = semantic.new_tensor([1, 2, 3, 4, 5, 6, 0])
            return mapping[semantic.clamp(0, mapping.numel() - 1)]
        return semantic

    def _sky_mask(
        self,
        street_semantic: Optional[torch.Tensor],
        size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if street_semantic is None:
            return torch.zeros(1, 1, *size, device=device, dtype=dtype)
        sem = street_semantic.to(device=device, dtype=torch.long)
        sem = self._remap_street_semantic(sem)
        if sem.dim() == 4:
            sem = sem[:, 0]
        sky = (sem == self.sky_class_id).float().unsqueeze(1).to(dtype=dtype)
        if sky.shape[-2:] != size:
            sky = F.interpolate(sky, size=size, mode="nearest")
        return sky

    def _gray_region_probs(self, gray_rgb: torch.Tensor, non_sky: torch.Tensor) -> torch.Tensor:
        gray_luma = self._luma(gray_rgb)
        dist = (gray_luma - self.gray_centers.to(device=gray_rgb.device, dtype=gray_rgb.dtype)) ** 2
        probs = torch.softmax(-dist / max(self.gray_temperature, 1e-6), dim=1)
        probs = probs * non_sky
        return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _satellite_chroma(self, satellite_img: torch.Tensor) -> torch.Tensor:
        luma = self._luma(satellite_img)
        return torch.cat([satellite_img[:, 0:1] - luma, satellite_img[:, 2:3] - luma], dim=1)

    def _chroma_region_probs(self, satellite_img: torch.Tensor) -> torch.Tensor:
        chroma = self._satellite_chroma(satellite_img)
        c = chroma.permute(0, 2, 3, 1).unsqueeze(1)
        centers = self.chroma_centers.to(device=satellite_img.device, dtype=satellite_img.dtype)
        centers = centers.view(1, self.num_chroma_regions, 1, 1, 2)
        dist = ((c - centers) ** 2).sum(dim=-1)
        probs = torch.softmax(-dist / max(self.chroma_temperature, 1e-6), dim=1)
        return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _match_chroma_prior(
        self,
        gray_probs: torch.Tensor,
        satellite_img: torch.Tensor,
        chroma_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = gray_probs.shape
        with torch.autocast(device_type=satellite_img.device.type, enabled=False):
            gray_p = gray_probs.float()
            sat_p = chroma_probs.float()
            sat_img = satellite_img.float()

            gray_area = gray_p.flatten(2).mean(dim=2)
            gray_luma = self.gray_centers.to(device=gray_probs.device).view(1, self.num_gray_regions).float()
            gray_luma = gray_luma.expand(b, -1)
            gray_desc = torch.stack([gray_luma, gray_area, torch.sqrt(gray_area.clamp_min(1e-6))], dim=-1)

            sat_area = sat_p.flatten(2).mean(dim=2)
            denom = sat_p.flatten(2).sum(dim=2).clamp_min(1e-6)
            sat_proto = torch.einsum("bchw,bkhw->bkc", sat_img, sat_p) / denom.unsqueeze(-1)
            sat_luma = (sat_proto * sat_proto.new_tensor([0.299, 0.587, 0.114])).sum(dim=-1, keepdim=True)
            sat_chroma = torch.cat([sat_proto[:, :, 0:1] - sat_luma, sat_proto[:, :, 2:3] - sat_luma], dim=-1)
            sat_desc = torch.cat([sat_chroma, sat_proto, sat_area.unsqueeze(-1)], dim=-1)

        q = F.normalize(self.gray_query(gray_desc.to(dtype=gray_probs.dtype)), dim=-1)
        k = F.normalize(self.chroma_key(sat_desc.to(dtype=gray_probs.dtype)), dim=-1)
        logits = torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        logits = logits + torch.log(sat_area.to(dtype=logits.dtype).unsqueeze(1).clamp_min(1e-6))
        attn = torch.softmax(logits, dim=-1)
        selected_rgb = torch.bmm(attn, sat_proto.to(dtype=attn.dtype))
        prior = torch.einsum("bkhw,bkc->bchw", gray_probs, selected_rgb)
        return prior.to(dtype=satellite_img.dtype), attn, sat_proto.to(dtype=satellite_img.dtype)

    def forward(
        self,
        base_rgb: torch.Tensor,
        gray_rgb: torch.Tensor,
        satellite_img: torch.Tensor,
        street_semantic: Optional[torch.Tensor] = None,
        satellite_semantic: Optional[torch.Tensor] = None,
    ) -> dict:
        del satellite_semantic
        b, _, h, w = base_rgb.shape
        if satellite_img is None:
            satellite_img = torch.zeros(b, 3, 256, 256, device=base_rgb.device, dtype=base_rgb.dtype)
        if satellite_img.shape[0] == 1 and b > 1:
            satellite_img = satellite_img.expand(b, -1, -1, -1)

        sky = self._sky_mask(street_semantic, (h, w), base_rgb.device, base_rgb.dtype)
        if sky.shape[0] == 1 and b > 1:
            sky = sky.expand(b, -1, -1, -1)
        non_sky = 1.0 - sky

        gray_probs = self._gray_region_probs(gray_rgb, non_sky)
        chroma_probs = self._chroma_region_probs(satellite_img)
        satellite_prior_rgb, region_attention, chroma_prototypes = self._match_chroma_prior(
            gray_probs,
            satellite_img,
            chroma_probs,
        )

        base_luma = self._luma(base_rgb)
        prior_luma = self._luma(satellite_prior_rgb)
        satellite_prior_chroma = satellite_prior_rgb - prior_luma
        satellite_luma_preserved = torch.clamp(base_luma + satellite_prior_chroma, 0.0, 1.0)

        sat_edge = self._edge_map(satellite_img)
        if sat_edge.shape[-2:] != (h, w):
            sat_edge = F.interpolate(sat_edge, size=(h, w), mode="bilinear", align_corners=False)

        gate_in = torch.cat([base_luma, satellite_prior_rgb, gray_probs, sat_edge], dim=1)
        gate = self.non_sky_gate(gate_in) * non_sky * self.satellite_prior_strength
        satellite_delta = (satellite_luma_preserved - base_rgb) * gate

        detail_delta = self.detail_delta(gate_in) * self.detail_scale * non_sky
        street_edge = self._edge_map(gray_rgb)
        sky_delta = self.sky_refine(torch.cat([base_rgb, base_luma, street_edge], dim=1))
        sky_delta = sky_delta * self.detail_scale * sky

        delta = (satellite_delta + detail_delta + sky_delta) * self.residual_scale
        final_rgb = torch.clamp(base_rgb + delta, 0.0, 1.0)

        return {
            "final_rgb": final_rgb,
            "delta_color": final_rgb - base_rgb,
            "satellite_color_prior": satellite_prior_rgb,
            "satellite_luma_preserved_prior": satellite_luma_preserved,
            "satellite_color_prototypes": chroma_prototypes,
            "satellite_token_attention": region_attention,
            "gray_region_probs": gray_probs,
            "satellite_chroma_regions": chroma_probs,
            "street_semantic": torch.cat([sky, non_sky], dim=1),
            "satellite_semantic": chroma_probs,
            "sky_mask": sky,
            "street_only_mask": sky,
            "non_sky_mask": non_sky,
            "satellite_guided_mask": non_sky,
            "satellite_gate": gate,
            "satellite_delta": satellite_delta,
            "line_detail_delta": detail_delta,
            "sky_delta": sky_delta,
        }
