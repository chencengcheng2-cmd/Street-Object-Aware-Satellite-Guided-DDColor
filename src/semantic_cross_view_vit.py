"""Semantic-guided cross-view ViT fusion.

Street-view tokens are intentionally larger than satellite tokens. The street
patch needs more context because a single street patch may contain sky, road,
objects, and perspective distortion; satellite tokens stay smaller to preserve
road markings and ground details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Project class order used by generated masks and fallback pseudo masks.
# 0 sky/no-match, 1 road, 2 vegetation, 3 building, 4 water, 5 other.


class SemanticGuidedCrossViewViTEncoder(nn.Module):
    """Cross-view ViT with semantic compatibility constraints."""

    def __init__(
        self,
        street_channels: int = 6,
        satellite_channels: int = 3,
        num_classes: int = 6,
        embed_dim: int = 192,
        depth: int = 3,
        num_heads: int = 3,
        street_patch_size: int = 16,
        satellite_patch_size: int = 8,
        context_dim: int = 512,
        feature_channels: int = 64,
        image_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if image_size % street_patch_size != 0:
            raise ValueError("image_size must be divisible by street_patch_size")
        if image_size % satellite_patch_size != 0:
            raise ValueError("image_size must be divisible by satellite_patch_size")

        self.image_size = image_size
        self.street_patch_size = street_patch_size
        self.satellite_patch_size = satellite_patch_size
        self.street_grid_size = image_size // street_patch_size
        self.satellite_grid_size = image_size // satellite_patch_size
        self.num_classes = num_classes
        self.feature_channels = feature_channels
        self.num_heads = num_heads

        street_tokens = self.street_grid_size * self.street_grid_size
        satellite_tokens = self.satellite_grid_size * self.satellite_grid_size

        self.street_patch_embed = nn.Conv2d(
            street_channels, embed_dim, kernel_size=street_patch_size, stride=street_patch_size
        )
        self.satellite_patch_embed = nn.Conv2d(
            satellite_channels, embed_dim, kernel_size=satellite_patch_size, stride=satellite_patch_size
        )
        self.street_pos_embed = nn.Parameter(torch.zeros(1, street_tokens, embed_dim))
        self.satellite_pos_embed = nn.Parameter(torch.zeros(1, satellite_tokens, embed_dim))
        self.no_match_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.street_semantic_proj = nn.Linear(num_classes, embed_dim)
        self.satellite_semantic_proj = nn.Linear(num_classes, embed_dim)

        street_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        satellite_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.street_encoder = nn.TransformerEncoder(street_layer, num_layers=depth)
        self.satellite_encoder = nn.TransformerEncoder(satellite_layer, num_layers=depth)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.fusion_norm = nn.LayerNorm(embed_dim)

        self.context_proj = nn.Sequential(
            nn.Linear(embed_dim, context_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
        )
        self.feature_proj = nn.Sequential(
            nn.Conv2d(embed_dim + num_classes, feature_channels, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )
        self.color_refine = nn.Sequential(
            nn.Conv2d(3 + feature_channels + num_classes, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        compatibility = torch.zeros(num_classes, num_classes, dtype=torch.bool)
        # sky has no overhead correspondence. It will use the no-match token.
        compatibility[1, 1] = True  # road -> road
        compatibility[2, 2] = True  # vegetation -> vegetation
        compatibility[3, 3] = True  # building -> building
        compatibility[4, 4] = True  # water -> water
        compatibility[5, :] = True  # other can look broadly
        compatibility[:, 5] = True  # satellite other is a weak fallback
        self.register_buffer("compatibility", compatibility, persistent=False)

        nn.init.trunc_normal_(self.street_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.satellite_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.no_match_token, std=0.02)

    def _resize_if_needed(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] == (self.image_size, self.image_size):
            return image
        return F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

    def _to_one_hot(self, semantic: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if semantic is None:
            raise ValueError("semantic tensor is required")
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

    def _fallback_street_semantic(self, gray_rgb: torch.Tensor, base_rgb: torch.Tensor) -> torch.Tensor:
        b, _, h, w = gray_rgb.shape
        y_grid = torch.linspace(0, 1, h, device=gray_rgb.device, dtype=gray_rgb.dtype).view(1, 1, h, 1)
        r, g, bl = base_rgb[:, 0:1], base_rgb[:, 1:2], base_rgb[:, 2:3]
        lum = base_rgb.mean(dim=1, keepdim=True)
        sat = base_rgb.max(dim=1, keepdim=True).values - base_rgb.min(dim=1, keepdim=True).values

        sky = (y_grid < 0.42) & ((bl > r * 1.03) | (lum > 0.62))
        vegetation = (g > r * 1.05) & (g > bl * 1.04) & (lum > 0.15)
        water = (bl > r * 1.08) & (bl >= g * 0.95) & (y_grid > 0.35)
        road = (y_grid > 0.45) & (sat < 0.22) & (lum > 0.18) & (lum < 0.82)
        building = (y_grid < 0.78) & (~sky) & (~vegetation) & (~water) & (~road)

        masks = torch.cat([sky, road, vegetation, building, water], dim=1).float()
        other = (1.0 - masks.max(dim=1, keepdim=True).values).clamp(0, 1)
        return torch.cat([masks, other], dim=1)

    def _fallback_satellite_semantic(self, satellite_img: torch.Tensor) -> torch.Tensor:
        r, g, bl = satellite_img[:, 0:1], satellite_img[:, 1:2], satellite_img[:, 2:3]
        lum = satellite_img.mean(dim=1, keepdim=True)
        sat = satellite_img.max(dim=1, keepdim=True).values - satellite_img.min(dim=1, keepdim=True).values

        sky = torch.zeros_like(lum, dtype=torch.bool)
        vegetation = (g > r * 1.08) & (g > bl * 1.05) & (lum > 0.18)
        water = (bl > r * 1.12) & (bl > g * 1.02) & (lum > 0.12)
        road = (sat < 0.16) & (lum > 0.20) & (lum < 0.78) & (~water)
        building = (~vegetation) & (~water) & (~road) & (lum > 0.22)

        masks = torch.cat([sky, road, vegetation, building, water], dim=1).float()
        other = (1.0 - masks.max(dim=1, keepdim=True).values).clamp(0, 1)
        return torch.cat([masks, other], dim=1)

    def _patch_semantic(self, one_hot: torch.Tensor, patch_size: int) -> torch.Tensor:
        pooled = F.avg_pool2d(one_hot, kernel_size=patch_size, stride=patch_size)
        return pooled.flatten(2).transpose(1, 2)

    def _patch_colors(self, satellite_img: torch.Tensor) -> torch.Tensor:
        colors = F.avg_pool2d(satellite_img, kernel_size=self.satellite_patch_size, stride=self.satellite_patch_size)
        return colors.flatten(2).transpose(1, 2)

    def _semantic_attn_mask(self, street_sem_tokens: torch.Tensor, satellite_sem_tokens: torch.Tensor) -> torch.Tensor:
        b, ns, _ = street_sem_tokens.shape
        nt = satellite_sem_tokens.shape[1]
        street_labels = street_sem_tokens.argmax(dim=-1)
        satellite_labels = satellite_sem_tokens.argmax(dim=-1)
        allowed = self.compatibility[street_labels.unsqueeze(-1), satellite_labels.unsqueeze(1)]
        no_match_allowed = torch.ones(b, ns, 1, device=allowed.device, dtype=torch.bool)
        # Sky tokens should only use no-match; this prevents satellite ground colors from leaking into sky.
        sky = street_labels.eq(0).unsqueeze(-1)
        allowed = torch.where(sky, torch.zeros_like(allowed), allowed)
        allowed = torch.cat([allowed, no_match_allowed], dim=-1)
        blocked = ~allowed
        return blocked.repeat_interleave(self.num_heads, dim=0)

    def forward(
        self,
        gray_rgb: torch.Tensor,
        base_rgb: torch.Tensor,
        satellite_img: torch.Tensor,
        street_semantic: torch.Tensor = None,
        satellite_semantic: torch.Tensor = None,
    ) -> dict:
        b, _, h, w = gray_rgb.shape
        street_input = self._resize_if_needed(torch.cat([gray_rgb, base_rgb], dim=1))
        satellite_input = self._resize_if_needed(satellite_img)

        if street_semantic is None:
            street_one_hot = self._fallback_street_semantic(street_input[:, :3], street_input[:, 3:])
        else:
            street_one_hot = self._to_one_hot(street_semantic, self.image_size, self.image_size)
        if satellite_semantic is None:
            satellite_one_hot = self._fallback_satellite_semantic(satellite_input)
        else:
            satellite_one_hot = self._to_one_hot(satellite_semantic, self.image_size, self.image_size)

        street_sem_tokens = self._patch_semantic(street_one_hot, self.street_patch_size)
        satellite_sem_tokens = self._patch_semantic(satellite_one_hot, self.satellite_patch_size)

        street_tokens = self.street_patch_embed(street_input).flatten(2).transpose(1, 2)
        satellite_tokens = self.satellite_patch_embed(satellite_input).flatten(2).transpose(1, 2)
        street_tokens = street_tokens + self.street_pos_embed + self.street_semantic_proj(street_sem_tokens)
        satellite_tokens = satellite_tokens + self.satellite_pos_embed + self.satellite_semantic_proj(satellite_sem_tokens)
        street_tokens = self.street_encoder(street_tokens)
        satellite_tokens = self.satellite_encoder(satellite_tokens)

        no_match = self.no_match_token.expand(b, -1, -1)
        satellite_tokens_with_null = torch.cat([satellite_tokens, no_match], dim=1)
        attn_mask = self._semantic_attn_mask(street_sem_tokens, satellite_sem_tokens)
        matched_tokens, attn = self.cross_attn(
            query=street_tokens,
            key=satellite_tokens_with_null,
            value=satellite_tokens_with_null,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        fused_tokens = self.fusion_norm(street_tokens + matched_tokens)

        token_map = fused_tokens.transpose(1, 2).reshape(b, -1, self.street_grid_size, self.street_grid_size)
        street_sem_map = street_sem_tokens.transpose(1, 2).reshape(
            b, self.num_classes, self.street_grid_size, self.street_grid_size
        )
        features = self.feature_proj(torch.cat([token_map, street_sem_map], dim=1))
        features = F.interpolate(features, size=(h, w), mode="bilinear", align_corners=False)

        attn_mean = attn.mean(dim=1)
        sat_attn = attn_mean[:, :, :-1]
        no_match_attention = attn_mean[:, :, -1]
        sat_color_tokens = self._patch_colors(satellite_input)
        matched_colors = torch.bmm(sat_attn, sat_color_tokens)
        matched_colors = matched_colors.transpose(1, 2).reshape(b, 3, self.street_grid_size, self.street_grid_size)
        matched_colors = F.interpolate(matched_colors, size=(h, w), mode="bilinear", align_corners=False)
        street_sem_full = F.interpolate(street_sem_map, size=(h, w), mode="nearest")
        color_prior = self.color_refine(torch.cat([matched_colors, features, street_sem_full], dim=1))

        no_match_map = no_match_attention.reshape(b, 1, self.street_grid_size, self.street_grid_size)
        no_match_map = F.interpolate(no_match_map, size=(h, w), mode="bilinear", align_corners=False)

        return {
            "context": self.context_proj(fused_tokens.mean(dim=1)),
            "features": features,
            "color_prior": color_prior,
            "no_match_map": no_match_map,
            "cross_attention": attn_mean,
            "street_semantic": street_one_hot,
            "satellite_semantic": satellite_one_hot,
        }
