"""Semantic and color-aware token matching for cross-view color correction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticColorTokenMatcher(nn.Module):
    """Match street tokens to same-semantic satellite tokens with color bias.

    Street tokens are deliberately larger than satellite tokens. A street token
    needs enough context to represent road/grass/building regions under
    perspective, while satellite tokens stay small to preserve local color
    candidates. The module outputs a token-level color delta prior.
    """

    def __init__(
        self,
        street_channels: int = 6,
        satellite_channels: int = 3,
        num_classes: int = 7,
        embed_dim: int = 192,
        depth: int = 2,
        num_heads: int = 3,
        street_patch_size: int = 32,
        satellite_patch_size: int = 8,
        context_dim: int = 512,
        feature_channels: int = 64,
        image_size: int = 256,
        context_size: tuple = None,
        color_weight: float = 3.0,
        semantic_distribution_weight: float = 2.0,
        boundary_weight: float = 2.0,
        semantic_block_value: float = -1e4,
        token_delta_scale: float = 0.35,
        dropout: float = 0.1,
    ):
        super().__init__()
        if image_size % street_patch_size != 0:
            raise ValueError("image_size must be divisible by street_patch_size")
        if context_size is None:
            context_size = (image_size, image_size)
        if context_size[0] % satellite_patch_size != 0 or context_size[1] % satellite_patch_size != 0:
            raise ValueError("context_size must be divisible by satellite_patch_size")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.image_size = image_size
        self.context_size = tuple(context_size)
        self.street_patch_size = street_patch_size
        self.satellite_patch_size = satellite_patch_size
        self.street_grid_size = image_size // street_patch_size
        self.satellite_grid_h = self.context_size[0] // satellite_patch_size
        self.satellite_grid_w = self.context_size[1] // satellite_patch_size
        self.satellite_grid_size = self.satellite_grid_h * self.satellite_grid_w
        self.num_classes = num_classes
        self.feature_channels = feature_channels
        self.num_heads = num_heads
        self.color_weight = color_weight
        self.semantic_distribution_weight = semantic_distribution_weight
        self.boundary_weight = boundary_weight
        self.semantic_block_value = semantic_block_value
        self.token_delta_scale = token_delta_scale

        street_tokens = self.street_grid_size * self.street_grid_size
        satellite_tokens = self.satellite_grid_size

        self.street_patch_embed = nn.Conv2d(
            street_channels,
            embed_dim,
            kernel_size=street_patch_size,
            stride=street_patch_size,
        )
        self.satellite_patch_embed = nn.Conv2d(
            satellite_channels,
            embed_dim,
            kernel_size=satellite_patch_size,
            stride=satellite_patch_size,
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
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(embed_dim)

        self.token_delta_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + num_classes + 9, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 6),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(embed_dim + num_classes + 6, context_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
        )
        self.feature_proj = nn.Sequential(
            nn.Conv2d(embed_dim + num_classes + 3, feature_channels, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

        compatibility = torch.zeros(num_classes, num_classes, dtype=torch.bool)
        for class_id in range(num_classes):
            compatibility[class_id, class_id] = True
        # Sky has no satellite correspondence. It can only use no-match.
        compatibility[0, :] = False
        # Background is a weak fallback, but it should not dominate clear classes.
        compatibility[num_classes - 1, :] = True
        compatibility[:, num_classes - 1] = True
        compatibility[0, num_classes - 1] = False
        self.register_buffer("compatibility", compatibility, persistent=False)

        nn.init.trunc_normal_(self.street_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.satellite_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.no_match_token, std=0.02)

    def _resize_if_needed(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] == (self.image_size, self.image_size):
            return image
        return F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

    def _resize_context_if_needed(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] == self.context_size:
            return image
        return F.interpolate(image, size=self.context_size, mode="bilinear", align_corners=False)

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

    def _fallback_street_semantic(self, base_rgb: torch.Tensor) -> torch.Tensor:
        b, _, h, w = base_rgb.shape
        y = torch.linspace(0, 1, h, device=base_rgb.device, dtype=base_rgb.dtype).view(1, 1, h, 1)
        r, g, bl = base_rgb[:, 0:1], base_rgb[:, 1:2], base_rgb[:, 2:3]
        lum = base_rgb.mean(dim=1, keepdim=True)
        sat = base_rgb.max(dim=1, keepdim=True).values - base_rgb.min(dim=1, keepdim=True).values
        sky = (y < 0.42) & ((bl > r * 1.03) | (lum > 0.62))
        road = (y > 0.35) & (sat < 0.20) & (lum > 0.18) & (lum < 0.88) & (~sky)
        tree = (g > r * 1.08) & (g > bl * 1.05) & (sat > 0.10)
        low_veg = (g > r * 0.96) & (g >= bl * 0.92) & (lum > 0.15) & (~tree)
        building = (y < 0.85) & (~sky) & (~road) & (~tree) & (~low_veg) & (lum > 0.18)
        car = torch.zeros_like(lum, dtype=torch.bool)
        masks = torch.cat([sky, road, building, low_veg, tree, car], dim=1).float()
        clutter = (1.0 - masks.max(dim=1, keepdim=True).values).clamp(0, 1)
        return torch.cat([masks, clutter], dim=1)

    def _fallback_satellite_semantic(self, satellite_img: torch.Tensor) -> torch.Tensor:
        r, g, bl = satellite_img[:, 0:1], satellite_img[:, 1:2], satellite_img[:, 2:3]
        lum = satellite_img.mean(dim=1, keepdim=True)
        sat = satellite_img.max(dim=1, keepdim=True).values - satellite_img.min(dim=1, keepdim=True).values
        sky = torch.zeros_like(lum, dtype=torch.bool)
        road = (sat < 0.18) & (lum > 0.20) & (lum < 0.90)
        tree = (g > r * 1.08) & (g > bl * 1.05) & (sat > 0.10)
        low_veg = ((g > r * 0.96) & (g >= bl * 0.92) & (lum > 0.15)) & (~tree)
        building = (~road) & (~tree) & (~low_veg) & (lum > 0.22)
        car = torch.zeros_like(lum, dtype=torch.bool)
        masks = torch.cat([sky, road, building, low_veg, tree, car], dim=1).float()
        clutter = (1.0 - masks.max(dim=1, keepdim=True).values).clamp(0, 1)
        return torch.cat([masks, clutter], dim=1)

    def _patch_semantic(self, one_hot: torch.Tensor, patch_size: int) -> torch.Tensor:
        pooled = F.avg_pool2d(one_hot, kernel_size=patch_size, stride=patch_size)
        denom = pooled.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (pooled / denom).flatten(2).transpose(1, 2)

    def _patch_colors(self, image: torch.Tensor, patch_size: int) -> torch.Tensor:
        colors = F.avg_pool2d(image, kernel_size=patch_size, stride=patch_size)
        return colors.flatten(2).transpose(1, 2)

    def _attention_bias(
        self,
        street_sem: torch.Tensor,
        satellite_sem: torch.Tensor,
        street_colors: torch.Tensor,
        satellite_colors: torch.Tensor,
    ) -> torch.Tensor:
        b, ns, _ = street_sem.shape
        street_labels = street_sem.argmax(dim=-1)
        satellite_labels = satellite_sem.argmax(dim=-1)
        allowed = self.compatibility[street_labels.unsqueeze(-1), satellite_labels.unsqueeze(1)]

        color_dist = torch.cdist(street_colors.float(), satellite_colors.float(), p=2)
        color_bias = -self.color_weight * color_dist
        semantic_mix_bias = self.semantic_distribution_weight * torch.bmm(
            street_sem.float(),
            satellite_sem.float().transpose(1, 2),
        )
        street_boundary = 1.0 - street_sem.max(dim=-1, keepdim=True).values
        satellite_boundary = 1.0 - satellite_sem.max(dim=-1, keepdim=True).values.transpose(1, 2)
        boundary_bias = -self.boundary_weight * torch.abs(street_boundary - satellite_boundary)
        semantic_bias = torch.where(
            allowed,
            torch.zeros_like(color_bias),
            torch.full_like(color_bias, self.semantic_block_value),
        )
        no_match_bias = torch.zeros(b, ns, 1, device=street_sem.device, dtype=color_bias.dtype)
        sky = street_labels.eq(0).unsqueeze(-1)
        semantic_bias = torch.where(sky, torch.full_like(semantic_bias, self.semantic_block_value), semantic_bias)
        bias = torch.cat(
            [semantic_bias + color_bias + semantic_mix_bias + boundary_bias, no_match_bias],
            dim=-1,
        )
        return bias.repeat_interleave(self.num_heads, dim=0)

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
        satellite_input = self._resize_context_if_needed(satellite_img)

        if street_semantic is None:
            street_one_hot = self._fallback_street_semantic(street_input[:, 3:])
        else:
            street_one_hot = self._to_one_hot(street_semantic, self.image_size, self.image_size)
        if satellite_semantic is None:
            satellite_one_hot = self._fallback_satellite_semantic(satellite_input)
        else:
            satellite_one_hot = self._to_one_hot(satellite_semantic, self.context_size[0], self.context_size[1])

        street_sem_tokens = self._patch_semantic(street_one_hot, self.street_patch_size)
        satellite_sem_tokens = self._patch_semantic(satellite_one_hot, self.satellite_patch_size)
        street_colors = self._patch_colors(street_input[:, 3:], self.street_patch_size)
        satellite_colors = self._patch_colors(satellite_input, self.satellite_patch_size)

        street_tokens = self.street_patch_embed(street_input).flatten(2).transpose(1, 2)
        satellite_tokens = self.satellite_patch_embed(satellite_input).flatten(2).transpose(1, 2)
        street_tokens = street_tokens + self.street_pos_embed + self.street_semantic_proj(street_sem_tokens)
        satellite_tokens = satellite_tokens + self.satellite_pos_embed + self.satellite_semantic_proj(satellite_sem_tokens)
        street_tokens = self.street_encoder(street_tokens)
        satellite_tokens = self.satellite_encoder(satellite_tokens)

        no_match = self.no_match_token.expand(b, -1, -1)
        satellite_tokens_with_null = torch.cat([satellite_tokens, no_match], dim=1)
        attn_bias = self._attention_bias(street_sem_tokens, satellite_sem_tokens, street_colors, satellite_colors)
        matched_tokens, attn = self.cross_attn(
            query=street_tokens,
            key=satellite_tokens_with_null,
            value=satellite_tokens_with_null,
            attn_mask=attn_bias,
            need_weights=True,
            average_attn_weights=False,
        )
        fused_tokens = self.fusion_norm(street_tokens + matched_tokens)

        attn_mean = attn.mean(dim=1)
        sat_attn = attn_mean[:, :, :-1]
        no_match_attention = attn_mean[:, :, -1]
        matched_colors = torch.bmm(sat_attn, satellite_colors)
        color_diff = matched_colors - street_colors
        delta_input = torch.cat(
            [
                street_tokens,
                matched_tokens,
                street_sem_tokens,
                street_colors,
                matched_colors,
                color_diff,
            ],
            dim=-1,
        )
        delta_raw = self.token_delta_mlp(delta_input)
        delta_gate = torch.sigmoid(delta_raw[..., :3])
        learned_delta = torch.tanh(delta_raw[..., 3:]) * self.token_delta_scale
        token_delta = delta_gate * color_diff + learned_delta
        # Sky and no-match-heavy tokens should rely on DDColor unless the residual head learns otherwise.
        sky = street_sem_tokens.argmax(dim=-1).eq(0).unsqueeze(-1)
        token_delta = torch.where(sky, torch.zeros_like(token_delta), token_delta)
        token_delta = token_delta * (1.0 - no_match_attention.unsqueeze(-1)).clamp(0, 1)

        token_delta_map = token_delta.transpose(1, 2).reshape(
            b, 3, self.street_grid_size, self.street_grid_size
        )
        token_delta_map = F.interpolate(token_delta_map, size=(h, w), mode="bilinear", align_corners=False)
        color_prior = torch.clamp(base_rgb + token_delta_map, 0, 1)

        token_map = fused_tokens.transpose(1, 2).reshape(b, -1, self.street_grid_size, self.street_grid_size)
        street_sem_map = street_sem_tokens.transpose(1, 2).reshape(
            b, self.num_classes, self.street_grid_size, self.street_grid_size
        )
        token_delta_low = token_delta.transpose(1, 2).reshape(b, 3, self.street_grid_size, self.street_grid_size)
        features = self.feature_proj(torch.cat([token_map, street_sem_map, token_delta_low], dim=1))
        features = F.interpolate(features, size=(h, w), mode="bilinear", align_corners=False)

        context_token = torch.cat(
            [
                fused_tokens.mean(dim=1),
                street_sem_tokens.mean(dim=1),
                street_colors.mean(dim=1),
                matched_colors.mean(dim=1),
            ],
            dim=-1,
        )
        no_match_map = no_match_attention.reshape(b, 1, self.street_grid_size, self.street_grid_size)
        no_match_map = F.interpolate(no_match_map, size=(h, w), mode="bilinear", align_corners=False)

        return {
            "context": self.context_proj(context_token),
            "features": features,
            "color_prior": color_prior,
            "token_delta": token_delta_map,
            "matched_colors": matched_colors,
            "street_token_colors": street_colors,
            "no_match_map": no_match_map,
            "cross_attention": attn_mean,
            "street_semantic": street_one_hot,
            "satellite_semantic": satellite_one_hot,
        }
