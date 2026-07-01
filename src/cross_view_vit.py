"""Cross-view ViT fusion between street-view patches and satellite images."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewViTEncoder(nn.Module):
    """
    Match street-view tokens with satellite tokens using cross-attention.

    A learnable no-match token is appended to the satellite tokens. Street-view
    regions that do not exist in the overhead view, especially sky, can attend
    to this token instead of forcing a false satellite correspondence.
    """

    def __init__(
        self,
        street_channels: int = 6,
        satellite_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 3,
        num_heads: int = 3,
        patch_size: int = 16,
        context_dim: int = 512,
        feature_channels: int = 64,
        image_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.feature_channels = feature_channels

        self.street_patch_embed = nn.Conv2d(
            street_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.satellite_patch_embed = nn.Conv2d(
            satellite_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_tokens = self.grid_size * self.grid_size
        self.street_pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.satellite_pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.no_match_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

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
        self.context_proj = nn.Sequential(
            nn.Linear(embed_dim, context_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
        )
        self.feature_proj = nn.Sequential(
            nn.Conv2d(embed_dim, feature_channels, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )
        self.color_refine = nn.Sequential(
            nn.Conv2d(3 + feature_channels, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        nn.init.trunc_normal_(self.street_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.satellite_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.no_match_token, std=0.02)

    def _resize_if_needed(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] == (self.image_size, self.image_size):
            return image
        return F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def _patch_colors(self, satellite_img: torch.Tensor) -> torch.Tensor:
        colors = F.avg_pool2d(
            satellite_img,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return colors.flatten(2).transpose(1, 2)

    def forward(
        self,
        gray_rgb: torch.Tensor,
        base_rgb: torch.Tensor,
        satellite_img: torch.Tensor,
    ) -> dict:
        b, _, h, w = gray_rgb.shape
        street_input = self._resize_if_needed(torch.cat([gray_rgb, base_rgb], dim=1))
        satellite_input = self._resize_if_needed(satellite_img)

        street_tokens = self.street_patch_embed(street_input).flatten(2).transpose(1, 2)
        satellite_tokens = self.satellite_patch_embed(satellite_input).flatten(2).transpose(1, 2)
        street_tokens = self.street_encoder(street_tokens + self.street_pos_embed)
        satellite_tokens = self.satellite_encoder(satellite_tokens + self.satellite_pos_embed)

        no_match = self.no_match_token.expand(b, -1, -1)
        satellite_tokens_with_null = torch.cat([satellite_tokens, no_match], dim=1)
        matched_tokens, attn = self.cross_attn(
            query=street_tokens,
            key=satellite_tokens_with_null,
            value=satellite_tokens_with_null,
            need_weights=True,
            average_attn_weights=False,
        )
        fused_tokens = self.fusion_norm(street_tokens + matched_tokens)

        features = fused_tokens.transpose(1, 2).reshape(
            b,
            -1,
            self.grid_size,
            self.grid_size,
        )
        features = self.feature_proj(features)
        features = F.interpolate(features, size=(h, w), mode="bilinear", align_corners=False)

        # Use cross-attention weights to collect satellite patch colors. The
        # no-match probability is not assigned any satellite color.
        attn_mean = attn.mean(dim=1)
        sat_attn = attn_mean[:, :, :-1]
        no_match_attention = attn_mean[:, :, -1]
        sat_color_tokens = self._patch_colors(satellite_input)
        matched_colors = torch.bmm(sat_attn, sat_color_tokens)
        matched_colors = matched_colors.transpose(1, 2).reshape(
            b,
            3,
            self.grid_size,
            self.grid_size,
        )
        matched_colors = F.interpolate(matched_colors, size=(h, w), mode="bilinear", align_corners=False)
        color_prior = self.color_refine(torch.cat([matched_colors, features], dim=1))

        no_match_map = no_match_attention.reshape(b, 1, self.grid_size, self.grid_size)
        no_match_map = F.interpolate(no_match_map, size=(h, w), mode="bilinear", align_corners=False)

        return {
            "context": self.context_proj(fused_tokens.mean(dim=1)),
            "features": features,
            "color_prior": color_prior,
            "no_match_map": no_match_map,
            "cross_attention": attn_mean,
        }
