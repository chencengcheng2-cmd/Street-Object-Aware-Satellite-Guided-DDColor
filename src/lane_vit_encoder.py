"""ViT-based street-view detail encoder for lane marking color correction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaneViTDetailEncoder(nn.Module):
    """
    Lightweight Vision Transformer that keeps spatial detail features.

    The encoder reads the grayscale street patch together with the DDColor base
    result. It is intended to learn thin road markings and local ground details
    that are lost by global context encoders.
    """

    def __init__(
        self,
        input_channels: int = 6,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        patch_size: int = 16,
        output_channels: int = 64,
        image_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.output_channels = output_channels

        self.patch_embed = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size * self.grid_size, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.feature_proj = nn.Sequential(
            nn.Conv2d(embed_dim, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.color_prior_head = nn.Sequential(
            nn.Conv2d(output_channels, output_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels // 2, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, gray_rgb: torch.Tensor, base_rgb: torch.Tensor) -> dict:
        x = torch.cat([gray_rgb, base_rgb], dim=1)
        b, _, h, w = x.shape
        if h != self.image_size or w != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = self.pos_drop(tokens + self.pos_embed)
        tokens = self.norm(self.transformer(tokens))
        features = tokens.transpose(1, 2).reshape(
            b,
            -1,
            self.grid_size,
            self.grid_size,
        )
        features = self.feature_proj(features)
        features = F.interpolate(features, size=(h, w), mode="bilinear", align_corners=False)
        color_prior = self.color_prior_head(features)

        return {
            "lane_detail_features": features,
            "lane_color_prior": color_prior,
        }
