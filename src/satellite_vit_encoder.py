"""ViT encoder for satellite road/ground color priors."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SatelliteViTColorEncoder(nn.Module):
    """
    Lightweight ViT for extracting color priors from overhead satellite images.

    This branch is intentionally satellite-only. It does not predict the street
    image directly; it provides a context vector plus a weak RGB prior that can
    guide residual color correction.
    """

    def __init__(
        self,
        input_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 4,
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
        self.color_prior_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 3),
            nn.Sigmoid(),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, satellite_img: torch.Tensor) -> dict:
        b, _, h, w = satellite_img.shape
        x = satellite_img
        if h != self.image_size or w != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = self.pos_drop(tokens + self.pos_embed)
        tokens = self.norm(self.transformer(tokens))
        pooled = tokens.mean(dim=1)

        features = tokens.transpose(1, 2).reshape(
            b,
            -1,
            self.grid_size,
            self.grid_size,
        )
        features = self.feature_proj(features)
        features = F.interpolate(features, size=(h, w), mode="bilinear", align_corners=False)

        color_prior = self.color_prior_head(pooled).view(b, 3, 1, 1).expand(b, 3, h, w)
        return {
            "context": self.context_proj(pooled),
            "features": features,
            "color_prior": color_prior,
        }
