"""
Residual Color Correction Module.

Learns small color corrections to improve DDColor output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCorrectionModule(nn.Module):
    """
    Residual color correction network.

    Takes base colorized output and context vector, outputs a small
    color correction delta.
    """

    def __init__(
        self,
        context_dim: int = 512,
        base_channels: int = 64,
        residual_scale: float = 0.1,
        use_film: bool = True,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.residual_scale = residual_scale
        self.use_film = use_film

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.enc1 = self._make_block(base_channels, base_channels * 2)
        self.enc2 = self._make_block(base_channels * 2, base_channels * 4)

        # Context fusion
        if use_film:
            from .film_module import FiLMLayer
            self.film1 = FiLMLayer(context_dim, base_channels * 2)
            self.film2 = FiLMLayer(context_dim, base_channels * 4)
        else:
            self.context_proj = nn.Linear(context_dim, base_channels * 4)

        # Decoder
        self.dec1 = self._make_block(base_channels * 4, base_channels * 2)
        self.dec2 = self._make_block(base_channels * 2, base_channels)

        # Output projection
        self.output_proj = nn.Conv2d(base_channels, 3, 3, padding=1)

        # Skip connection projection
        self.skip_proj = nn.Conv2d(3, base_channels, 1)

    def _make_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        base_rgb: torch.Tensor,
        context: torch.Tensor,
        gray_input: torch.Tensor = None,
    ) -> dict:
        """
        Compute residual color correction.

        Args:
            base_rgb: DDColor base output (B, 3, H, W), range [0, 1]
            context: Context vector from polar encoder (B, context_dim)
            gray_input: Optional grayscale input (B, 3, H, W)

        Returns:
            Dictionary with:
                - 'final_rgb': Corrected output
                - 'delta_color': Color correction delta
                - 'features': Intermediate features for debugging
        """
        # Initial projection
        x = self.input_proj(base_rgb)

        # Save skip connection
        skip1 = x

        # Encoder
        x = F.avg_pool2d(x, 2)
        x = self.enc1(x)

        if self.use_film:
            x = self.film1(x, context)

        skip2 = x
        x = F.avg_pool2d(x, 2)
        x = self.enc2(x)

        if self.use_film:
            x = self.film2(x, context)

        # Decoder
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + skip2
        x = self.dec1(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + skip1
        x = self.dec2(x)

        # Output residual
        delta = self.output_proj(x)

        # Scale the residual
        delta = delta * self.residual_scale

        # Alternatively use tanh to bound the residual
        # delta = torch.tanh(delta) * self.residual_scale

        # Apply residual correction
        final_rgb = base_rgb + delta

        # Clip to valid range
        final_rgb = torch.clamp(final_rgb, 0, 1)

        return {
            'final_rgb': final_rgb,
            'delta_color': delta,
        }


class LightCorrectionModule(nn.Module):
    """
    Lightweight residual correction module for faster training.
    """

    def __init__(
        self,
        context_dim: int = 512,
        hidden_dim: int = 64,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.residual_scale = residual_scale

        # Simple network
        # Input: RGB (3) + projected context features (hidden_dim/4 for spatial efficiency)
        context_spatial_dim = hidden_dim // 4
        self.conv1 = nn.Conv2d(3 + context_spatial_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 3, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # Context projection to spatial features
        self.context_proj = nn.Linear(context_dim, context_spatial_dim)
        from .film_module import FiLMLayer
        self.film1 = FiLMLayer(context_dim, hidden_dim)
        self.film2 = FiLMLayer(context_dim, hidden_dim)

    def forward(
        self,
        base_rgb: torch.Tensor,
        context: torch.Tensor,
    ) -> dict:
        """
        Compute lightweight residual correction.

        Args:
            base_rgb: DDColor base output (B, 3, H, W)
            context: Context vector (B, context_dim)

        Returns:
            Dictionary with final_rgb and delta_color
        """
        B, C, H, W = base_rgb.shape

        # Project context to spatial features
        context_feat = self.context_proj(context)  # (B, hidden_dim)
        context_feat = context_feat.view(B, -1, 1, 1).expand(B, -1, H, W)

        # Concatenate base RGB with context features
        x = torch.cat([base_rgb, context_feat], dim=1)

        # Process
        x = self.relu(self.film1(self.bn1(self.conv1(x)), context))
        x = self.relu(self.film2(self.bn2(self.conv2(x)), context))
        delta = torch.tanh(self.conv3(x)) * self.residual_scale

        # Apply correction
        final_rgb = torch.clamp(base_rgb + delta, 0, 1)

        return {
            'final_rgb': final_rgb,
            'delta_color': delta,
        }


class DetailAwareLightCorrectionModule(nn.Module):
    """
    Spatial residual correction module with ViT lane/detail features.

    Unlike global FiLM-only correction, this module receives a full-resolution
    detail feature map so it can modify small structures such as lane markings.
    """

    def __init__(
        self,
        context_dim: int = 512,
        hidden_dim: int = 64,
        detail_channels: int = 64,
        residual_scale: float = 0.15,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.detail_channels = detail_channels
        context_spatial_dim = hidden_dim // 4

        # Input: base RGB + context map + ViT detail map + ViT color prior.
        input_channels = 3 + context_spatial_dim + detail_channels + 3
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.out = nn.Conv2d(hidden_dim, 3, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        self.context_proj = nn.Linear(context_dim, context_spatial_dim)
        from .film_module import FiLMLayer
        self.film1 = FiLMLayer(context_dim, hidden_dim)
        self.film2 = FiLMLayer(context_dim, hidden_dim)
        self.film3 = FiLMLayer(context_dim, hidden_dim)

    def forward(
        self,
        base_rgb: torch.Tensor,
        context: torch.Tensor,
        detail_features: torch.Tensor = None,
        color_prior: torch.Tensor = None,
    ) -> dict:
        b, _, h, w = base_rgb.shape
        context_feat = self.context_proj(context).view(b, -1, 1, 1).expand(b, -1, h, w)

        if detail_features is None:
            detail_features = torch.zeros(
                b,
                self.detail_channels,
                h,
                w,
                device=base_rgb.device,
                dtype=base_rgb.dtype,
            )
        elif detail_features.shape[-2:] != (h, w):
            detail_features = F.interpolate(detail_features, size=(h, w), mode="bilinear", align_corners=False)

        if color_prior is None:
            color_prior = base_rgb
        elif color_prior.shape[-2:] != (h, w):
            color_prior = F.interpolate(color_prior, size=(h, w), mode="bilinear", align_corners=False)

        x = torch.cat([base_rgb, context_feat, detail_features, color_prior], dim=1)
        x = self.relu(self.film1(self.bn1(self.conv1(x)), context))
        x = self.relu(self.film2(self.bn2(self.conv2(x)), context))
        x = self.relu(self.film3(self.bn3(self.conv3(x)), context))
        delta = torch.tanh(self.out(x)) * self.residual_scale
        final_rgb = torch.clamp(base_rgb + delta, 0, 1)

        return {
            'final_rgb': final_rgb,
            'delta_color': delta,
        }


class TokenColorCorrectionModule(nn.Module):
    """Residual correction head with an explicit token-level color delta prior."""

    def __init__(
        self,
        context_dim: int = 512,
        hidden_dim: int = 64,
        detail_channels: int = 64,
        residual_scale: float = 0.25,
        token_delta_scale: float = 0.8,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.token_delta_scale = token_delta_scale
        self.detail_channels = detail_channels
        context_spatial_dim = hidden_dim // 4
        input_channels = 3 + 3 + 3 + context_spatial_dim + detail_channels

        self.context_proj = nn.Linear(context_dim, context_spatial_dim)
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.out = nn.Conv2d(hidden_dim, 3, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        from .film_module import FiLMLayer
        self.film1 = FiLMLayer(context_dim, hidden_dim)
        self.film2 = FiLMLayer(context_dim, hidden_dim)
        self.film3 = FiLMLayer(context_dim, hidden_dim)

    def forward(
        self,
        base_rgb: torch.Tensor,
        context: torch.Tensor,
        detail_features: torch.Tensor = None,
        color_prior: torch.Tensor = None,
        token_delta: torch.Tensor = None,
    ) -> dict:
        b, _, h, w = base_rgb.shape
        context_feat = self.context_proj(context).view(b, -1, 1, 1).expand(b, -1, h, w)

        if detail_features is None:
            detail_features = torch.zeros(
                b,
                self.detail_channels,
                h,
                w,
                device=base_rgb.device,
                dtype=base_rgb.dtype,
            )
        elif detail_features.shape[-2:] != (h, w):
            detail_features = F.interpolate(detail_features, size=(h, w), mode="bilinear", align_corners=False)

        if token_delta is None:
            token_delta = torch.zeros_like(base_rgb)
        elif token_delta.shape[-2:] != (h, w):
            token_delta = F.interpolate(token_delta, size=(h, w), mode="bilinear", align_corners=False)

        if color_prior is None:
            color_prior = torch.clamp(base_rgb + token_delta, 0, 1)
        elif color_prior.shape[-2:] != (h, w):
            color_prior = F.interpolate(color_prior, size=(h, w), mode="bilinear", align_corners=False)

        x = torch.cat([base_rgb, color_prior, token_delta, context_feat, detail_features], dim=1)
        x = self.relu(self.film1(self.bn1(self.conv1(x)), context))
        x = self.relu(self.film2(self.bn2(self.conv2(x)), context))
        x = self.relu(self.film3(self.bn3(self.conv3(x)), context))
        learned_delta = torch.tanh(self.out(x)) * self.residual_scale
        delta = token_delta * self.token_delta_scale + learned_delta
        final_rgb = torch.clamp(base_rgb + delta, 0, 1)
        return {
            "final_rgb": final_rgb,
            "delta_color": delta,
            "token_delta": token_delta,
            "learned_delta": learned_delta,
        }


class SemanticAwareCorrectionModule(nn.Module):
    """CNN residual head conditioned on street semantics and satellite context.

    This is the no-ViT path: semantic masks decide where corrections are applied,
    while the context vector summarizes satellite-side class ratios and colors.
    """

    def __init__(
        self,
        context_dim: int = 512,
        hidden_dim: int = 96,
        semantic_channels: int = 6,
        residual_scale: float = 0.25,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.semantic_channels = semantic_channels
        context_spatial_dim = hidden_dim // 4
        input_channels = 3 + 3 + semantic_channels + context_spatial_dim

        self.context_proj = nn.Linear(context_dim, context_spatial_dim)
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.out = nn.Conv2d(hidden_dim, 3, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.bn4 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        from .film_module import FiLMLayer
        self.film1 = FiLMLayer(context_dim, hidden_dim)
        self.film2 = FiLMLayer(context_dim, hidden_dim)
        self.film3 = FiLMLayer(context_dim, hidden_dim)
        self.film4 = FiLMLayer(context_dim, hidden_dim)

    def _empty_semantic(self, base_rgb: torch.Tensor) -> torch.Tensor:
        b, _, h, w = base_rgb.shape
        semantic = torch.zeros(
            b,
            self.semantic_channels,
            h,
            w,
            device=base_rgb.device,
            dtype=base_rgb.dtype,
        )
        semantic[:, -1:] = 1.0
        return semantic

    def forward(
        self,
        base_rgb: torch.Tensor,
        context: torch.Tensor,
        gray_input: torch.Tensor = None,
        street_semantic: torch.Tensor = None,
    ) -> dict:
        b, _, h, w = base_rgb.shape
        if gray_input is None:
            gray_input = base_rgb.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        elif gray_input.shape[-2:] != (h, w):
            gray_input = F.interpolate(gray_input, size=(h, w), mode="bilinear", align_corners=False)

        if street_semantic is None:
            street_semantic = self._empty_semantic(base_rgb)
        elif street_semantic.shape[-2:] != (h, w):
            street_semantic = F.interpolate(street_semantic.float(), size=(h, w), mode="nearest")

        context_feat = self.context_proj(context).view(b, -1, 1, 1).expand(b, -1, h, w)
        x = torch.cat([base_rgb, gray_input, street_semantic.float(), context_feat], dim=1)
        x = self.relu(self.film1(self.bn1(self.conv1(x)), context))
        residual = x
        x = self.relu(self.film2(self.bn2(self.conv2(x)), context))
        x = self.relu(self.film3(self.bn3(self.conv3(x)), context))
        x = self.relu(self.film4(self.bn4(self.conv4(x + residual)), context))
        delta = torch.tanh(self.out(x)) * self.residual_scale
        final_rgb = torch.clamp(base_rgb + delta, 0, 1)
        return {
            "final_rgb": final_rgb,
            "delta_color": delta,
        }


if __name__ == "__main__":
    # Test Residual Correction Module
    base_rgb = torch.randn(4, 3, 256, 256)
    context = torch.randn(4, 512)

    # Test full module
    correction = ResidualCorrectionModule(context_dim=512)
    result = correction(base_rgb, context)

    print(f"Base RGB shape: {base_rgb.shape}")
    print(f"Final RGB shape: {result['final_rgb'].shape}")
    print(f"Delta shape: {result['delta_color'].shape}")
    print(f"Delta range: [{result['delta_color'].min():.4f}, {result['delta_color'].max():.4f}]")

    # Test lightweight module
    light_correction = LightCorrectionModule(context_dim=512)
    result_light = light_correction(base_rgb, context)
    print(f"\nLightweight module:")
    print(f"Final RGB shape: {result_light['final_rgb'].shape}")
