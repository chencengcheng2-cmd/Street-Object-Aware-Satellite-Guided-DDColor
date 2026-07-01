"""
Polar Context Encoder using ResNet18 backbone.

Extracts global scene context from polar satellite views.
"""

import torch
import torch.nn as nn
from torchvision import models


class PolarContextEncoder(nn.Module):
    """Encode polar satellite views into context vectors."""

    def __init__(
        self,
        context_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        input_channels: int = 3,
    ):
        super().__init__()
        self.context_dim = context_dim

        # Load ResNet18 backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Modify first layer if input channels != 3
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Up to avgpool

        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(512, context_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(context_dim, context_dim),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, polar_img: torch.Tensor) -> torch.Tensor:
        """
        Extract context vector from polar image.

        Args:
            polar_img: Polar image (B, C, H, W), range [0, 1]

        Returns:
            Context vector (B, context_dim)
        """
        # Extract features
        features = self.backbone(polar_img)  # (B, 512, H', W')

        # Global average pooling
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)  # (B, 512, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, 512)

        # Project to context dimension
        context = self.fc(pooled)  # (B, context_dim)

        return context

    def get_num_parameters(self) -> tuple:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == "__main__":
    # Test Polar Context Encoder
    encoder = PolarContextEncoder(context_dim=512, pretrained=False)
    print(f"Total parameters: {encoder.get_num_parameters()[0]:,}")
    print(f"Trainable parameters: {encoder.get_num_parameters()[1]:,}")

    # Test forward
    x = torch.randn(2, 3, 256, 512)
    context = encoder(x)
    print(f"Input shape: {x.shape}, Output shape: {context.shape}")