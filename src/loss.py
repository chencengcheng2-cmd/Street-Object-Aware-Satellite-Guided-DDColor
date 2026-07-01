"""
Loss functions for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class L1Loss(nn.Module):
    """L1 loss (mean absolute error)."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        return F.l1_loss(pred, target, reduction=self.reduction)


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss."""

    def __init__(
        self,
        vgg_type: str = 'vgg16',
        layer_weights: dict = None,
        use_input_norm: bool = True,
        range_norm: bool = False,
    ):
        super().__init__()

        if layer_weights is None:
            layer_weights = {
                'conv3_3': 0.25,
                'conv4_3': 0.5,
                'conv5_3': 1.0,
            }

        self.layer_weights = layer_weights
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        # Load VGG
        if vgg_type == 'vgg16':
            vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if self.use_input_norm else None)
        elif vgg_type == 'vgg19':
            vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT if self.use_input_norm else None)
        else:
            raise ValueError(f"Unknown VGG type: {vgg_type}")

        # Extract feature layers
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:36])  # Up to conv5_3

        # Freeze VGG
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # Register mean/std for normalization
        if use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Layer indices for VGG16_bn
        self.layer_indices = {
            'conv3_3': 16,
            'conv4_3': 23,
            'conv5_3': 30,
        }

    def forward(self, pred, target):
        """Compute perceptual loss."""
        # Normalize inputs
        if self.range_norm:
            pred = (pred + 1) / 2
            target = (target + 1) / 2

        if self.use_input_norm:
            pred = torch.clamp(pred, 0, 1)
            target = torch.clamp(target, 0, 1)
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std

        # Extract features
        with torch.no_grad():
            target_features = self._extract_features(target)

        pred_features = self._extract_features(pred)

        # Compute loss
        loss = 0
        for layer_name, weight in self.layer_weights.items():
            if layer_name in pred_features and layer_name in target_features:
                loss += F.l1_loss(
                    pred_features[layer_name],
                    target_features[layer_name],
                    reduction='mean'
                ) * weight

        return loss

    def _extract_features(self, x):
        """Extract features from specific layers."""
        features = {}
        x = x
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            for layer_name, idx in self.layer_indices.items():
                if i == idx:
                    features[layer_name] = x
        return features


class ResidualRegularization(nn.Module):
    """Regularization to keep residual corrections small."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, delta):
        """Compute L1 loss on delta to encourage small corrections."""
        return F.l1_loss(delta, torch.zeros_like(delta), reduction=self.reduction)


class LaneMarkingLoss(nn.Module):
    """Weighted L1 loss for yellow/white road marking pixels."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    @torch.no_grad()
    def _create_mask(self, target: torch.Tensor) -> torch.Tensor:
        r = target[:, 0:1]
        g = target[:, 1:2]
        b = target[:, 2:3]
        max_rgb = target.max(dim=1, keepdim=True).values
        min_rgb = target.min(dim=1, keepdim=True).values

        yellow = (r > 0.45) & (g > 0.35) & (b < 0.38) & ((r - b) > 0.12) & ((g - b) > 0.10)
        white = (r > 0.62) & (g > 0.62) & (b > 0.62) & ((max_rgb - min_rgb) < 0.14)
        mask = (yellow | white).float()

        # Road markings are more likely in the lower part of the street-view patch.
        h = target.shape[-2]
        vertical_weight = torch.linspace(0.25, 1.0, h, device=target.device, dtype=target.dtype)
        vertical_weight = vertical_weight.view(1, 1, h, 1)
        return mask * vertical_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self._create_mask(target)
        denom = mask.sum() * pred.shape[1] + self.eps
        if denom.item() <= self.eps:
            return pred.new_tensor(0.0)
        return (torch.abs(pred - target) * mask).sum() / denom


class EdgeLoss(nn.Module):
    """Sobel edge L1 loss for thin structure preservation."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _edges(self, image: torch.Tensor) -> torch.Tensor:
        gray = image.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._edges(pred), self._edges(target), reduction='mean')


class ChromaLoss(nn.Module):
    """L1 loss on chroma components after removing per-pixel luminance."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_chroma = pred - pred.mean(dim=1, keepdim=True)
        target_chroma = target - target.mean(dim=1, keepdim=True)
        return F.l1_loss(pred_chroma, target_chroma, reduction='mean')


class ColorfulRegionLoss(nn.Module):
    """Extra chroma supervision on naturally colorful target regions."""

    def __init__(self, saturation_threshold: float = 0.18, eps: float = 1e-6):
        super().__init__()
        self.saturation_threshold = saturation_threshold
        self.eps = eps

    @staticmethod
    def _saturation(rgb: torch.Tensor) -> torch.Tensor:
        max_rgb = rgb.max(dim=1, keepdim=True).values
        min_rgb = rgb.min(dim=1, keepdim=True).values
        return (max_rgb - min_rgb) / (max_rgb + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = (self._saturation(target) > self.saturation_threshold).float()
        denom = mask.sum() * pred.shape[1] + self.eps
        if denom.item() <= self.eps:
            return pred.new_tensor(0.0)
        pred_chroma = pred - pred.mean(dim=1, keepdim=True)
        target_chroma = target - target.mean(dim=1, keepdim=True)
        return (torch.abs(pred_chroma - target_chroma) * mask).sum() / denom


class EdgeChromaLoss(nn.Module):
    """Chroma loss weighted by target edges for small colored structures."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.edge_loss = EdgeLoss()
        self.eps = eps

    def _edges(self, image: torch.Tensor) -> torch.Tensor:
        return self.edge_loss._edges(image)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        edge = self._edges(target)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        pred_chroma = pred - pred.mean(dim=1, keepdim=True)
        target_chroma = target - target.mean(dim=1, keepdim=True)
        denom = edge.sum() * pred.shape[1] + self.eps
        return (torch.abs(pred_chroma - target_chroma) * edge).sum() / denom


class FineDetailColorLoss(nn.Module):
    """Color loss focused on thin structures and local detail regions.

    This loss is stronger around target edges, local contrast changes, and
    naturally saturated small regions. It is intended for road markings,
    object boundaries, vegetation edges, building contours, and other details
    that are easily washed out by global color correction.
    """

    def __init__(
        self,
        edge_gain: float = 2.0,
        saturation_gain: float = 1.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.edge_gain = edge_gain
        self.saturation_gain = saturation_gain
        self.eps = eps
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        laplace = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplace", laplace)

    @staticmethod
    def _saturation(rgb: torch.Tensor) -> torch.Tensor:
        max_rgb = rgb.max(dim=1, keepdim=True).values
        min_rgb = rgb.min(dim=1, keepdim=True).values
        return (max_rgb - min_rgb) / (max_rgb + 1e-6)

    def _detail_weight(self, target: torch.Tensor) -> torch.Tensor:
        gray = target.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)

        lap = torch.abs(F.conv2d(gray, self.laplace, padding=1))
        lap = lap / (lap.amax(dim=(2, 3), keepdim=True) + 1e-6)

        saturation = self._saturation(target)
        sat_weight = torch.clamp(saturation / 0.25, 0.0, 1.0)

        detail = torch.clamp(edge + 0.75 * lap, 0.0, 1.0)
        return 1.0 + self.edge_gain * detail + self.saturation_gain * sat_weight * detail

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self._detail_weight(target).detach()
        rgb_loss = torch.abs(pred - target).mean(dim=1, keepdim=True)
        pred_chroma = pred - pred.mean(dim=1, keepdim=True)
        target_chroma = target - target.mean(dim=1, keepdim=True)
        chroma_loss = torch.abs(pred_chroma - target_chroma).mean(dim=1, keepdim=True)
        return ((rgb_loss + chroma_loss) * weight).sum() / weight.sum().clamp_min(self.eps)


class TotalLoss(nn.Module):
    """Combined loss for training."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        residual_weight: float = 0.1,
        lane_marking_weight: float = 0.0,
        edge_weight: float = 0.0,
        chroma_weight: float = 0.0,
        colorful_weight: float = 0.0,
        edge_chroma_weight: float = 0.0,
        fine_detail_color_weight: float = 0.0,
        base_improvement_weight: float = 0.0,
        use_perceptual: bool = True,
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.residual_weight = residual_weight
        self.lane_marking_weight = lane_marking_weight
        self.edge_weight = edge_weight
        self.chroma_weight = chroma_weight
        self.colorful_weight = colorful_weight
        self.edge_chroma_weight = edge_chroma_weight
        self.fine_detail_color_weight = fine_detail_color_weight
        self.base_improvement_weight = base_improvement_weight
        self.use_perceptual = use_perceptual

        self.l1_loss = L1Loss()
        self.residual_loss = ResidualRegularization()
        self.lane_marking_loss = LaneMarkingLoss()
        self.edge_loss = EdgeLoss()
        self.chroma_loss = ChromaLoss()
        self.colorful_loss = ColorfulRegionLoss()
        self.edge_chroma_loss = EdgeChromaLoss()
        self.fine_detail_color_loss = FineDetailColorLoss()

        if use_perceptual:
            try:
                self.perceptual_loss = PerceptualLoss()
            except Exception as e:
                print(f"Warning: Could not load perceptual loss: {e}")
                self.perceptual_loss = None
                self.use_perceptual = False
        else:
            self.perceptual_loss = None

    def forward(self, pred, target, delta=None, base=None) -> dict:
        """
        Compute total loss.

        Args:
            pred: Predicted RGB (B, 3, H, W)
            target: Ground truth RGB (B, 3, H, W)
            delta: Color correction delta (B, 3, H, W), optional

        Returns:
            Dictionary with individual losses and total
        """
        losses = {}

        # L1 loss
        l1 = self.l1_loss(pred, target)
        losses['l1'] = l1

        # Perceptual loss
        if self.use_perceptual and self.perceptual_loss is not None:
            try:
                perceptual = self.perceptual_loss(pred, target)
                losses['perceptual'] = perceptual
            except Exception as e:
                print(f"Warning: Perceptual loss computation failed: {e}")
                losses['perceptual'] = torch.tensor(0.0, device=pred.device)
        else:
            losses['perceptual'] = torch.tensor(0.0, device=pred.device)

        # Residual regularization
        if delta is not None:
            residual = self.residual_loss(delta)
            losses['residual'] = residual
        else:
            losses['residual'] = torch.tensor(0.0, device=pred.device)

        if self.lane_marking_weight > 0:
            losses['lane_marking'] = self.lane_marking_loss(pred, target)
        else:
            losses['lane_marking'] = torch.tensor(0.0, device=pred.device)

        if self.edge_weight > 0:
            losses['edge'] = self.edge_loss(pred, target)
        else:
            losses['edge'] = torch.tensor(0.0, device=pred.device)

        if self.chroma_weight > 0:
            losses['chroma'] = self.chroma_loss(pred, target)
        else:
            losses['chroma'] = torch.tensor(0.0, device=pred.device)

        if self.colorful_weight > 0:
            losses['colorful'] = self.colorful_loss(pred, target)
        else:
            losses['colorful'] = torch.tensor(0.0, device=pred.device)

        if self.edge_chroma_weight > 0:
            losses['edge_chroma'] = self.edge_chroma_loss(pred, target)
        else:
            losses['edge_chroma'] = torch.tensor(0.0, device=pred.device)

        if self.fine_detail_color_weight > 0:
            losses['fine_detail_color'] = self.fine_detail_color_loss(pred, target)
        else:
            losses['fine_detail_color'] = torch.tensor(0.0, device=pred.device)

        if self.base_improvement_weight > 0 and base is not None:
            pred_err = (pred - target).pow(2).mean(dim=1, keepdim=True)
            base_err = (base.detach() - target).pow(2).mean(dim=1, keepdim=True)
            losses['base_improvement'] = F.relu(pred_err - base_err).mean()
        else:
            losses['base_improvement'] = torch.tensor(0.0, device=pred.device)

        # Total loss
        total = (
            self.l1_weight * losses['l1'] +
            self.perceptual_weight * losses['perceptual'] +
            self.residual_weight * losses['residual'] +
            self.lane_marking_weight * losses['lane_marking'] +
            self.edge_weight * losses['edge'] +
            self.chroma_weight * losses['chroma'] +
            self.colorful_weight * losses['colorful'] +
            self.edge_chroma_weight * losses['edge_chroma'] +
            self.fine_detail_color_weight * losses['fine_detail_color'] +
            self.base_improvement_weight * losses['base_improvement']
        )
        losses['total'] = total

        return losses


if __name__ == "__main__":
    # Test losses
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    delta = torch.randn(2, 3, 256, 256) * 0.1

    # Test L1
    l1_loss = L1Loss()
    print(f"L1 loss: {l1_loss(pred, target).item():.4f}")

    # Test Residual
    res_loss = ResidualRegularization()
    print(f"Residual loss: {res_loss(delta).item():.4f}")

    # Test Total Loss
    total_loss = TotalLoss(l1_weight=1.0, perceptual_weight=0.5, residual_weight=0.1)
    losses = total_loss(pred, target, delta)
    print(f"\nTotal loss components:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
