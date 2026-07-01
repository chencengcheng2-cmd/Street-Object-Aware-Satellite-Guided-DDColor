"""
Evaluation metrics: PSNR, SSIM, LPIPS, FID.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PSNR:
    """Peak Signal-to-Noise Ratio."""

    def __init__(self, max_val: float = 1.0):
        self.max_val = max_val

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute PSNR.

        Args:
            pred: Predicted image (B, C, H, W) in range [0, max_val]
            target: Ground truth image (B, C, H, W) in range [0, max_val]

        Returns:
            PSNR value in dB
        """
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(self.max_val / torch.sqrt(mse)).item()


class SSIM:
    """Structural Similarity Index."""

    def __init__(
        self,
        window_size: int = 11,
        max_val: float = 1.0,
        channels: int = 3,
    ):
        self.window_size = window_size
        self.max_val = max_val
        self.channels = channels
        self.window = self._create_window(window_size, channels)

    def _create_window(self, window_size: int, channels: int) -> torch.Tensor:
        """Create Gaussian window."""
        sigma = 1.5 * window_size / 11
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        window = gauss / gauss.sum()
        window = window.unsqueeze(1).mm(window.unsqueeze(1).t()).float()
        # Shape: (window_size, window_size) -> (window_size, window_size, 1, 1) -> (channels, 1, window_size, window_size)
        window = window.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()
        return window

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute SSIM."""
        window = self.window.to(pred.device)

        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=self.channels)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=self.channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=self.channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()


class LPIPS:
    """Learned Perceptual Image Patch Similarity."""

    def __init__(self, use_gpu: bool = True):
        try:
            import lpips
            self.model = lpips.LPIPS(net='vgg')
            if use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
        except ImportError:
            print("Warning: lpips not installed. LPIPS will return None.")
            self.model = None

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute LPIPS."""
        if self.model is None:
            return None

        # LPIPS expects inputs in range [-1, 1]
        pred_lpips = pred * 2 - 1
        target_lpips = target * 2 - 1

        with torch.no_grad():
            score = self.model(pred_lpips, target_lpips)

        return score.mean().item()


class FID:
    """Frechet Inception Distance."""

    def __init__(self, device: str = 'auto'):
        try:
            from torchvision.models import inception_v3
            self.model = inception_v3(pretrained=True, transform_input=False)
            self.model.fc = nn.Identity()
            self.model.eval()

            if device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device

            self.model = self.model.to(self.device)

            self.features_pred = []
            self.features_target = []
        except ImportError:
            print("Warning: Could not load Inception model. FID will return None.")
            self.model = None

    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract Inception features."""
        if self.model is None:
            return None

        # Resize to 299x299 for Inception
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        with torch.no_grad():
            features = self.model(images)

        return features.cpu().numpy()

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update feature caches."""
        if self.model is None:
            return

        feat_pred = self.extract_features(pred)
        feat_target = self.extract_features(target)

        self.features_pred.append(feat_pred)
        self.features_target.append(feat_target)

    def compute(self) -> float:
        """Compute FID from accumulated features."""
        if self.model is None or not self.features_pred:
            return None

        pred_all = np.concatenate(self.features_pred, axis=0)
        target_all = np.concatenate(self.features_target, axis=0)

        # Compute mean and covariance
        mu_pred = np.mean(pred_all, axis=0)
        mu_target = np.mean(target_all, axis=0)

        sigma_pred = np.cov(pred_all, rowvar=False)
        sigma_target = np.cov(target_all, rowvar=False)

        # Compute FID
        from scipy import linalg

        diff = mu_pred - mu_target
        covmean = linalg.sqrtm(sigma_pred.dot(sigma_target))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma_pred + sigma_target - 2 * covmean)

        return float(fid)

    def reset(self):
        """Reset feature caches."""
        self.features_pred = []
        self.features_target = []


class MetricsCalculator:
    """Calculate all metrics."""

    def __init__(self, device: str = 'auto', enable_lpips: bool = True, enable_fid: bool = True):
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.lpips = LPIPS(use_gpu=(device != 'cpu')) if enable_lpips else None
        self.fid = FID(device=device) if enable_fid else None

    def compute_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        accumulate_fid: bool = True,
    ) -> dict:
        """Compute metrics for a batch."""
        metrics = {}

        # PSNR
        psnr_val = self.psnr(pred, target)
        metrics['psnr'] = psnr_val

        # SSIM
        ssim_val = self.ssim(pred, target)
        metrics['ssim'] = ssim_val

        # LPIPS
        lpips_val = self.lpips(pred, target) if self.lpips else None
        metrics['lpips'] = lpips_val

        # FID (accumulated)
        if accumulate_fid and self.fid:
            self.fid.update(pred, target)

        return metrics

    def get_fid(self, reset: bool = True) -> float:
        """Get computed FID and optionally reset."""
        fid_val = self.fid.compute() if self.fid else None
        if reset and self.fid:
            self.fid.reset()
        return fid_val


if __name__ == "__main__":
    # Test metrics
    pred = torch.rand(4, 3, 256, 256)
    target = torch.rand(4, 3, 256, 256)

    calculator = MetricsCalculator()
    metrics = calculator.compute_batch(pred, target)

    print("Batch metrics:")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")
