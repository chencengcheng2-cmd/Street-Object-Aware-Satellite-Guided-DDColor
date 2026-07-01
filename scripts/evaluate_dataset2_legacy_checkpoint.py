"""Evaluate the first single-context checkpoint on the same 500 dataset2 samples."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import Inception_V3_Weights, inception_v3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference import prepare_panorama
from scripts.export_dataset2_comparisons import find_pairs, load_rgb, merge_patches, resize_rgb
from src.correction_module import LightCorrectionModule, ResidualCorrectionModule
from src.ddcolor_wrapper import DDColorWrapper
from src.metrics import PSNR, SSIM
from src.polar_encoder import PolarContextEncoder
from src.utils import load_config, load_matching_state_dict


class LegacyPolarGuidedDDColor(nn.Module):
    """Original model: frozen DDColor + Polar encoder + residual correction."""

    def __init__(
        self,
        ddcolor_weights_path: str,
        ddcolor_code_path: str | None,
        context_dim: int,
        polar_encoder_pretrained: bool,
        correction_type: str,
        residual_scale: float,
        device: str,
    ):
        super().__init__()
        self.device = device
        self.ddcolor = DDColorWrapper(
            model_path=ddcolor_weights_path,
            model_code_path=ddcolor_code_path,
            input_size=256,
            device=device,
        )
        self.polar_encoder = PolarContextEncoder(
            context_dim=context_dim,
            pretrained=polar_encoder_pretrained,
            freeze_backbone=False,
        )
        if correction_type == "light":
            self.correction = LightCorrectionModule(
                context_dim=context_dim,
                residual_scale=residual_scale,
            )
        else:
            self.correction = ResidualCorrectionModule(
                context_dim=context_dim,
                residual_scale=residual_scale,
                use_film=True,
            )
        self.to(device)

    def forward(self, gray_rgb: torch.Tensor, polar_img: torch.Tensor) -> dict:
        with torch.no_grad():
            base_rgb = self.ddcolor.colorize(gray_rgb)
        context_vector = self.polar_encoder(polar_img)
        result = self.correction(base_rgb, context_vector)
        return {
            "base_rgb": base_rgb,
            "context_vector": context_vector,
            "final_rgb": result["final_rgb"],
            "delta_color": result["delta_color"],
        }


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.clip(image, 0, 1).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)


def simple_metrics(pred: np.ndarray, target: np.ndarray, psnr_metric: PSNR, ssim_metric: SSIM, device: torch.device) -> dict:
    pred_t = to_tensor(pred, device)
    target_t = to_tensor(target, device)
    diff = pred - target
    return {
        "psnr": float(psnr_metric(pred_t, target_t)),
        "ssim": float(ssim_metric(pred_t, target_t)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def fid_from_features(features_a: np.ndarray, features_b: np.ndarray) -> float:
    mu_a = np.mean(features_a, axis=0)
    mu_b = np.mean(features_b, axis=0)
    sigma_a = np.cov(features_a, rowvar=False)
    sigma_b = np.cov(features_b, rowvar=False)
    covmean = linalg.sqrtm(sigma_a.dot(sigma_b))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    return float(diff.dot(diff) + np.trace(sigma_a + sigma_b - 2 * covmean))


@torch.inference_mode()
def inception_features(model: nn.Module, image: torch.Tensor) -> np.ndarray:
    image = F.interpolate(image, size=(299, 299), mode="bilinear", align_corners=False)
    return model(image).detach().cpu().numpy()


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    import lpips

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(Path(args.street_dir), Path(args.polar_dir), Path(args.gray_dir))[: args.limit]
    if not pairs:
        raise RuntimeError("No matched street/gray/polar samples were found.")
    print(f"Matched samples: {len(pairs)}", flush=True)
    print(f"Device: {device}", flush=True)

    model = LegacyPolarGuidedDDColor(
        ddcolor_weights_path=config["ddcolor"]["weights_path"],
        ddcolor_code_path=config["ddcolor"].get("code_path"),
        context_dim=config["model"]["context_dim"],
        polar_encoder_pretrained=config["model"]["polar_encoder_pretrained"],
        correction_type=config["model"]["correction_type"],
        residual_scale=config["model"]["residual_scale"],
        device=str(device),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_matching_state_dict(model, checkpoint["model_state_dict"])
    model.eval()

    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    fid_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    fid_model.fc = nn.Identity()
    fid_model.eval().to(device)

    polar_size = tuple(config["model"]["polar_input_size"])
    rows = []
    lpips_base_values = []
    lpips_final_values = []
    base_features = []
    final_features = []
    gt_features = []

    for index, pair in enumerate(pairs, start=1):
        gray_rgb = load_rgb(pair["gray"])
        _, gray_patches = prepare_panorama(gray_rgb)

        polar_rgb = load_rgb(pair["polar"])
        polar_rgb = cv2.resize(polar_rgb, (polar_size[1], polar_size[0]), interpolation=cv2.INTER_AREA)
        polar = torch.from_numpy(polar_rgb).permute(2, 0, 1).float().div(255.0)
        polar = polar.unsqueeze(0).repeat(4, 1, 1, 1).to(device)

        output = model(gray_patches.to(device), polar)
        base = merge_patches(output["base_rgb"])
        final = merge_patches(output["final_rgb"])
        gt = resize_rgb(load_rgb(pair["street"])).astype(np.float32) / 255.0

        base_metrics = simple_metrics(base, gt, psnr_metric, ssim_metric, device)
        final_metrics = simple_metrics(final, gt, psnr_metric, ssim_metric, device)

        base_t = to_tensor(base, device)
        final_t = to_tensor(final, device)
        gt_t = to_tensor(gt, device)
        base_lpips = lpips_model(base_t * 2 - 1, gt_t * 2 - 1).mean().item()
        final_lpips = lpips_model(final_t * 2 - 1, gt_t * 2 - 1).mean().item()
        lpips_base_values.append(base_lpips)
        lpips_final_values.append(final_lpips)
        base_features.append(inception_features(fid_model, base_t))
        final_features.append(inception_features(fid_model, final_t))
        gt_features.append(inception_features(fid_model, gt_t))

        rows.append(
            {
                "id": pair["id"],
                "base_psnr": base_metrics["psnr"],
                "final_psnr": final_metrics["psnr"],
                "base_ssim": base_metrics["ssim"],
                "final_ssim": final_metrics["ssim"],
                "base_mae": base_metrics["mae"],
                "final_mae": final_metrics["mae"],
                "base_rmse": base_metrics["rmse"],
                "final_rmse": final_metrics["rmse"],
                "base_lpips": base_lpips,
                "final_lpips": final_lpips,
            }
        )

        if index == 1 or index % args.log_interval == 0 or index == len(pairs):
            print(f"[{index}/{len(pairs)}] final PSNR={final_metrics['psnr']:.4f}, LPIPS={final_lpips:.4f}", flush=True)

    summary = {"num_samples": len(rows)}
    for key in rows[0]:
        if key == "id":
            continue
        summary[key] = float(np.mean([row[key] for row in rows]))
    summary["psnr_improvement_over_base"] = summary["final_psnr"] - summary["base_psnr"]
    summary["ssim_improvement_over_base"] = summary["final_ssim"] - summary["base_ssim"]
    summary["mae_reduction_over_base"] = summary["base_mae"] - summary["final_mae"]
    summary["rmse_reduction_over_base"] = summary["base_rmse"] - summary["final_rmse"]
    summary["lpips_reduction_over_base"] = summary["base_lpips"] - summary["final_lpips"]
    summary["base_fid"] = fid_from_features(np.concatenate(base_features, axis=0), np.concatenate(gt_features, axis=0))
    summary["final_fid"] = fid_from_features(np.concatenate(final_features, axis=0), np.concatenate(gt_features, axis=0))
    summary["fid_reduction_over_base"] = summary["base_fid"] - summary["final_fid"]

    out_path = output_dir / args.output_name
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved: {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate legacy single-context checkpoint on dataset2.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/film_ddcolor_cu130_20260527/best.pth")
    parser.add_argument("--street_dir", default=r"C:\Users\31133\Desktop\polar and bing\streetview")
    parser.add_argument("--gray_dir", default=r"C:\Users\31133\Desktop\polar and bing\black and white")
    parser.add_argument("--polar_dir", default=r"C:\Users\31133\Desktop\dataset2\Dataset_CVUSA\polarmap\normal")
    parser.add_argument("--output_dir", default=r"C:\Users\31133\Desktop\数据对比图02")
    parser.add_argument("--output_name", default="first_model_500_metrics_summary.json")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
