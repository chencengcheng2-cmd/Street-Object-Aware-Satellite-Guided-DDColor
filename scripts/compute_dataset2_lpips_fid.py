"""Compute LPIPS and FID for the 500 exported dataset2 comparisons."""

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

from inference import default_patch_indices, prepare_panorama
from scripts.export_dataset2_comparisons import find_by_id, find_pairs, load_rgb, merge_patches, resize_rgb
from src.model import SatelliteGuidedDDColor
from src.utils import load_config, load_matching_state_dict


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.clip(image, 0, 1).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)


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

    pairs = find_pairs(Path(args.street_dir), Path(args.polar_dir), Path(args.gray_dir))
    satellite_files = find_by_id(Path(args.satellite_dir))
    pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("No matched samples found.")

    print(f"Matched samples: {len(pairs)}", flush=True)
    print(f"Device: {device}", flush=True)

    color_model = SatelliteGuidedDDColor(
        ddcolor_weights_path=config["ddcolor"]["weights_path"],
        ddcolor_code_path=config["ddcolor"].get("code_path"),
        context_dim=config["model"]["context_dim"],
        polar_encoder_pretrained=config["model"]["polar_encoder_pretrained"],
        satellite_encoder_pretrained=config["model"].get("satellite_encoder_pretrained", True),
        correction_type=config["model"]["correction_type"],
        residual_scale=config["model"]["residual_scale"],
        use_polar_context=config["model"].get("use_polar_context", True),
        use_lane_vit=config["model"].get("use_lane_vit", False),
        lane_vit_embed_dim=config["model"].get("lane_vit_embed_dim", 192),
        lane_vit_depth=config["model"].get("lane_vit_depth", 4),
        lane_vit_heads=config["model"].get("lane_vit_heads", 3),
        lane_vit_patch_size=config["model"].get("lane_vit_patch_size", 16),
        lane_feature_dim=config["model"].get("lane_feature_dim", 64),
        use_satellite_vit=config["model"].get("use_satellite_vit", False),
        satellite_vit_embed_dim=config["model"].get("satellite_vit_embed_dim", 192),
        satellite_vit_depth=config["model"].get("satellite_vit_depth", 4),
        satellite_vit_heads=config["model"].get("satellite_vit_heads", 3),
        satellite_vit_patch_size=config["model"].get("satellite_vit_patch_size", 16),
        satellite_vit_feature_dim=config["model"].get("satellite_vit_feature_dim", 64),
        use_cross_view_vit=config["model"].get("use_cross_view_vit", False),
        cross_view_embed_dim=config["model"].get("cross_view_embed_dim", 192),
        cross_view_depth=config["model"].get("cross_view_depth", 3),
        cross_view_heads=config["model"].get("cross_view_heads", 3),
        cross_view_patch_size=config["model"].get("cross_view_patch_size", 16),
        cross_view_feature_dim=config["model"].get("cross_view_feature_dim", 64),
        device=str(device),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_matching_state_dict(color_model, checkpoint["model_state_dict"])
    color_model.eval()

    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    fid_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    fid_model.fc = nn.Identity()
    fid_model.eval().to(device)

    polar_size = tuple(config["model"]["polar_input_size"])
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

        satellite_path = satellite_files.get(pair["id"])
        if satellite_path is None:
            raise RuntimeError(f"Missing satellite image for sample {pair['id']}")
        satellite_rgb = resize_rgb(load_rgb(satellite_path), size=(256, 256))
        satellite = torch.from_numpy(satellite_rgb).permute(2, 0, 1).float().div(255.0)
        satellite = satellite.unsqueeze(0).repeat(4, 1, 1, 1).to(device)

        output = color_model(gray_patches.to(device), polar, satellite, default_patch_indices(device))
        base = to_tensor(merge_patches(output["base_rgb"]), device)
        final = to_tensor(merge_patches(output["final_rgb"]), device)
        gt = to_tensor(resize_rgb(load_rgb(pair["street"])).astype(np.float32) / 255.0, device)

        base_lpips = lpips_model(base * 2 - 1, gt * 2 - 1).mean().item()
        final_lpips = lpips_model(final * 2 - 1, gt * 2 - 1).mean().item()
        lpips_base_values.append(base_lpips)
        lpips_final_values.append(final_lpips)

        base_features.append(inception_features(fid_model, base))
        final_features.append(inception_features(fid_model, final))
        gt_features.append(inception_features(fid_model, gt))

        if index == 1 or index % args.log_interval == 0 or index == len(pairs):
            print(f"[{index}/{len(pairs)}] LPIPS base={base_lpips:.4f}, final={final_lpips:.4f}", flush=True)

    base_features_np = np.concatenate(base_features, axis=0)
    final_features_np = np.concatenate(final_features, axis=0)
    gt_features_np = np.concatenate(gt_features, axis=0)

    result = {
        "num_samples": len(pairs),
        "base_lpips": float(np.mean(lpips_base_values)),
        "final_lpips": float(np.mean(lpips_final_values)),
        "lpips_reduction": float(np.mean(lpips_base_values) - np.mean(lpips_final_values)),
        "lpips_reduction_percent": float(
            (np.mean(lpips_base_values) - np.mean(lpips_final_values)) / np.mean(lpips_base_values) * 100.0
        ),
        "base_fid": fid_from_features(base_features_np, gt_features_np),
        "final_fid": fid_from_features(final_features_np, gt_features_np),
    }
    result["fid_reduction"] = result["base_fid"] - result["final_fid"]
    result["fid_reduction_percent"] = result["fid_reduction"] / result["base_fid"] * 100.0

    out_path = output_dir / "lpips_fid_summary.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved: {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LPIPS and FID for dataset2 comparisons.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/dual_context_v1/best.pth")
    parser.add_argument("--street_dir", default=r"C:\Users\31133\Desktop\polar and bing\streetview")
    parser.add_argument("--gray_dir", default=r"C:\Users\31133\Desktop\polar and bing\black and white")
    parser.add_argument("--polar_dir", default=r"C:\Users\31133\Desktop\dataset2\Dataset_CVUSA\polarmap\normal")
    parser.add_argument("--satellite_dir", default=r"C:\Users\31133\Desktop\dataset2\Dataset_CVUSA\satellite")
    parser.add_argument("--output_dir", default=r"C:\Users\31133\Desktop\鏁版嵁瀵规瘮鍥?2")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())


