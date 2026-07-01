"""Export comparison images for dataset2.

Each output image contains:
DDColor base result | satellite-guided final result | original street-view RGB.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference import default_patch_indices, prepare_panorama
from src.metrics import PSNR, SSIM
from src.model import SatelliteGuidedDDColor
from src.utils import load_config, load_matching_state_dict


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def resize_rgb(image: np.ndarray, size: tuple[int, int] = (1024, 256)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def merge_patches(tensor: torch.Tensor) -> np.ndarray:
    patches = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    return np.clip(np.concatenate(list(patches), axis=1), 0, 1)


def make_comparison(base: np.ndarray, final: np.ndarray, gt: np.ndarray) -> Image.Image:
    base_u8 = (np.clip(base, 0, 1) * 255.0).round().astype(np.uint8)
    final_u8 = (np.clip(final, 0, 1) * 255.0).round().astype(np.uint8)
    gt_u8 = (np.clip(gt, 0, 1) * 255.0).round().astype(np.uint8)

    panels = [
        ("DDColor Base", base_u8),
        ("Improved Final", final_u8),
        ("Streetview GT", gt_u8),
    ]
    panel_w, panel_h = panels[0][1].shape[1], panels[0][1].shape[0]
    label_h = 34
    gap = 8
    canvas = Image.new("RGB", (panel_w * 3 + gap * 4, panel_h + label_h + gap * 2), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    x = gap
    for label, image in panels:
        draw.text((x, gap), label, fill=(20, 20, 20), font=font)
        canvas.paste(Image.fromarray(image), (x, gap + label_h))
        x += panel_w + gap
    return canvas


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    image = np.clip(image, 0, 1).astype(np.float32)
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_simple_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    psnr_metric: PSNR,
    ssim_metric: SSIM,
    device: torch.device,
) -> dict:
    pred_t = image_to_tensor(pred, device)
    target_t = image_to_tensor(target, device)
    diff = pred - target
    return {
        "psnr": float(psnr_metric(pred_t, target_t)),
        "ssim": float(ssim_metric(pred_t, target_t)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def find_pairs(street_dir: Path, polar_dir: Path, gray_dir: Path) -> list[dict]:
    street_files = {p.stem: p for p in street_dir.iterdir() if p.is_file()}
    gray_files = {p.stem: p for p in gray_dir.iterdir() if p.is_file()}
    polar_files = {}
    for p in polar_dir.iterdir():
        if not p.is_file():
            continue
        stem = p.stem
        if stem.startswith("input"):
            stem = stem[len("input"):]
        polar_files[stem] = p

    ids = sorted(set(street_files) & set(gray_files) & set(polar_files))
    return [
        {
            "id": sample_id,
            "street": street_files[sample_id],
            "gray": gray_files[sample_id],
            "polar": polar_files[sample_id],
        }
        for sample_id in ids
    ]


def find_by_id(directory: Path) -> dict[str, Path]:
    files = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        stem = path.stem
        if stem.startswith("input"):
            stem = stem[len("input"):]
        files[stem] = path
    return files


@torch.inference_mode()
def export(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(Path(args.street_dir), Path(args.polar_dir), Path(args.gray_dir))
    satellite_files = find_by_id(Path(args.satellite_dir))
    if not pairs:
        raise RuntimeError("No matched street/gray/polar samples were found.")
    pairs = pairs[: args.limit]

    print(f"Matched samples: {len(pairs)}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    model = SatelliteGuidedDDColor(
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
    load_matching_state_dict(model, checkpoint["model_state_dict"])
    model.eval()

    polar_size = tuple(config["model"]["polar_input_size"])
    manifest = []
    metric_rows = []
    psnr_metric = PSNR()
    ssim_metric = SSIM()

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
        patch_idx = default_patch_indices(device)

        output = model(gray_patches.to(device), polar, satellite, patch_idx)
        base = merge_patches(output["base_rgb"])
        final = merge_patches(output["final_rgb"])
        gt = resize_rgb(load_rgb(pair["street"])).astype(np.float32) / 255.0

        base_metrics = compute_simple_metrics(base, gt, psnr_metric, ssim_metric, device)
        final_metrics = compute_simple_metrics(final, gt, psnr_metric, ssim_metric, device)
        metric_rows.append(
            {
                "id": pair["id"],
                "base_psnr": base_metrics["psnr"],
                "final_psnr": final_metrics["psnr"],
                "delta_psnr": final_metrics["psnr"] - base_metrics["psnr"],
                "base_ssim": base_metrics["ssim"],
                "final_ssim": final_metrics["ssim"],
                "delta_ssim": final_metrics["ssim"] - base_metrics["ssim"],
                "base_mae": base_metrics["mae"],
                "final_mae": final_metrics["mae"],
                "delta_mae": final_metrics["mae"] - base_metrics["mae"],
                "base_rmse": base_metrics["rmse"],
                "final_rmse": final_metrics["rmse"],
                "delta_rmse": final_metrics["rmse"] - base_metrics["rmse"],
            }
        )

        comparison = make_comparison(base, final, gt)
        out_path = output_dir / f"{pair['id']}_comparison.jpg"
        comparison.save(out_path, quality=args.quality)

        manifest.append(
            {
                "id": pair["id"],
                "output": str(out_path),
                "street": str(pair["street"]),
                "gray": str(pair["gray"]),
                "polar": str(pair["polar"]),
                "satellite": str(satellite_path),
            }
        )

        if index == 1 or index % args.log_interval == 0 or index == len(pairs):
            print(f"[{index}/{len(pairs)}] saved {out_path.name}", flush=True)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_csv_path = output_dir / "metrics_per_sample.csv"
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {"num_samples": len(metric_rows)}
    for key in metric_rows[0]:
        if key == "id":
            continue
        summary[key] = float(np.mean([row[key] for row in metric_rows]))
    summary["psnr_improvement"] = summary["final_psnr"] - summary["base_psnr"]
    summary["ssim_improvement"] = summary["final_ssim"] - summary["base_ssim"]
    summary["mae_reduction"] = summary["base_mae"] - summary["final_mae"]
    summary["rmse_reduction"] = summary["base_rmse"] - summary["final_rmse"]
    summary["mae_reduction_percent"] = (
        summary["mae_reduction"] / summary["base_mae"] * 100.0 if summary["base_mae"] else 0.0
    )
    summary["rmse_reduction_percent"] = (
        summary["rmse_reduction"] / summary["base_rmse"] * 100.0 if summary["base_rmse"] else 0.0
    )

    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Manifest: {manifest_path}", flush=True)
    print(f"Metrics CSV: {metrics_csv_path}", flush=True)
    print(f"Metrics summary: {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export 500 dataset2 comparison images.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/film_ddcolor_cu130_20260527/best.pth")
    parser.add_argument("--street_dir", default=r"C:\Users\31133\Desktop\dataset2\Dataset_CVUSA\streetview")
    parser.add_argument("--polar_dir", default=r"C:\Users\31133\Desktop\dataset2\Dataset_CVUSA\polarmap\normal")
    parser.add_argument("--satellite_dir", default=r"C:\Users\31133\Desktop\dataset2\Dataset_CVUSA\satellite")
    parser.add_argument("--gray_dir", default=r"C:\Users\31133\Desktop\dataset2\black and white")
    parser.add_argument("--output_dir", default=r"C:\Users\31133\Desktop\鏁版嵁闆嗗姣斿浘")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())


