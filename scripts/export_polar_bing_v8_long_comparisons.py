"""Export long panorama comparisons for the `polar and bing` folder.

Inputs are paired as:
  streetview/<id>.jpg          -> RGB ground truth
  black and white/<id>.jpg     -> grayscale street input
  bingmap/input<id>.png        -> satellite image
  normal/input<id>.png         -> polar image, kept in manifest for traceability

For v11, the evaluation path converts the satellite image to a widened Polar
RGB map, segments that Polar map online, and applies street-object-aware
residual correction.
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

from inference import colorize_panorama
from src.metrics import MetricsCalculator
from src.model import SatelliteGuidedDDColor
from src.utils import load_config, load_matching_state_dict


def read_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


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


def find_pairs(root: Path) -> list[dict]:
    street = find_by_id(root / "streetview")
    gray = find_by_id(root / "black and white")
    satellite = find_by_id(root / "bingmap")
    polar = find_by_id(root / "normal")
    ids = sorted(set(street) & set(gray) & set(satellite) & set(polar))
    return [
        {
            "id": sample_id,
            "street": street[sample_id],
            "gray": gray[sample_id],
            "satellite": satellite[sample_id],
            "polar": polar[sample_id],
        }
        for sample_id in ids
    ]


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    h, w = reference.shape[:2]
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def float_to_u8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def make_comparison(gray: np.ndarray, base: np.ndarray, final: np.ndarray, gt: np.ndarray) -> Image.Image:
    gray_u8 = gray if gray.dtype == np.uint8 else float_to_u8(gray)
    base_u8 = float_to_u8(base)
    final_u8 = float_to_u8(final)
    gt_u8 = gt if gt.dtype == np.uint8 else float_to_u8(gt)
    panels = [
        ("Gray Input", gray_u8),
        ("DDColor Base", base_u8),
        ("v11 Final", final_u8),
        ("Streetview GT", gt_u8),
    ]
    panel_w, panel_h = panels[0][1].shape[1], panels[0][1].shape[0]
    label_h = 34
    gap = 8
    canvas = Image.new("RGB", (panel_w * 4 + gap * 5, panel_h + label_h + gap * 2), "white")
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


def build_model(config: dict, checkpoint_path: str, device: torch.device) -> SatelliteGuidedDDColor:
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
        use_satellite_vit=config["model"].get("use_satellite_vit", False),
        use_cross_view_vit=config["model"].get("use_cross_view_vit", False),
        use_semantic_cross_view_vit=config["model"].get("use_semantic_cross_view_vit", False),
        use_semantic_color_token_match=config["model"].get("use_semantic_color_token_match", False),
        use_semantic_cnn_context=config["model"].get("use_semantic_cnn_context", False),
        semantic_num_classes=config["model"].get("semantic_num_classes", 7),
        cross_view_embed_dim=config["model"].get("cross_view_embed_dim", 192),
        cross_view_depth=config["model"].get("cross_view_depth", 2),
        cross_view_heads=config["model"].get("cross_view_heads", 3),
        cross_view_patch_size=config["model"].get("cross_view_patch_size", 16),
        cross_view_street_patch_size=config["model"].get(
            "cross_view_street_patch_size",
            config["model"].get("cross_view_patch_size", 16),
        ),
        cross_view_satellite_patch_size=config["model"].get("cross_view_satellite_patch_size", 8),
        cross_view_feature_dim=config["model"].get("cross_view_feature_dim", 64),
        color_token_match_weight=config["model"].get("color_token_match_weight", 3.0),
        semantic_distribution_weight=config["model"].get("semantic_distribution_weight", 2.0),
        boundary_match_weight=config["model"].get("boundary_match_weight", 2.0),
        token_delta_scale=config["model"].get("token_delta_scale", 0.35),
        token_correction_scale=config["model"].get("token_correction_scale", 0.8),
        use_polar_token_match=config["model"].get("use_polar_token_match", False),
        street_object_hidden_dim=config["model"].get("street_object_hidden_dim", 96),
        street_object_num_masks=config["model"].get("street_object_num_masks", 8),
        street_object_detail_scale=config["model"].get("street_object_detail_scale", 0.18),
        dino_model_name=config["model"].get("dino_model_name", "vit_small_patch16_224.dino"),
        dino_pretrained=config["model"].get("dino_pretrained", True),
        device=str(device),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_matching_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Export long panorama comparisons.")
    parser.add_argument("--root", default=r"C:\Users\31133\Desktop\polar and bing")
    parser.add_argument("--config", default="configs/street_object_aware_v11.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/street_object_aware_v11/best.pth")
    parser.add_argument("--output_dir", default=r"C:\Users\31133\Desktop\v11_polar_semantic_对比图500")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--disable_online_semantics", action="store_true")
    parser.add_argument("--disable_lpips_fid", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(Path(args.root))
    if not pairs:
        raise RuntimeError(f"No matched samples found under {args.root}")
    pairs = pairs[: args.limit]
    print(f"Matched samples: {len(pairs)}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Online semantics: {not args.disable_online_semantics}", flush=True)

    model = build_model(config, args.checkpoint, device)
    enable_heavy_metrics = not args.disable_lpips_fid
    base_calc = MetricsCalculator(device=str(device), enable_lpips=enable_heavy_metrics, enable_fid=enable_heavy_metrics)
    final_calc = MetricsCalculator(device=str(device), enable_lpips=enable_heavy_metrics, enable_fid=enable_heavy_metrics)

    metric_values = {key: [] for key in ["base_psnr", "final_psnr", "base_ssim", "final_ssim", "base_lpips", "final_lpips"]}
    rows = []
    manifest = []
    polar_size = tuple(config["model"].get("polar_input_size", [256, 512]))

    for index, pair in enumerate(pairs, start=1):
        gray_input = read_rgb(pair["gray"])
        satellite = read_rgb(pair["satellite"])
        gt_u8 = read_rgb(pair["street"])
        result = colorize_panorama(
            model,
            gray_input,
            satellite,
            device,
            polar_size=polar_size,
            use_online_semantics=not args.disable_online_semantics,
            token_context="satellite",
        )
        base = result["base"]
        final = result["final"]
        gray_vis = result["gray"]
        gt_resized_u8 = resize_like(gt_u8, final)
        gt = gt_resized_u8.astype(np.float32) / 255.0

        base_t = to_tensor(base, device)
        final_t = to_tensor(final, device)
        gt_t = to_tensor(gt, device)
        base_metrics = base_calc.compute_batch(base_t, gt_t, accumulate_fid=enable_heavy_metrics)
        final_metrics = final_calc.compute_batch(final_t, gt_t, accumulate_fid=enable_heavy_metrics)

        for metric in ["psnr", "ssim", "lpips"]:
            if base_metrics.get(metric) is not None:
                metric_values[f"base_{metric}"].append(float(base_metrics[metric]))
            if final_metrics.get(metric) is not None:
                metric_values[f"final_{metric}"].append(float(final_metrics[metric]))

        out_path = output_dir / f"{index:04d}_{pair['id']}_comparison.jpg"
        make_comparison(gray_vis, base, final, gt_resized_u8).save(out_path, quality=args.quality)

        rows.append(
            {
                "index": index,
                "id": pair["id"],
                "base_psnr": base_metrics["psnr"],
                "final_psnr": final_metrics["psnr"],
                "base_ssim": base_metrics["ssim"],
                "final_ssim": final_metrics["ssim"],
                "base_lpips": base_metrics.get("lpips"),
                "final_lpips": final_metrics.get("lpips"),
                "output": str(out_path),
            }
        )
        manifest.append({**{k: str(v) for k, v in pair.items()}, "output": str(out_path)})

        if index == 1 or index % args.log_interval == 0 or index == len(pairs):
            print(f"[{index}/{len(pairs)}] {pair['id']} saved", flush=True)

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "root": args.root,
        "num_samples": len(rows),
        "online_semantics": not args.disable_online_semantics,
        "model": "street_object_aware_v11",
        "token_context": "satellite_to_wide_polar_semantic",
        "polar_semantic": True,
        "context_flow": "satellite -> widened polar RGB -> polar semantic segmentation -> v11 residual correction",
    }
    for key, values in metric_values.items():
        clean = [v for v in values if np.isfinite(v)]
        summary[key] = float(np.mean(clean)) if clean else None
    summary["psnr_improvement"] = summary["final_psnr"] - summary["base_psnr"]
    summary["ssim_improvement"] = summary["final_ssim"] - summary["base_ssim"]
    if summary["base_lpips"] is not None and summary["final_lpips"] is not None:
        summary["lpips_reduction"] = summary["base_lpips"] - summary["final_lpips"]
    if enable_heavy_metrics:
        summary["base_fid"] = base_calc.get_fid()
        summary["final_fid"] = final_calc.get_fid()
        if summary["base_fid"] is not None and summary["final_fid"] is not None:
            summary["fid_reduction"] = summary["base_fid"] - summary["final_fid"]

    with (output_dir / "metrics_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved {len(rows)} comparisons to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

