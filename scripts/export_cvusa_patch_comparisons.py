"""Export CVUSA patch comparisons and metrics for a trained checkpoint.

Each output image contains:
Gray input | DDColor base | Improved final | RGB ground truth
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import collate_fn
from src.dataset import CVUSADataset
from src.metrics import MetricsCalculator
from src.model import SatelliteGuidedDDColor
from src.utils import load_config, load_matching_state_dict


def build_model(config: dict, device: torch.device) -> SatelliteGuidedDDColor:
    return SatelliteGuidedDDColor(
        ddcolor_weights_path=config["ddcolor"]["weights_path"],
        ddcolor_code_path=config.get("ddcolor", {}).get("code_path"),
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
        use_semantic_cross_view_vit=config["model"].get("use_semantic_cross_view_vit", False),
        use_semantic_color_token_match=config["model"].get("use_semantic_color_token_match", False),
        use_polar_token_match=config["model"].get("use_polar_token_match", False),
        use_semantic_cnn_context=config["model"].get("use_semantic_cnn_context", False),
        semantic_num_classes=config["model"].get("semantic_num_classes", 6),
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
        street_object_hidden_dim=config["model"].get("street_object_hidden_dim", 96),
        street_object_num_masks=config["model"].get("street_object_num_masks", 8),
        street_object_detail_scale=config["model"].get("street_object_detail_scale", 0.18),
        satellite_prior_strength=config["model"].get("satellite_prior_strength", 0.65),
        use_street_gray_edges=config["model"].get("use_street_gray_edges", False),
        use_street_gray_modulation=config["model"].get("use_street_gray_modulation", True),
        use_gray_satellite_token_selection=config["model"].get("use_gray_satellite_token_selection", True),
        use_satellite_chroma_token_selection=config["model"].get("use_satellite_chroma_token_selection", False),
        token_selection_patch_size=config["model"].get("token_selection_patch_size", 16),
        token_selection_dim=config["model"].get("token_selection_dim", 32),
        lane_detail_strength=config["model"].get("lane_detail_strength", 0.45),
        satellite_dependency_boost=config["model"].get("satellite_dependency_boost", 1.35),
        lane_evidence_threshold=config["model"].get("lane_evidence_threshold", 0.002),
        street_semantic_source=config["model"].get("street_semantic_source", "dino_v12"),
        satellite_semantic_source=config["model"].get("satellite_semantic_source", "neos"),        dino_model_name=config["model"].get("dino_model_name", "vit_small_patch16_224.dino"),
        dino_pretrained=config["model"].get("dino_pretrained", True),
        device=str(device),
    ).to(device)


def to_u8(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def make_comparison(gray: np.ndarray, base: np.ndarray, final: np.ndarray, gt: np.ndarray) -> Image.Image:
    panels = [
        ("Gray Input", gray),
        ("DDColor Base", base),
        ("Improved Final", final),
        ("Ground Truth", gt),
    ]
    panel_w, panel_h = panels[0][1].shape[1], panels[0][1].shape[0]
    label_h = 30
    gap = 8
    canvas = Image.new("RGB", (panel_w * 4 + gap * 5, panel_h + label_h + gap * 2), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    x = gap
    for label, image in panels:
        draw.text((x, gap), label, fill=(20, 20, 20), font=font)
        canvas.paste(Image.fromarray(image), (x, gap + label_h))
        x += panel_w + gap
    return canvas


def average_metric_list(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(clean)) if clean else None


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Export CVUSA patch comparison images.")
    parser.add_argument("--config", default="configs/semantic_neos_cnn_residual_v7.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/semantic_neos_cnn_residual_v7/best.pth")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", default=r"C:\Users\31133\Desktop\v12_500_comparison")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--with_lpips_fid", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    semantic_cfg = config.get("semantic", {})
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_street_object_aware = config["model"].get("correction_type") == "street_object_aware"
    load_semantics = bool(
        (
            config["model"].get("use_semantic_cross_view_vit", False)
            or config["model"].get("use_semantic_color_token_match", False)
            or config["model"].get("use_polar_token_match", False)
            or config["model"].get("use_semantic_cnn_context", False)
            or use_street_object_aware
        )
        and semantic_cfg.get("load_precomputed", False)
    )
    dataset = CVUSADataset(
        config["dataset"]["root"],
        split=args.split,
        load_polar=True,
        use_segmap=use_street_object_aware,
        load_semantics=load_semantics,
        street_semantic_dirname=semantic_cfg.get("street_dirname", "street_semantic"),
        satellite_semantic_dirname=semantic_cfg.get("satellite_dirname", "overhead_satellite_semantic"),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found for split: {args.split}")
    limit = min(args.limit, len(dataset))
    dataset = Subset(dataset, list(range(limit)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    print(f"Device: {device}", flush=True)
    print(f"Samples: {limit}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    model = build_model(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_matching_state_dict(model, checkpoint["model_state_dict"])
    model.eval()

    base_calc = MetricsCalculator(device=str(device), enable_lpips=args.with_lpips_fid, enable_fid=args.with_lpips_fid)
    final_calc = MetricsCalculator(device=str(device), enable_lpips=args.with_lpips_fid, enable_fid=args.with_lpips_fid)
    rows = []
    metric_values = {
        "base_psnr": [],
        "final_psnr": [],
        "base_ssim": [],
        "final_ssim": [],
        "base_lpips": [],
        "final_lpips": [],
    }
    saved = 0

    for batch in tqdm(loader, desc="Exporting"):
        gray = batch["gray"].to(device)
        rgb = batch["rgb"].to(device)
        polar = batch["polar"].to(device)
        satellite = batch["satellite"].to(device)
        patch_idx = batch["patch_idx"].to(device)
        street_semantic = batch.get("street_semantic")
        satellite_semantic = batch.get("satellite_semantic")
        if street_semantic is not None:
            street_semantic = street_semantic.to(device)
        if satellite_semantic is not None:
            satellite_semantic = satellite_semantic.to(device)

        output = model(
            gray,
            polar,
            satellite,
            patch_idx,
            street_semantic=street_semantic,
            satellite_semantic=satellite_semantic,
        )

        base_metrics = base_calc.compute_batch(output["base_rgb"], rgb, accumulate_fid=True)
        final_metrics = final_calc.compute_batch(output["final_rgb"], rgb, accumulate_fid=True)
        for key in ("psnr", "ssim", "lpips"):
            if key in base_metrics and base_metrics[key] is not None:
                metric_values[f"base_{key}"].append(base_metrics[key])
            if key in final_metrics and final_metrics[key] is not None:
                metric_values[f"final_{key}"].append(final_metrics[key])

        bsz = gray.shape[0]
        for i in range(bsz):
            sample_id = batch["file_id"][i]
            gray_u8 = to_u8(gray[i])
            base_u8 = to_u8(output["base_rgb"][i])
            final_u8 = to_u8(output["final_rgb"][i])
            gt_u8 = to_u8(rgb[i])
            out_path = output_dir / f"{saved + 1:04d}_{sample_id}_comparison.jpg"
            make_comparison(gray_u8, base_u8, final_u8, gt_u8).save(out_path, quality=args.quality)
            rows.append({"index": saved + 1, "file_id": sample_id, "output": str(out_path)})
            saved += 1

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_samples": saved,
        "base_psnr": average_metric_list(metric_values["base_psnr"]),
        "final_psnr": average_metric_list(metric_values["final_psnr"]),
        "base_ssim": average_metric_list(metric_values["base_ssim"]),
        "final_ssim": average_metric_list(metric_values["final_ssim"]),
        "base_lpips": average_metric_list(metric_values["base_lpips"]),
        "final_lpips": average_metric_list(metric_values["final_lpips"]),
    }
    summary["psnr_improvement"] = (
        summary["final_psnr"] - summary["base_psnr"]
        if summary["base_psnr"] is not None and summary["final_psnr"] is not None
        else None
    )
    summary["ssim_improvement"] = (
        summary["final_ssim"] - summary["base_ssim"]
        if summary["base_ssim"] is not None and summary["final_ssim"] is not None
        else None
    )
    summary["lpips_reduction"] = (
        summary["base_lpips"] - summary["final_lpips"]
        if summary["base_lpips"] is not None and summary["final_lpips"] is not None
        else None
    )

    if args.with_lpips_fid:
        summary["base_fid"] = base_calc.get_fid()
        summary["final_fid"] = final_calc.get_fid()
        summary["fid_reduction"] = (
            summary["base_fid"] - summary["final_fid"]
            if summary["base_fid"] is not None and summary["final_fid"] is not None
            else None
        )

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "file_id", "output"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved {saved} comparison images to {output_dir}", flush=True)


if __name__ == "__main__":
    main()



