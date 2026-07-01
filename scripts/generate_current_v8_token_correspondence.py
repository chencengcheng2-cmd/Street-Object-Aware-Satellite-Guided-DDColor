"""Generate one token correspondence visualization for the current v8 model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference import (
    build_online_semantics,
    colorize_neos_labels,
    default_patch_indices,
    load_rgb,
    prepare_panorama,
)
from scripts.export_polar_bing_v8_long_comparisons import build_model
from src.utils import load_config


def pil_font(size: int):
    for path in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_grid(image: Image.Image, step: int, color, width=1) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for x in range(0, w + 1, step):
        draw.line([(x, 0), (x, h)], fill=color, width=width)
    for y in range(0, h + 1, step):
        draw.line([(0, y), (w, y)], fill=color, width=width)
    return out


def choose_road_token(
    street_semantic: torch.Tensor,
    attention: np.ndarray,
    patch_size: int = 16,
) -> tuple[int, int, int, float]:
    labels = street_semantic.detach().cpu().numpy()
    best = None
    for patch_idx in range(labels.shape[0]):
        label = labels[patch_idx]
        for gy in range(16):
            for gx in range(16):
                block = label[gy * patch_size:(gy + 1) * patch_size, gx * patch_size:(gx + 1) * patch_size]
                road = float((block == 1).mean())
                if road < 0.25:
                    continue
                token_index = gy * 16 + gx
                no_match = float(attention[patch_idx, token_index, -1])
                max_sat_attn = float(attention[patch_idx, token_index, :-1].max())
                # Prefer road-like tokens that actually attend to satellite tokens.
                score = road * 2.0 + max_sat_attn * 4.0 - no_match * 2.0 + gy / 16.0 - abs(gx - 8) / 12.0
                if best is None or score > best[0]:
                    best = (score, patch_idx, gy, gx, road)
    if best is None:
        return 0, 12, 8, 0.0
    _, patch_idx, gy, gx, road = best
    return patch_idx, gy, gx, road


def heatmap_overlay(satellite_rgb: np.ndarray, attention: np.ndarray) -> Image.Image:
    attn = attention.astype(np.float32)
    attn = attn - attn.min()
    attn = attn / max(float(attn.max()), 1e-8)
    heat = cv2.resize(attn, (256, 256), interpolation=cv2.INTER_CUBIC)
    heat_u8 = np.uint8(np.clip(heat * 255.0, 0, 255))
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.48 * satellite_rgb + 0.52 * heat_color, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", default="0000052")
    parser.add_argument("--root", default=r"C:\Users\31133\Desktop\polar and bing")
    parser.add_argument("--config", default="configs/semantic_color_token_match_v8.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/semantic_color_token_match_v8/best.pth")
    parser.add_argument("--output", default=r"C:\Users\31133\Desktop\current_v8_token_correspondence.jpg")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    model = build_model(config, args.checkpoint, device)

    root = Path(args.root)
    gray_path = root / "black and white" / f"{args.sample_id}.jpg"
    sat_path = root / "bingmap" / f"input{args.sample_id}.png"
    if not gray_path.exists():
        gray_path = root / "streetview" / f"{args.sample_id}.jpg"
    street_rgb = load_rgb(str(gray_path))
    satellite_rgb = load_rgb(str(sat_path))

    gray_input, gray_patches = prepare_panorama(street_rgb)
    num_patches = gray_patches.shape[0]
    satellite_resized = cv2.resize(satellite_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    satellite = torch.from_numpy(satellite_resized).permute(2, 0, 1).float().div(255.0)
    satellite = satellite.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(device)
    polar = torch.zeros(num_patches, 3, 256, 512, dtype=torch.float32, device=device)
    patch_idx = default_patch_indices(device, num_patches=num_patches)

    street_semantic, satellite_semantic = build_online_semantics(
        model,
        street_rgb,
        satellite_resized,
        device,
    )

    with torch.inference_mode():
        output = model(
            gray_patches.to(device),
            polar,
            satellite,
            patch_idx,
            street_semantic=street_semantic,
            satellite_semantic=satellite_semantic,
        )

    attn = output["cross_view_attention"].detach().cpu().numpy()
    if attn is None:
        raise RuntimeError("Current model did not return cross_view_attention.")

    selected_patch, gy, gx, road_frac = choose_road_token(street_semantic, attn)
    token_index = gy * 16 + gx
    token_attention = attn[selected_patch, token_index, :-1].reshape(32, 32)
    no_match = float(attn[selected_patch, token_index, -1])
    top_indices = np.argsort(token_attention.reshape(-1))[::-1][:8]

    street_resized = cv2.resize(street_rgb, (1024, 256), interpolation=cv2.INTER_AREA) if gray_input.shape[1] == 1024 else cv2.resize(street_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    street_patch = street_resized[:, selected_patch * 256:(selected_patch + 1) * 256]
    street_patch_img = draw_grid(Image.fromarray(street_patch), 16, (255, 230, 0), 1)
    draw = ImageDraw.Draw(street_patch_img)
    draw.rectangle((gx * 16, gy * 16, (gx + 1) * 16, (gy + 1) * 16), outline=(255, 0, 0), width=4)

    sat_img = draw_grid(Image.fromarray(satellite_resized), 8, (255, 255, 255), 1)
    overlay = heatmap_overlay(satellite_resized, token_attention)
    overlay_draw = ImageDraw.Draw(overlay)
    for rank, idx in enumerate(top_indices, start=1):
        sy, sx = divmod(int(idx), 32)
        x0, y0 = sx * 8, sy * 8
        overlay_draw.rectangle((x0, y0, x0 + 8, y0 + 8), outline=(255, 255, 255), width=2)
        overlay_draw.text((x0 + 1, y0 - 13), str(rank), fill=(255, 255, 255), font=pil_font(12))

    street_sem_vis = colorize_neos_labels(street_semantic[selected_patch].detach().cpu().numpy())
    sat_sem_vis = colorize_neos_labels(satellite_semantic[0].detach().cpu().numpy())
    street_sem_img = Image.fromarray(street_sem_vis)
    sat_sem_img = Image.fromarray(sat_sem_vis)

    base_patch = output["base_rgb"][selected_patch].detach().cpu().permute(1, 2, 0).numpy()
    final_patch = output["final_rgb"][selected_patch].detach().cpu().permute(1, 2, 0).numpy()
    base_img = Image.fromarray(np.uint8(np.clip(base_patch * 255, 0, 255)))
    final_img = Image.fromarray(np.uint8(np.clip(final_patch * 255, 0, 255)))

    scale = 1.45
    panel_size = int(256 * scale)
    panels = [
        ("Selected street token", street_patch_img),
        ("Satellite attention heatmap", overlay),
        ("Street semantic mask", street_sem_img),
        ("Satellite semantic mask", sat_sem_img),
        ("DDColor base patch", base_img),
        ("v8 final patch", final_img),
    ]
    canvas_w = panel_size * 3 + 80
    canvas_h = panel_size * 2 + 245
    canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 250, 252))
    cd = ImageDraw.Draw(canvas)
    cd.text((32, 22), "Current v8 Token Correspondence: Street Token Querying Raw Satellite Tokens", font=pil_font(30), fill=(15, 23, 42))
    cd.text(
        (32, 62),
        f"sample={args.sample_id}, panorama_patch={selected_patch + 1}, street_token=(x={gx}, y={gy}), road_fraction={road_frac:.2f}, no_match_attention={no_match:.3f}",
        font=pil_font(18),
        fill=(71, 85, 105),
    )
    for i, (title, img) in enumerate(panels):
        row, col = divmod(i, 3)
        x = 32 + col * (panel_size + 24)
        y = 110 + row * (panel_size + 52)
        cd.text((x, y - 25), title, font=pil_font(18), fill=(15, 23, 42))
        canvas.paste(img.resize((panel_size, panel_size), Image.Resampling.NEAREST), (x, y))

    cd.text(
        (32, canvas_h - 78),
        "Interpretation: warm heatmap regions are satellite tokens with higher attention for the selected street token.",
        font=pil_font(18),
        fill=(51, 65, 85),
    )
    cd.text(
        (32, canvas_h - 48),
        "White boxes mark the top-8 satellite tokens. A lower no-match value means stronger satellite correspondence.",
        font=pil_font(18),
        fill=(51, 65, 85),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)
    print(out_path)


if __name__ == "__main__":
    main()
