"""
Convert CVUSA-style Bing satellite images to Polar images.

This script matches the observed pairs in:
    C:/Users/31133/Desktop/polar and bing/bingmap
    C:/Users/31133/Desktop/polar and bing/normal

Observed target format:
    input satellite: 370 x 370
    output polar:    512 x 128

The output uses horizontal angle and vertical radius:
    - left/right is a circular angular seam
    - top row is the outer radius
    - bottom row is the satellite image center
    - angle starts from the downward direction in image coordinates
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    if image.ndim == 2:
        return image

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


def satellite_to_polar(
    image: np.ndarray,
    output_width: int = 512,
    output_height: int = 128,
    center_x: float | None = None,
    center_y: float | None = None,
    radius: float | None = None,
    start_angle_degrees: float = 90.0,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Convert a centered satellite image to the CVUSA-style polar map."""
    height, width = image.shape[:2]

    if center_x is None:
        center_x = width / 2.0
    if center_y is None:
        center_y = height / 2.0
    if radius is None:
        radius = min(width, height) / 2.0 - 1.0

    columns = np.arange(output_width, dtype=np.float32)
    rows = np.arange(output_height, dtype=np.float32)

    theta = columns / float(output_width) * (2.0 * math.pi)
    theta = theta + math.radians(start_angle_degrees)

    # Top row = farthest satellite context, bottom row = camera center.
    radial = radius * (output_height - 1 - rows) / float(output_height - 1)

    theta_grid, radial_grid = np.meshgrid(theta, radial)
    map_x = center_x + radial_grid * np.cos(theta_grid)
    map_y = center_y + radial_grid * np.sin(theta_grid)

    polar = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return polar


def iter_images(input_path: Path, recursive: bool) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield input_path
        return

    pattern = "**/*" if recursive else "*"
    for path in sorted(input_path.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def resolve_output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:
    if output_root.suffix.lower() in IMAGE_EXTENSIONS and input_root.is_file():
        return output_root

    if input_root.is_file():
        return output_root / input_file.name

    relative = input_file.relative_to(input_root)
    return output_root / relative


def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def make_comparison(source: np.ndarray, generated: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    source_resized = cv2.resize(source, (generated.shape[1], generated.shape[0]), interpolation=cv2.INTER_AREA)
    panels = [source_resized, generated]
    if reference is not None:
        if reference.shape[:2] != generated.shape[:2]:
            reference = cv2.resize(reference, (generated.shape[1], generated.shape[0]), interpolation=cv2.INTER_LINEAR)
        diff = cv2.absdiff(generated, reference)
        panels.extend([reference, diff])
    return np.concatenate(panels, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert satellite overhead images to CVUSA-style polar images."
    )
    parser.add_argument("--input", required=True, help="Input satellite image file or folder.")
    parser.add_argument("--output", required=True, help="Output polar image file or folder.")
    parser.add_argument("--width", type=int, default=512, help="Polar output width. Default: 512.")
    parser.add_argument("--height", type=int, default=128, help="Polar output height. Default: 128.")
    parser.add_argument("--center-x", type=float, default=None, help="Polar center x. Default: image width / 2.")
    parser.add_argument("--center-y", type=float, default=None, help="Polar center y. Default: image height / 2.")
    parser.add_argument("--radius", type=float, default=None, help="Max radius. Default: min(width, height)/2 - 1.")
    parser.add_argument(
        "--start-angle",
        type=float,
        default=90.0,
        help="Start angle in degrees. Default 90 matches the provided normal polar images.",
    )
    parser.add_argument("--recursive", action="store_true", help="Process subfolders recursively.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images.")
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Optional folder of target polar images with matching file names; writes PSNR report.",
    )
    parser.add_argument(
        "--comparison-dir",
        default=None,
        help="Optional folder for visual comparisons: source/generated/reference/diff.",
    )
    parser.add_argument("--report", default=None, help="Optional CSV report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    reference_dir = Path(args.reference_dir) if args.reference_dir else None
    comparison_dir = Path(args.comparison_dir) if args.comparison_dir else None
    report_path = Path(args.report) if args.report else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    input_root = input_path
    if input_path.is_file() and output_path.suffix.lower() in IMAGE_EXTENSIONS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    if comparison_dir is not None:
        comparison_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float]] = []
    files = list(iter_images(input_path, args.recursive))
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        raise RuntimeError(f"No image files found in: {input_path}")

    for index, file_path in enumerate(files, start=1):
        source = read_image(file_path)
        polar = satellite_to_polar(
            source,
            output_width=args.width,
            output_height=args.height,
            center_x=args.center_x,
            center_y=args.center_y,
            radius=args.radius,
            start_angle_degrees=args.start_angle,
        )

        dst = resolve_output_path(file_path, input_root, output_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), polar)

        row: dict[str, str | float] = {
            "input": str(file_path),
            "output": str(dst),
            "width": args.width,
            "height": args.height,
        }

        reference = None
        if reference_dir is not None:
            reference_path = reference_dir / file_path.name
            if reference_path.exists():
                reference = read_image(reference_path)
                score = psnr_uint8(polar, reference)
                row["reference"] = str(reference_path)
                row["psnr"] = score
            else:
                row["reference"] = ""
                row["psnr"] = ""

        if comparison_dir is not None:
            comparison = make_comparison(source, polar, reference)
            comparison_path = comparison_dir / file_path.name
            cv2.imwrite(str(comparison_path), comparison)
            row["comparison"] = str(comparison_path)

        rows.append(row)
        if index == 1 or index % 100 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] {file_path.name} -> {dst}")

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Report saved: {report_path}")

    psnr_values = [float(row["psnr"]) for row in rows if row.get("psnr") not in (None, "")]
    if psnr_values:
        print(f"Reference PSNR mean: {np.mean(psnr_values):.4f} dB")

    print(f"Finished. Converted {len(files)} image(s).")


if __name__ == "__main__":
    main()
