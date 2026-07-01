"""
Polar transform for converting satellite overhead images to polar coordinates.

Used for creating the polar coordinate representation that provides global scene context.
"""

import cv2
import numpy as np
from typing import Tuple


def cart_to_polar(
    img: np.ndarray,
    center: Tuple[float, float] = None,
    radius: float = None,
    output_shape: Tuple[int, int] = None,
) -> np.ndarray:
    """
    Convert cartesian coordinates to polar coordinates.

    Args:
        img: Input image (H, W, C)
        center: Center point (x, y) for polar transform
        radius: Maximum radius for polar transform
        output_shape: Output shape (height, width)

    Returns:
        Polar transformed image
    """
    h, w = img.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    if radius is None:
        radius = min(w, h) / 2

    if output_shape is None:
        output_shape = (h, w)

    # Create polar coordinate grid
    max_radius = radius
    theta = np.linspace(0, 2 * np.pi, output_shape[1])
    radius_vals = np.linspace(0, max_radius, output_shape[0])

    theta_grid, radius_grid = np.meshgrid(theta, radius_vals)

    # Convert to cartesian coordinates
    x = center[0] + radius_grid * np.cos(theta_grid)
    y = center[1] + radius_grid * np.sin(theta_grid)

    # Remap using interpolation
    polar_img = cv2.remap(
        img,
        x.astype(np.float32),
        y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return polar_img


def polar_to_cart(
    img: np.ndarray,
    center: Tuple[float, float] = None,
    radius: float = None,
    output_shape: Tuple[int, int] = None,
) -> np.ndarray:
    """
    Convert polar coordinates back to cartesian coordinates.

    Args:
        img: Polar image (H, W, C)
        center: Center point (x, y) for inverse transform
        radius: Maximum radius for inverse transform
        output_shape: Output shape (height, width)

    Returns:
        Cartesian image
    """
    if output_shape is None:
        output_shape = (img.shape[0], img.shape[1])

    h, w = output_shape

    if center is None:
        center = (w / 2, h / 2)

    if radius is None:
        radius = min(w, h) / 2

    # Create cartesian coordinate grid
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Convert to polar coordinates
    dx = x - center[0]
    dy = y - center[1]
    radius_grid = np.sqrt(dx**2 + dy**2)
    theta_grid = np.arctan2(dy, dx)

    # Normalize theta to [0, 2π] and radius to [0, max_radius]
    theta_grid = (theta_grid + 2 * np.pi) % (2 * np.pi)
    theta_normalized = theta_grid / (2 * np.pi) * img.shape[1]
    radius_normalized = radius_grid / radius * img.shape[0]

    # Remap
    cart_img = cv2.remap(
        img,
        theta_normalized.astype(np.float32),
        radius_normalized.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return cart_img


def create_polar_from_satellite(
    satellite_img: np.ndarray,
    output_size: Tuple[int, int] = (256, 512),
    keep_ratio: bool = True,
) -> np.ndarray:
    """
    Create a CVUSA-style polar representation from a satellite overhead image.

    The local reference pairs in ``polar and bing`` use this convention:
    - horizontal axis: 360-degree angle
    - vertical axis: radius
    - top row: farthest radius
    - bottom row: image center
    - angle starts from the downward direction in image coordinates

    This is intentionally different from OpenCV's default warpPolar layout.

    Args:
        satellite_img: Satellite image (H, W, C)
        output_size: Target size (height, width) for polar output
        keep_ratio: Kept for backward compatibility. The CVUSA transform
            always writes exactly ``output_size``.

    Returns:
        Polar transformed image of size output_size
    """
    if len(satellite_img.shape) == 2:
        satellite_img = cv2.cvtColor(satellite_img, cv2.COLOR_GRAY2BGR)

    h, w = satellite_img.shape[:2]
    out_h, out_w = output_size

    center_x = w / 2.0
    center_y = h / 2.0
    max_radius = min(w, h) / 2.0 - 1.0

    columns = np.arange(out_w, dtype=np.float32)
    rows = np.arange(out_h, dtype=np.float32)
    theta = columns / float(out_w) * (2.0 * np.pi) + (np.pi / 2.0)
    radial = max_radius * (out_h - 1 - rows) / float(out_h - 1)

    theta_grid, radial_grid = np.meshgrid(theta, radial)
    map_x = center_x + radial_grid * np.cos(theta_grid)
    map_y = center_y + radial_grid * np.sin(theta_grid)

    polar = cv2.remap(
        satellite_img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return polar


def visualize_polar_transform(
    img: np.ndarray,
    save_path: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Visualize polar transform by showing side-by-side comparison.

    Args:
        img: Input image (H, W, C)
        save_path: Optional path to save visualization

    Returns:
        Tuple of (polar_img, visualization)
    """
    polar = create_polar_from_satellite(img)

    # Resize images to same height for visualization
    h = min(img.shape[0], polar.shape[0])
    img_resized = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    polar_resized = cv2.resize(polar, (int(polar.shape[1] * h / polar.shape[0]), h))

    # Concatenate
    vis = np.concatenate([img_resized, polar_resized], axis=1)

    if save_path:
        cv2.imwrite(save_path, vis)

    return polar, vis


if __name__ == "__main__":
    # Test polar transform
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        if img is not None:
            polar, vis = visualize_polar_transform(img, "polar_vis.jpg")
            print(f"Input shape: {img.shape}, Polar shape: {polar.shape}")
            print("Visualization saved to polar_vis.jpg")
        else:
            print(f"Failed to load image: {img_path}")
