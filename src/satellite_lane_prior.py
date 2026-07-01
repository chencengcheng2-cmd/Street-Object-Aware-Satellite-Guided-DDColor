"""Rule-based satellite lane-color prior.

This module extracts weak yellow/white road-marking evidence from satellite RGB
inside the predicted road mask. It is intentionally conservative: if the
satellite view does not provide color evidence, the prior should stay near zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SatelliteLanePrior:
    yellow_score: float
    white_score: float
    yellow_mask: np.ndarray
    white_mask: np.ndarray
    road_mask: np.ndarray
    visualization: np.ndarray


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _filter_line_components(mask: np.ndarray, min_area: int = 3) -> np.ndarray:
    """Keep small elongated components and remove large blobs/noise."""
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    h, w = mask.shape[:2]
    max_area = max(20, int(0.015 * h * w))

    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        if area < min_area or area > max_area:
            continue
        short = max(1, min(bw, bh))
        long = max(bw, bh)
        elongation = long / short
        compactness = area / max(1, bw * bh)
        if elongation >= 2.2 or (long >= 8 and compactness <= 0.55):
            filtered[labels == idx] = 1

    return filtered


def _score_line_mask(mask: np.ndarray, road_mask: np.ndarray) -> float:
    road_area = float(max(1, int(road_mask.sum())))
    line_area = float(mask.sum())
    if line_area <= 0:
        return 0.0
    ratio = line_area / road_area
    # Keep this conservative. A high score should mean visible chromatic evidence,
    # not merely a few noisy pixels in the road mask.
    return float(np.clip(ratio * 10.0, 0.0, 1.0))


def extract_satellite_lane_prior(
    satellite_rgb: np.ndarray,
    satellite_labels: np.ndarray,
    road_label: int = 1,
) -> SatelliteLanePrior:
    """Extract yellow/white lane evidence from satellite RGB and road labels.

    Args:
        satellite_rgb: RGB satellite image, usually 256x256.
        satellite_labels: NEOS-compatible semantic label map. Road is label 1.
        road_label: Label id for road/impervious surfaces.
    """
    rgb = _ensure_uint8_rgb(satellite_rgb)
    labels = np.asarray(satellite_labels)
    if labels.shape[:2] != rgb.shape[:2]:
        labels = cv2.resize(labels.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    road_mask = (labels == road_label).astype(np.uint8)
    if road_mask.sum() < 20:
        empty = np.zeros(rgb.shape[:2], dtype=np.uint8)
        return SatelliteLanePrior(0.0, 0.0, empty, empty, road_mask, rgb.copy())

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Yellow evidence: high red/green, lower blue, enough saturation and brightness.
    yellow = (
        (road_mask > 0)
        & (h >= 16)
        & (h <= 38)
        & (s >= 60)
        & (v >= 95)
        & (r.astype(np.int16) > b.astype(np.int16) + 25)
        & (g.astype(np.int16) > b.astype(np.int16) + 18)
    )

    # White evidence: bright, low saturation road markings.
    white = (
        (road_mask > 0)
        & (s <= 35)
        & (v >= 175)
        & (np.maximum.reduce([r, g, b]).astype(np.int16) - np.minimum.reduce([r, g, b]).astype(np.int16) <= 32)
    )

    # Prefer elongated thin evidence. A light opening keeps continuous lane-like structures.
    kernel = np.ones((2, 2), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    yellow_mask = _filter_line_components(yellow_mask)
    white_mask = _filter_line_components(white_mask)

    yellow_score = _score_line_mask(yellow_mask, road_mask)
    white_score = _score_line_mask(white_mask, road_mask)

    vis = rgb.copy()
    road_overlay = np.zeros_like(vis)
    road_overlay[:, :, 2] = 120
    vis = np.where(road_mask[:, :, None] > 0, (0.85 * vis + 0.15 * road_overlay).astype(np.uint8), vis)
    vis[yellow_mask > 0] = np.array([255, 220, 0], dtype=np.uint8)
    vis[white_mask > 0] = np.array([255, 255, 255], dtype=np.uint8)

    return SatelliteLanePrior(
        yellow_score=yellow_score,
        white_score=white_score,
        yellow_mask=yellow_mask.astype(np.uint8),
        white_mask=white_mask.astype(np.uint8),
        road_mask=road_mask.astype(np.uint8),
        visualization=vis,
    )


def street_lane_candidate_mask(
    gray_rgb: np.ndarray,
    street_labels: np.ndarray | None,
    road_label: int = 1,
) -> np.ndarray:
    """Find thin bright/edge-like candidates in the street road region."""
    rgb = _ensure_uint8_rgb(gray_rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    h, w = gray.shape[:2]

    if street_labels is not None:
        labels = np.asarray(street_labels)
        if labels.shape[:2] != (h, w):
            labels = cv2.resize(labels.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        road_mask = labels == road_label
    else:
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        road_mask = y > 0.45

    lower_half = np.zeros((h, w), dtype=bool)
    lower_half[int(h * 0.35):, :] = True

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 130)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1) > 0
    bright = gray > max(120, int(np.percentile(gray[road_mask], 68)) if road_mask.any() else 130)
    candidate = road_mask & lower_half & edges & bright
    candidate = _filter_line_components(candidate.astype(np.uint8), min_area=4)
    candidate = cv2.dilate(candidate, np.ones((2, 2), np.uint8), iterations=1)
    return candidate.astype(np.float32)


def apply_satellite_lane_prior(
    final_rgb_float: np.ndarray,
    gray_rgb: np.ndarray,
    street_labels: np.ndarray | None,
    lane_prior: SatelliteLanePrior | None,
    score_threshold: float = 0.25,
    max_strength: float = 0.26,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Conservatively enhance street lane colors using satellite-visible evidence."""
    if lane_prior is None:
        return final_rgb_float, None

    yellow_score = lane_prior.yellow_score
    white_score = lane_prior.white_score
    if yellow_score < score_threshold and white_score < score_threshold:
        return final_rgb_float, None

    candidate = street_lane_candidate_mask(gray_rgb, street_labels)
    if candidate.max() <= 0:
        return final_rgb_float, None

    result = np.clip(final_rgb_float.astype(np.float32), 0.0, 1.0).copy()
    candidate_3 = candidate[:, :, None]
    yellow_target = np.array([0.95, 0.82, 0.22], dtype=np.float32)
    white_target = np.array([0.92, 0.92, 0.86], dtype=np.float32)

    if yellow_score >= score_threshold:
        strength = candidate_3 * min(max_strength, max_strength * yellow_score)
        result = result * (1.0 - strength) + yellow_target * strength
    if white_score >= score_threshold and white_score > yellow_score * 0.75:
        strength = candidate_3 * min(max_strength * 0.75, max_strength * 0.75 * white_score)
        result = result * (1.0 - strength) + white_target * strength

    lane_vis = np.zeros((*candidate.shape, 3), dtype=np.uint8)
    lane_vis[candidate > 0] = np.array([255, 220, 0], dtype=np.uint8) if yellow_score >= white_score else np.array([255, 255, 255], dtype=np.uint8)
    return np.clip(result, 0.0, 1.0), lane_vis
