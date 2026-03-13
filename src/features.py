from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from PIL import Image


@dataclass
class FeatureVector:
    color_hist: np.ndarray
    aspect_ratio: float
    fill_ratio: float
    area_ratio: float
    edge_density: float


def load_rgb(image_path: str, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize(size)
    return np.asarray(image, dtype=np.float32)


def extract_features(rgb_image: np.ndarray) -> FeatureVector:
    gray = (0.299 * rgb_image[:, :, 0] + 0.587 * rgb_image[:, :, 1] + 0.114 * rgb_image[:, :, 2]).astype(np.float32)
    threshold = _otsu_threshold(gray)
    fg_mask = gray < threshold

    if fg_mask.sum() < 20:
        # In case thresholding fails, fallback to central area assumption.
        h, w = gray.shape
        fg_mask = np.zeros_like(gray, dtype=bool)
        fg_mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True

    y_indices, x_indices = np.where(fg_mask)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    bbox_h = max(1, y_max - y_min + 1)
    bbox_w = max(1, x_max - x_min + 1)
    aspect_ratio = float(bbox_w / bbox_h)
    fill_ratio = float(fg_mask.sum() / (bbox_h * bbox_w))
    area_ratio = float(fg_mask.sum() / fg_mask.size)

    hsv = np.asarray(Image.fromarray(rgb_image.astype(np.uint8)).convert("HSV"), dtype=np.float32)
    hist_h = np.histogram(hsv[:, :, 0], bins=12, range=(0, 255), density=True)[0]
    hist_s = np.histogram(hsv[:, :, 1], bins=8, range=(0, 255), density=True)[0]
    hist_v = np.histogram(hsv[:, :, 2], bins=8, range=(0, 255), density=True)[0]
    color_hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)

    gx, gy = np.gradient(gray / 255.0)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    edge_density = float((grad_mag > 0.1).mean())

    return FeatureVector(
        color_hist=color_hist,
        aspect_ratio=aspect_ratio,
        fill_ratio=fill_ratio,
        area_ratio=area_ratio,
        edge_density=edge_density,
    )


def compare_feature_vectors(a: FeatureVector, b: FeatureVector) -> Dict[str, object]:
    color_similarity = _cosine_similarity(a.color_hist, b.color_hist)

    shape_gap = abs(a.aspect_ratio - b.aspect_ratio) + abs(a.fill_ratio - b.fill_ratio)
    shape_similarity = max(0.0, 1.0 - shape_gap / 1.2)

    size_similarity = max(0.0, 1.0 - abs(a.area_ratio - b.area_ratio) / 0.35)
    texture_similarity = max(0.0, 1.0 - abs(a.edge_density - b.edge_density) / 0.30)

    checks = {
        "color": color_similarity >= 0.70,
        "shape": shape_similarity >= 0.55,
        "size": size_similarity >= 0.55,
        "texture": texture_similarity >= 0.55,
    }

    score = sum(1 for v in checks.values() if v)
    verdict = score >= 3

    return {
        "checks": checks,
        "score": score,
        "verdict": verdict,
        "metrics": {
            "color": round(color_similarity, 3),
            "shape": round(shape_similarity, 3),
            "size": round(size_similarity, 3),
            "texture": round(texture_similarity, 3),
        },
    }


def compare_images(sample_path: str, query_path: str) -> Dict[str, object]:
    sample = extract_features(load_rgb(sample_path))
    query = extract_features(load_rgb(query_path))
    return compare_feature_vectors(sample, query)


def _otsu_threshold(gray: np.ndarray) -> float:
    hist, bin_edges = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)

    sum_bg = 0.0
    weight_bg = 0.0
    max_between = -1.0
    threshold = 127

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            threshold = t

    return float(threshold)


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den < 1e-8:
        return 0.0
    return float(np.dot(x, y) / den)
