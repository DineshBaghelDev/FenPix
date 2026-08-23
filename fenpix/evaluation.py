from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from .color import reconstruct_indexed_png
from .text import FrozenVisionLanguageEncoder, TextEncoderConfig


PROMPT_EVAL_SET = (
    "red knight sprite transparent",
    "blue potion icon transparent",
    "grass tile",
    "small wooden object transparent",
    "stone house building",
    "tiny forest scene",
    "isometric stone building",
    "transparent sword icon",
)


@dataclass(frozen=True)
class QualityMetrics:
    palette_fidelity: float
    transparency_iou: float
    boundary_f1: float
    connected_component_consistency: float
    grid_pixel_alignment: float
    text_image_alignment: float
    inference_latency_ms: float


def _to_numpy_rgba(image: Image.Image | np.ndarray) -> np.ndarray:
    return np.asarray(image.convert("RGBA") if isinstance(image, Image.Image) else image, dtype=np.uint8)


def _fit_rgba_to_shape(rgba: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if rgba.shape[:2] == shape:
        return rgba
    out = np.zeros((height, width, 4), dtype=np.uint8)
    copy_h = min(height, rgba.shape[0])
    copy_w = min(width, rgba.shape[1])
    out[:copy_h, :copy_w] = rgba[:copy_h, :copy_w]
    return out


def _alpha_mask(rgba: np.ndarray) -> np.ndarray:
    return rgba[..., 3] >= 128


def palette_fidelity(pred_rgba: np.ndarray, target_rgba: np.ndarray) -> float:
    pred_colors = np.unique(pred_rgba.reshape(-1, 4), axis=0).astype(np.float32)
    target_colors = np.unique(target_rgba.reshape(-1, 4), axis=0).astype(np.float32)
    if len(pred_colors) == 0 or len(target_colors) == 0:
        return 1.0
    distances = np.sqrt(((pred_colors[:, None] - target_colors[None]) ** 2).sum(axis=2))
    return float(np.clip(1.0 - distances.min(axis=1).mean() / 510.0, 0.0, 1.0))


def transparency_iou(pred_rgba: np.ndarray, target_rgba: np.ndarray) -> float:
    pred = _alpha_mask(pred_rgba)
    target = _alpha_mask(target_rgba)
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, target).sum() / union)


def _boundary(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(mask, dtype=bool)
    edge[1:] |= mask[1:] != mask[:-1]
    edge[:-1] |= mask[1:] != mask[:-1]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return edge


def boundary_f1(pred_rgba: np.ndarray, target_rgba: np.ndarray) -> float:
    pred = _boundary(_alpha_mask(pred_rgba))
    target = _boundary(_alpha_mask(target_rgba))
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    denom = 2 * tp + fp + fn
    return float(1.0 if denom == 0 else (2 * tp) / denom)


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if seen[y, x] or not mask[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return count


def connected_component_consistency(pred_rgba: np.ndarray, target_rgba: np.ndarray) -> float:
    pred = _component_count(_alpha_mask(pred_rgba))
    target = _component_count(_alpha_mask(target_rgba))
    return float(1.0 - abs(pred - target) / max(pred, target, 1))


def grid_pixel_alignment(pred_rgba: np.ndarray) -> float:
    alpha = pred_rgba[..., 3]
    hard_alpha = np.logical_or(alpha == 0, alpha == 255).mean()
    integer_rgba = (pred_rgba == pred_rgba.round()).mean()
    return float((hard_alpha + integer_rgba) / 2)


def text_image_alignment(prompts: list[str], images: list[Image.Image], encoder: FrozenVisionLanguageEncoder | None = None) -> float:
    encoder = encoder or FrozenVisionLanguageEncoder(TextEncoderConfig())
    scores = []
    for start in range(0, len(images), 64):
        text = encoder.encode(prompts[start : start + 64])
        image = encoder.encode_images(images[start : start + 64])
        scores.append((text * image).sum(1).cpu())
    return float(torch.cat(scores).mean().item()) if scores else 0.0


def compute_quality_metrics(
    pred_images: list[Image.Image | np.ndarray],
    target_images: list[Image.Image | np.ndarray],
    prompts: list[str],
    *,
    encoder: FrozenVisionLanguageEncoder | None = None,
    latency_ms: float = 0.0,
) -> QualityMetrics:
    pred = [_to_numpy_rgba(image) for image in pred_images]
    target = [_fit_rgba_to_shape(_to_numpy_rgba(image), p.shape[:2]) for p, image in zip(pred, target_images)]
    return QualityMetrics(
        palette_fidelity=float(np.mean([palette_fidelity(p, t) for p, t in zip(pred, target)])),
        transparency_iou=float(np.mean([transparency_iou(p, t) for p, t in zip(pred, target)])),
        boundary_f1=float(np.mean([boundary_f1(p, t) for p, t in zip(pred, target)])),
        connected_component_consistency=float(np.mean([connected_component_consistency(p, t) for p, t in zip(pred, target)])),
        grid_pixel_alignment=float(np.mean([grid_pixel_alignment(p) for p in pred])),
        text_image_alignment=text_image_alignment(prompts, [Image.fromarray(p, "RGBA") for p in pred], encoder),
        inference_latency_ms=latency_ms,
    )


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def save_comparison_gallery(
    targets: list[Image.Image | np.ndarray],
    generated: list[Image.Image | np.ndarray],
    path: str | Path,
    prompts: list[str],
    max_items: int = 8,
) -> None:
    rows = []
    font_h = 14
    for row in range(min(max_items, len(generated))):
        pred = _to_numpy_rgba(generated[row])
        target = _fit_rgba_to_shape(_to_numpy_rgba(targets[row]), pred.shape[:2])
        strip = np.concatenate([target, pred], axis=1)
        label = Image.new("RGBA", (strip.shape[1], font_h), (255, 255, 255, 255))
        ImageDraw.Draw(label).text((2, 1), prompts[row % len(prompts)][:80], fill=(0, 0, 0, 255))
        rows.append(np.concatenate([np.asarray(label), strip], axis=0))
    canvas = np.concatenate(rows, axis=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas.astype(np.uint8), "RGBA").save(path)


def render_indexed_batch(indices: torch.Tensor, palette: torch.Tensor, palette_mask: torch.Tensor, valid_mask: torch.Tensor) -> list[Image.Image]:
    images = []
    for row in range(indices.shape[0]):
        valid = valid_mask[row].detach().cpu()
        h = int(valid.any(1).sum().item())
        w = int(valid.any(0).sum().item())
        row_palette = palette[row][palette_mask[row]]
        row_indices = indices[row, :h, :w].detach().cpu().masked_fill(~valid[:h, :w], 0)
        images.append(reconstruct_indexed_png(row_indices, row_palette))
    return images


def timed(fn):
    start = perf_counter()
    value = fn()
    return value, (perf_counter() - start) * 1000
