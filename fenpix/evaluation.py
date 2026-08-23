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
from .refiner import PaletteLogitFlowRefiner, compare_refinement
from .text import FrozenPretrainedTextEncoder, TextEncoderConfig


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
    structure_accuracy: float
    index_accuracy: float
    palette_validity: float
    transparency_accuracy: float
    edge_detail_score: float
    text_image_alignment: float
    inference_latency_ms: float


def edge_detail_score(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    target_h = target[:, 1:] != target[:, :-1]
    pred_h = pred[:, 1:] != pred[:, :-1]
    valid_h = valid[:, 1:] & valid[:, :-1]
    target_w = target[:, :, 1:] != target[:, :, :-1]
    pred_w = pred[:, :, 1:] != pred[:, :, :-1]
    valid_w = valid[:, :, 1:] & valid[:, :, :-1]
    hits = pred_h.eq(target_h).logical_and(valid_h).float().sum()
    hits = hits + pred_w.eq(target_w).logical_and(valid_w).float().sum()
    total = valid_h.float().sum() + valid_w.float().sum()
    return float((hits / total.clamp_min(1)).item())


def text_image_alignment(prompts: list[str], metadata: list[dict[str, Any]], dim: int = 64) -> float:
    targets = []
    for meta, prompt in zip(metadata, prompts):
        targets.append(str(meta.get("caption") or meta.get("category") or Path(str(meta.get("path", prompt))).stem))
    encoder = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=dim))
    return float((encoder.encode(prompts) * encoder.encode(targets)).sum(1).mean().item())


def compute_quality_metrics(
    logits: torch.Tensor,
    indices: torch.Tensor,
    valid_mask: torch.Tensor,
    palette_mask: torch.Tensor,
    prompts: list[str],
    metadata: list[dict[str, Any]],
    *,
    structure_tokens: torch.Tensor | None = None,
) -> QualityMetrics:
    start = perf_counter()
    pred = logits.argmax(dim=1).masked_fill(~valid_mask, -1)
    latency_ms = (perf_counter() - start) * 1000
    total = valid_mask.float().sum().clamp_min(1)
    index_accuracy = pred.eq(indices).logical_and(valid_mask).float().sum() / total
    structure = structure_tokens if structure_tokens is not None else indices
    structure_accuracy = pred.eq(structure).logical_and(valid_mask).float().sum() / total
    palette_sizes = palette_mask.sum(1)[:, None, None].to(pred.device)
    palette_valid = (pred.ge(0) & pred.lt(palette_sizes)).logical_or(~valid_mask)
    transparent_target = indices.eq(0) & valid_mask
    transparent_pred = pred.eq(0) & valid_mask
    transparency = transparent_pred.eq(transparent_target).logical_and(valid_mask).float().sum() / total
    return QualityMetrics(
        structure_accuracy=float(structure_accuracy.item()),
        index_accuracy=float(index_accuracy.item()),
        palette_validity=float(palette_valid.float().mean().item()),
        transparency_accuracy=float(transparency.item()),
        edge_detail_score=edge_detail_score(pred, indices, valid_mask),
        text_image_alignment=text_image_alignment(prompts, metadata, logits.shape[1]),
        inference_latency_ms=latency_ms,
    )


def compare_color_and_refiner(
    base_logits: torch.Tensor,
    indices: torch.Tensor,
    valid_mask: torch.Tensor,
    structure_tokens: torch.Tensor,
    text_embeddings: torch.Tensor | None,
    palette: torch.Tensor,
    palette_mask: torch.Tensor,
    prompts: list[str],
    metadata: list[dict[str, Any]],
    refiner: PaletteLogitFlowRefiner | None = None,
    steps: tuple[int, ...] = (1, 2, 4),
) -> dict[str, Any]:
    baseline = compute_quality_metrics(base_logits, indices, valid_mask, palette_mask, prompts, metadata, structure_tokens=structure_tokens)
    out: dict[str, Any] = {"baseline_color": baseline.__dict__}
    if refiner is not None:
        for step_count, raw in compare_refinement(refiner, base_logits, indices, valid_mask, structure_tokens, text_embeddings, palette, palette_mask, steps=steps).items():
            out[f"flow_refiner_{step_count}"] = compute_quality_metrics(raw["logits"], indices, valid_mask, palette_mask, prompts, metadata, structure_tokens=structure_tokens).__dict__ | {
                "inference_latency_ms": raw["latency_ms"]
            }
    best = max(out, key=lambda key: out[key]["index_accuracy"] + out[key]["edge_detail_score"] + out[key]["text_image_alignment"])
    out["flow_refinement_default"] = best if best != "baseline_color" and out[best]["index_accuracy"] >= baseline.index_accuracy + 0.02 else "disabled"
    return out


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def save_comparison_gallery(
    outputs: dict[str, torch.Tensor],
    palette: torch.Tensor,
    palette_mask: torch.Tensor,
    path: str | Path,
    prompts: list[str],
    max_items: int = 8,
) -> None:
    names = list(outputs)
    rows = []
    font_h = 14
    for row in range(min(max_items, next(iter(outputs.values())).shape[0])):
        panels = []
        for name in names:
            row_palette = palette[row][palette_mask[row]]
            row_indices = outputs[name][row].clamp(0, max(0, len(row_palette) - 1))
            image = reconstruct_indexed_png(row_indices, row_palette)
            panels.append(np.asarray(image, dtype=np.uint8))
        strip = np.concatenate(panels, axis=1)
        label = Image.new("RGBA", (strip.shape[1], font_h), (255, 255, 255, 255))
        draw = ImageDraw.Draw(label)
        draw.text((2, 1), prompts[row % len(prompts)][:80], fill=(0, 0, 0, 255))
        rows.append(np.concatenate([np.asarray(label), strip], axis=0))
    canvas = np.concatenate(rows, axis=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas.astype(np.uint8), "RGBA").save(path)
