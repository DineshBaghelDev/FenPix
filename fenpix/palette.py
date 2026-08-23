from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


RGBA = tuple[int, int, int, int]


@dataclass(frozen=True)
class StructureEncoding:
    indices: np.ndarray
    width: int
    height: int
    metadata: dict[str, Any]
    lossy: bool
    unique_color_count: int


@dataclass(frozen=True)
class PaletteEncoding:
    structure: StructureEncoding
    palette: np.ndarray

    @property
    def indices(self) -> np.ndarray:
        return self.structure.indices

    @property
    def width(self) -> int:
        return self.structure.width

    @property
    def height(self) -> int:
        return self.structure.height

    @property
    def metadata(self) -> dict[str, Any]:
        return self.structure.metadata

    @property
    def lossy(self) -> bool:
        return self.structure.lossy

    @property
    def unique_color_count(self) -> int:
        return self.structure.unique_color_count


def _as_rgba_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def _unique_rows(rows: np.ndarray) -> np.ndarray:
    if len(rows) == 0:
        return np.empty((0, 4), dtype=np.uint8)
    unique = np.unique(rows, axis=0)
    order = np.lexsort((unique[:, 2], unique[:, 1], unique[:, 0], unique[:, 3]))
    return unique[order].astype(np.uint8)


def _quantize_rgb(rows: np.ndarray, max_colors: int) -> np.ndarray:
    if max_colors <= 0 or len(rows) == 0:
        return np.empty((0, 4), dtype=np.uint8)

    unique_rgb = np.unique(rows[:, :3], axis=0)
    if len(unique_rgb) <= max_colors:
        alpha = np.full((len(unique_rgb), 1), 255, dtype=np.uint8)
        return np.concatenate([unique_rgb, alpha], axis=1)

    strip = Image.fromarray(unique_rgb.reshape(1, len(unique_rgb), 3), "RGB")
    quantized = strip.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
    palette = np.array(quantized.getpalette()[: max_colors * 3], dtype=np.uint8)
    palette = palette.reshape(-1, 3)
    used = np.unique(np.asarray(quantized, dtype=np.uint8))
    rgb = palette[used]
    alpha = np.full((len(rgb), 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=1)


def extract_palette(image: Image.Image, max_colors: int = 64) -> np.ndarray:
    if not 8 <= max_colors <= 64:
        raise ValueError("max_colors must be between 8 and 64")

    rgba = _as_rgba_array(image).reshape(-1, 4)
    transparent = rgba[:, 3] == 0
    visible = rgba[~transparent]

    if len(_unique_rows(rgba)) <= max_colors:
        palette = _unique_rows(rgba)
    else:
        transparent_palette = (
            np.array([[0, 0, 0, 0]], dtype=np.uint8) if transparent.any() else np.empty((0, 4), dtype=np.uint8)
        )
        visible_palette = _quantize_rgb(visible, max_colors - len(transparent_palette))
        palette = np.concatenate([transparent_palette, visible_palette], axis=0)

    if len(palette) == 0:
        return np.array([[0, 0, 0, 0]], dtype=np.uint8)
    return palette[:max_colors].astype(np.uint8)


def image_to_indices(image: Image.Image, max_colors: int = 64) -> PaletteEncoding:
    rgba = _as_rgba_array(image)
    height, width = rgba.shape[:2]
    flat = rgba.reshape(-1, 4)
    unique_colors = _unique_rows(flat)
    palette = extract_palette(image, max_colors=max_colors)
    lossy = len(unique_colors) > max_colors

    if not lossy:
        lookup: dict[RGBA, int] = {tuple(color): i for i, color in enumerate(palette.tolist())}
        indices = np.array([lookup[tuple(pixel)] for pixel in flat], dtype=np.int64)
    else:
        transparent_index = 0 if tuple(palette[0]) == (0, 0, 0, 0) else None
        candidate_indices = np.arange(len(palette), dtype=np.int64)
        if transparent_index is not None:
            candidate_indices = candidate_indices[1:]
        visible_palette = palette[candidate_indices, :3].astype(np.int32)
        distances = ((flat[:, None, :3].astype(np.int32) - visible_palette[None, :, :]) ** 2).sum(axis=2)
        indices = candidate_indices[distances.argmin(axis=1)]
        if transparent_index is not None:
            indices[flat[:, 3] == 0] = transparent_index

    metadata = dict(image.info)
    return PaletteEncoding(
        structure=StructureEncoding(
            indices=indices.reshape(height, width),
            width=width,
            height=height,
            metadata=metadata,
            lossy=lossy,
            unique_color_count=len(unique_colors),
        ),
        palette=palette,
    )


def reconstruct_rgba(indices: np.ndarray, palette: np.ndarray) -> Image.Image:
    indices = np.asarray(indices, dtype=np.int64)
    palette = np.asarray(palette, dtype=np.uint8)
    if indices.size and (indices.min() < 0 or indices.max() >= len(palette)):
        raise ValueError("indices reference colors outside the palette")
    rgba = palette[indices]
    return Image.fromarray(rgba.astype(np.uint8), "RGBA")
