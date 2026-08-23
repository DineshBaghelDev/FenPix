from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .palette import image_to_indices


class PixelArtDataset(Dataset):
    def __init__(self, root: str | Path, max_colors: int = 64):
        self.root = Path(root)
        self.max_colors = max_colors
        self.paths = sorted(self.root.rglob("*.png"))
        if not self.paths:
            raise ValueError(f"no PNG files found under {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        with Image.open(path) as image:
            encoding = image_to_indices(image, max_colors=self.max_colors)

        meta_path = path.with_suffix(".json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        return {
            "path": str(path),
            "indices": torch.from_numpy(encoding.indices).long(),
            "palette": torch.from_numpy(encoding.palette).to(torch.uint8),
            "size": torch.tensor([encoding.width, encoding.height], dtype=torch.int32),
            "palette_size": torch.tensor(len(encoding.palette), dtype=torch.int16),
            "metadata": metadata,
        }


def pixel_art_collate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return samples
