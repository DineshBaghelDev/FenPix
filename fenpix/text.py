from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TextEncoderConfig:
    dim: int = 512
    seed: int = 13
    model_name: str = "openai/clip-vit-base-patch32"
    provider: str = "clip"
    device: str = "cpu"
    local_files_only: bool = False


_FEATURES = (
    "red green blue yellow orange purple pink white black gray brown stone wood metal grass water fire ice light dark "
    "sprite icon tile object building scene isometric transparent opaque edge detail flat round small large fantasy nature"
).split()

_TOKEN_FEATURES: dict[str, tuple[str, ...]] = {
    "potion": ("object", "transparent", "round", "fantasy"),
    "tree": ("object", "green", "brown", "nature"),
    "house": ("building", "stone", "wood"),
    "castle": ("building", "stone", "fantasy"),
    "grass": ("tile", "green", "nature"),
    "water": ("tile", "water", "blue"),
    "stone": ("stone", "gray", "tile"),
    "wood": ("wood", "brown"),
    "coin": ("object", "yellow", "metal", "round"),
    "sword": ("object", "metal", "edge"),
    "knight": ("sprite", "metal", "fantasy"),
    "sprite": ("sprite", "transparent", "detail"),
    "icon": ("icon", "transparent", "edge"),
    "tile": ("tile", "flat"),
    "object": ("object", "transparent"),
    "building": ("building", "opaque"),
    "scene": ("scene", "large"),
    "isometric": ("isometric", "building", "object"),
    "transparent": ("transparent",),
}


def _projection(rows: int, cols: int) -> torch.Tensor:
    r = torch.arange(1, rows + 1, dtype=torch.float32)[:, None]
    c = torch.arange(1, cols + 1, dtype=torch.float32)[None]
    return F.normalize(torch.sin(r * c * 0.37) + torch.cos(r * c * 0.11), dim=0)


class FrozenTinyTextEncoder:
    def __init__(self, config: TextEncoderConfig | None = None):
        self.config = config or TextEncoderConfig(provider="tiny", dim=64)
        self.feature_to_index = {feature: i for i, feature in enumerate(_FEATURES)}
        self.proj = F.normalize(_projection(len(_FEATURES), self.config.dim), dim=1)

    def encode(self, texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        features = torch.zeros((len(texts), len(_FEATURES)), dtype=torch.float32)
        for row, text in enumerate(texts):
            words = text.lower().replace("_", " ").split()
            for token in words:
                for feature in _TOKEN_FEATURES.get(token, (token,)):
                    index = self.feature_to_index.get(feature)
                    if index is not None:
                        features[row, index] += 1
            if not features[row].any():
                features[row, self.feature_to_index["object"]] = 1
        return F.normalize(features @ self.proj, dim=1)


class FrozenPretrainedTextEncoder:
    def __init__(self, config: TextEncoderConfig | None = None):
        self.config = config or TextEncoderConfig()
        if self.config.provider == "tiny":
            self._tiny = FrozenTinyTextEncoder(self.config)
            self.model = None
            self.processor = None
            self.out_proj = None
            return
        if self.config.provider != "clip":
            raise ValueError("provider must be 'clip' or 'tiny'")
        from transformers import CLIPModel, CLIPProcessor

        self._tiny = None
        self.device = torch.device(self.config.device)
        self.processor = CLIPProcessor.from_pretrained(self.config.model_name, local_files_only=self.config.local_files_only)
        self.model = CLIPModel.from_pretrained(self.config.model_name, local_files_only=self.config.local_files_only).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        hidden = int(self.model.config.projection_dim)
        self.out_proj = _projection(hidden, self.config.dim) if hidden != self.config.dim else None

    @torch.no_grad()
    def encode(self, texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        if self._tiny is not None:
            return self._tiny.encode(texts)
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(self.device)
        features = self.model.get_text_features(**inputs)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        features = features.float().cpu()
        if self.out_proj is not None:
            features = features @ self.out_proj
        return F.normalize(features, dim=1)


class FrozenVisionLanguageEncoder(FrozenPretrainedTextEncoder):
    @torch.no_grad()
    def encode_images(self, images: list[Any]) -> torch.Tensor:
        if self._tiny is not None:
            rows = []
            for image in images:
                arr = np.asarray(image.convert("RGBA") if hasattr(image, "convert") else image, dtype=np.float32)
                rows.append(torch.tensor([float(arr[..., :3].mean()), float(arr[..., :3].std()), float(arr[..., 3].mean())]))
            raw = torch.stack(rows)
            return F.normalize(raw @ _projection(raw.shape[1], self.config.dim), dim=1)
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        features = features.float().cpu()
        if self.out_proj is not None:
            features = features @ self.out_proj
        return F.normalize(features, dim=1)


FrozenHashTextEncoder = FrozenTinyTextEncoder


class TextEmbeddingCache:
    def __init__(self, path: str | Path, encoder: FrozenPretrainedTextEncoder | FrozenTinyTextEncoder):
        self.path = Path(path)
        self.encoder = encoder
        self.cache: dict[str, torch.Tensor | str] = {}
        version = f"{encoder.config.provider}:{encoder.config.model_name}:{encoder.config.dim}"
        if self.path.exists():
            self.cache = torch.load(self.path, map_location="cpu", weights_only=False)
            if self.cache.get("__encoder__") != version:
                self.cache = {"__encoder__": version}
        else:
            self.cache = {"__encoder__": version}

    def encode(self, texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        missing = [text for text in texts if text not in self.cache]
        if missing:
            encoded = self.encoder.encode(missing)
            for text, embedding in zip(missing, encoded):
                self.cache[text] = embedding.cpu()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.cache, self.path)
        return torch.stack([self.cache[text] for text in texts])  # type: ignore[list-item]
