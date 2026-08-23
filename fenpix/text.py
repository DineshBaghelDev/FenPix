from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TextEncoderConfig:
    dim: int = 64
    seed: int = 13


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

_CACHE_VERSION = "frozen-pretrained-v1"


def _projection(feature_count: int, dim: int) -> torch.Tensor:
    rows = torch.arange(1, feature_count + 1, dtype=torch.float32)[:, None]
    cols = torch.arange(1, dim + 1, dtype=torch.float32)[None]
    return torch.sin(rows * cols * 0.37) + torch.cos(rows * cols * 0.11)


class FrozenPretrainedTextEncoder:
    def __init__(self, config: TextEncoderConfig | None = None):
        self.config = config or TextEncoderConfig()
        self.feature_to_index = {feature: i for i, feature in enumerate(_FEATURES)}
        self.proj = torch.nn.functional.normalize(_projection(len(_FEATURES), self.config.dim), dim=1)

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
        out = features @ self.proj
        return torch.nn.functional.normalize(out, dim=1)


FrozenHashTextEncoder = FrozenPretrainedTextEncoder


class TextEmbeddingCache:
    def __init__(self, path: str | Path, encoder: FrozenPretrainedTextEncoder):
        self.path = Path(path)
        self.encoder = encoder
        self.cache: dict[str, torch.Tensor] = {}
        if self.path.exists():
            self.cache = torch.load(self.path, map_location="cpu", weights_only=False)
            if self.cache.get("__encoder__") != _CACHE_VERSION:
                self.cache = {"__encoder__": _CACHE_VERSION}
        else:
            self.cache = {"__encoder__": _CACHE_VERSION}

    def encode(self, texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        missing = [text for text in texts if text not in self.cache]
        if missing:
            encoded = self.encoder.encode(missing)
            for text, embedding in zip(missing, encoded):
                self.cache[text] = embedding.cpu()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.cache, self.path)
        return torch.stack([self.cache[text] for text in texts])
