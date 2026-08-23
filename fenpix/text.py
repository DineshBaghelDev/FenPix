from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TextEncoderConfig:
    dim: int = 64
    buckets: int = 512
    seed: int = 13


class FrozenHashTextEncoder:
    def __init__(self, config: TextEncoderConfig | None = None):
        self.config = config or TextEncoderConfig()
        generator = torch.Generator().manual_seed(self.config.seed)
        self.proj = torch.randn(self.config.buckets, self.config.dim, generator=generator) / self.config.dim**0.5

    def encode(self, texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        out = torch.zeros((len(texts), self.config.dim), dtype=torch.float32)
        for row, text in enumerate(texts):
            words = text.lower().replace("_", " ").split()
            for token in words or [""]:
                key = hashlib.sha256(token.encode("utf-8")).digest()
                out[row] += self.proj[int.from_bytes(key[:4], "little") % self.config.buckets]
        return torch.nn.functional.normalize(out, dim=1)


class TextEmbeddingCache:
    def __init__(self, path: str | Path, encoder: FrozenHashTextEncoder):
        self.path = Path(path)
        self.encoder = encoder
        self.cache: dict[str, torch.Tensor] = {}
        if self.path.exists():
            self.cache = torch.load(self.path, map_location="cpu", weights_only=False)

    def encode(self, texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        missing = [text for text in texts if text not in self.cache]
        if missing:
            encoded = self.encoder.encode(missing)
            for text, embedding in zip(missing, encoded):
                self.cache[text] = embedding.cpu()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.cache, self.path)
        return torch.stack([self.cache[text] for text in texts])
