from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .maskgit import MaskGIT, MaskGITConfig, random_mask_tokens
from .palette import reconstruct_rgba


@dataclass(frozen=True)
class IndexedColorConfig:
    max_colors: int = 64
    min_colors: int = 8
    structure_vocab_size: int = 128
    hidden_dim: int = 64
    depth: int = 2
    heads: int = 4
    text_dim: int = 64
    cond_tokens: int = 1
    max_height: int = 128
    max_width: int = 128


def palette_mask_from_sizes(sizes: torch.Tensor, max_colors: int = 64) -> torch.Tensor:
    return torch.arange(max_colors, device=sizes.device)[None, :] < sizes[:, None]


def palette_to_uint8(palette_logits: torch.Tensor, sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    palette = (palette_logits.sigmoid() * 255).round().clamp(0, 255).to(torch.uint8)
    mask = palette_mask_from_sizes(sizes, palette.shape[1])
    palette = palette.masked_fill(~mask[:, :, None], 0)
    transparent = palette[:, :, 3] < 128
    palette = palette.masked_fill(transparent[:, :, None], 0)
    return palette, mask


def reconstruct_indexed_png(indices: torch.Tensor | np.ndarray, palette: torch.Tensor | np.ndarray):
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    if isinstance(palette, torch.Tensor):
        palette = palette.detach().cpu().numpy()
    return reconstruct_rgba(indices, palette)


def _pad_palette_batch(palette: torch.Tensor, palette_mask: torch.Tensor, max_colors: int) -> tuple[torch.Tensor, torch.Tensor]:
    if palette.shape[1] == max_colors:
        return palette, palette_mask
    out = torch.zeros((palette.shape[0], max_colors, 4), dtype=palette.dtype, device=palette.device)
    mask = torch.zeros((palette.shape[0], max_colors), dtype=palette_mask.dtype, device=palette_mask.device)
    take = min(palette.shape[1], max_colors)
    out[:, :take] = palette[:, :take]
    mask[:, :take] = palette_mask[:, :take]
    return out, mask


class IndexedColorModel(nn.Module):
    def __init__(self, config: IndexedColorConfig | None = None):
        super().__init__()
        self.config = config or IndexedColorConfig()
        if not 8 <= self.config.min_colors <= self.config.max_colors <= 64:
            raise ValueError("palette range must be within 8..64")
        self.structure_embed = nn.Embedding(self.config.structure_vocab_size + 2, self.config.hidden_dim)
        cond_dim = self.config.hidden_dim + self.config.text_dim
        self.size_head = nn.Sequential(nn.Linear(cond_dim, self.config.hidden_dim), nn.GELU(), nn.Linear(self.config.hidden_dim, self.config.max_colors - self.config.min_colors + 1))
        self.palette_head = nn.Sequential(nn.Linear(cond_dim, self.config.hidden_dim), nn.GELU(), nn.Linear(self.config.hidden_dim, self.config.max_colors * 4))
        self.palette_cond = nn.Linear(self.config.max_colors * 4, self.config.text_dim) if self.config.text_dim else None
        self.index_model = MaskGIT(
            MaskGITConfig(
                vocab_size=self.config.max_colors,
                hidden_dim=self.config.hidden_dim,
                depth=self.config.depth,
                heads=self.config.heads,
                max_height=self.config.max_height,
                max_width=self.config.max_width,
                text_dim=self.config.text_dim,
                cond_tokens=self.config.cond_tokens,
                structure_cond=True,
                structure_vocab_size=self.config.structure_vocab_size,
            )
        )

    def _pooled(self, structure_tokens: torch.Tensor, valid_mask: torch.Tensor, text_embeddings: torch.Tensor | None) -> torch.Tensor:
        safe = structure_tokens.clamp(0, self.config.structure_vocab_size + 1).masked_fill(~valid_mask, self.config.structure_vocab_size + 1)
        emb = self.structure_embed(safe) * valid_mask[..., None]
        pooled = emb.flatten(1, 2).sum(1) / valid_mask.flatten(1).sum(1).clamp_min(1)[:, None]
        if self.config.text_dim:
            if text_embeddings is None:
                text_embeddings = torch.zeros((structure_tokens.shape[0], self.config.text_dim), device=structure_tokens.device)
            pooled = torch.cat([pooled, text_embeddings.to(structure_tokens.device)], dim=1)
        return pooled

    def predict_palette(
        self,
        structure_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pooled = self._pooled(structure_tokens, valid_mask, text_embeddings)
        size_logits = self.size_head(pooled)
        size = size_logits.argmax(dim=1) + self.config.min_colors
        palette_logits = self.palette_head(pooled).reshape(-1, self.config.max_colors, 4)
        palette, palette_mask = palette_to_uint8(palette_logits, size)
        return {"size_logits": size_logits, "palette_logits": palette_logits, "palette_size": size, "palette": palette, "palette_mask": palette_mask}

    def _index_text(self, text_embeddings: torch.Tensor | None, palette: torch.Tensor, palette_mask: torch.Tensor) -> torch.Tensor | None:
        if self.config.text_dim == 0:
            return None
        batch = palette.shape[0]
        if text_embeddings is None:
            text_embeddings = torch.zeros((batch, self.config.text_dim), device=palette.device)
        padded = palette.float().masked_fill(~palette_mask[:, :, None], 0) / 255.0
        return text_embeddings.to(palette.device) + self.palette_cond(padded.flatten(1))

    def palette_loss(self, prediction: dict[str, torch.Tensor], palette: torch.Tensor, palette_mask: torch.Tensor, palette_size: torch.Tensor) -> torch.Tensor:
        palette, palette_mask = _pad_palette_batch(palette, palette_mask, self.config.max_colors)
        target_size = palette_size.long().clamp(self.config.min_colors, self.config.max_colors) - self.config.min_colors
        size_loss = F.cross_entropy(prediction["size_logits"], target_size)
        target = palette.float().to(prediction["palette_logits"].device) / 255.0
        mask = palette_mask.to(prediction["palette_logits"].device)
        color_loss = F.mse_loss(prediction["palette_logits"].sigmoid()[mask], target[mask]) if mask.any() else prediction["palette_logits"].sum() * 0
        return size_loss + color_loss

    def index_loss(
        self,
        indices: torch.Tensor,
        valid_mask: torch.Tensor,
        structure_tokens: torch.Tensor,
        text_embeddings: torch.Tensor | None,
        palette: torch.Tensor,
        palette_mask: torch.Tensor,
        cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        palette, palette_mask = _pad_palette_batch(palette, palette_mask, self.config.max_colors)
        masked, labels = random_mask_tokens(indices.clamp_min(0), valid_mask, self.index_model.config.mask_token_id)
        logits = self.index_model(masked, valid_mask, self._index_text(text_embeddings, palette, palette_mask), structure_tokens, cond_drop_prob)
        logits = logits.masked_fill(~palette_mask[:, :, None, None].to(logits.device), -1e9)
        return F.cross_entropy(logits.permute(0, 2, 3, 1).reshape(-1, self.config.max_colors), labels.reshape(-1), ignore_index=-100)

    def loss(
        self,
        indices: torch.Tensor,
        valid_mask: torch.Tensor,
        structure_tokens: torch.Tensor,
        text_embeddings: torch.Tensor | None,
        palette: torch.Tensor,
        palette_mask: torch.Tensor,
        palette_size: torch.Tensor,
        cond_drop_prob: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        pred = self.predict_palette(structure_tokens, valid_mask, text_embeddings)
        palette_loss = self.palette_loss(pred, palette, palette_mask, palette_size)
        index_loss = self.index_loss(indices, valid_mask, structure_tokens, text_embeddings, palette, palette_mask, cond_drop_prob)
        return {"loss": palette_loss + index_loss, "palette_loss": palette_loss, "index_loss": index_loss}

    @torch.no_grad()
    def sample(
        self,
        structure_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
        steps: int = 8,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        pred = self.predict_palette(structure_tokens, valid_mask, text_embeddings)
        indices = self._sample_indices(
            structure_tokens,
            valid_mask,
            self._index_text(text_embeddings, pred["palette"].float(), pred["palette_mask"]),
            pred["palette_mask"],
            steps,
            temperature,
            guidance_scale,
        )
        return pred | {"indices": indices}

    def _sample_indices(
        self,
        structure_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        text_embeddings: torch.Tensor | None,
        palette_mask: torch.Tensor,
        steps: int,
        temperature: float,
        guidance_scale: float,
    ) -> torch.Tensor:
        shape = structure_tokens.shape
        tokens = torch.full(shape, self.index_model.config.mask_token_id, dtype=torch.long, device=structure_tokens.device)
        tokens = tokens.masked_fill(~valid_mask, self.index_model.config.pad_token_id)
        for step in range(max(1, steps)):
            still_masked = tokens.eq(self.index_model.config.mask_token_id) & valid_mask
            if not still_masked.any():
                break
            logits = self.index_model(tokens, valid_mask, text_embeddings, structure_tokens)
            if text_embeddings is not None and guidance_scale != 1.0:
                uncond = self.index_model(tokens, valid_mask, None, structure_tokens)
                logits = uncond + (logits - uncond) * guidance_scale
            logits = logits.masked_fill(~palette_mask[:, :, None, None].to(logits.device), -1e9)
            probs = (logits.clamp(-50, 50) / max(temperature, 1e-6)).softmax(dim=1)
            sampled = torch.multinomial(probs.permute(0, 2, 3, 1).reshape(-1, self.config.max_colors), 1).reshape(shape)
            confidence = probs.gather(1, sampled[:, None]).squeeze(1).masked_fill(~still_masked, -1)
            flat_tokens = tokens.flatten(1)
            flat_sampled = sampled.flatten(1)
            flat_conf = confidence.flatten(1)
            flat_mask = still_masked.flatten(1)
            remaining = max(1, steps - step)
            for b in range(shape[0]):
                count = int(flat_mask[b].sum())
                take = max(1, (count + remaining - 1) // remaining)
                chosen = flat_conf[b].topk(take).indices
                flat_tokens[b, chosen] = flat_sampled[b, chosen]
        return tokens.masked_fill(~valid_mask, self.index_model.config.pad_token_id)

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "IndexedColorModel":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(IndexedColorConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model
