from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FlowRefinerConfig:
    max_colors: int = 64
    structure_vocab_size: int = 128
    hidden_dim: int = 64
    depth: int = 2
    text_dim: int = 64


def _palette_mask_from_palette(palette_mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return palette_mask[:, :, None, None].expand(-1, -1, height, width)


def _pad_palette_mask(palette_mask: torch.Tensor, max_colors: int) -> torch.Tensor:
    if palette_mask.shape[1] == max_colors:
        return palette_mask
    out = torch.zeros((palette_mask.shape[0], max_colors), dtype=torch.bool, device=palette_mask.device)
    out[:, : min(palette_mask.shape[1], max_colors)] = palette_mask[:, :max_colors]
    return out


class PaletteLogitFlowRefiner(nn.Module):
    def __init__(self, config: FlowRefinerConfig | None = None):
        super().__init__()
        self.config = config or FlowRefinerConfig()
        self.structure_embed = nn.Embedding(self.config.structure_vocab_size + 2, self.config.hidden_dim)
        self.in_proj = nn.Conv2d(self.config.max_colors + self.config.hidden_dim, self.config.hidden_dim, 1)
        self.palette_proj = nn.Linear(self.config.max_colors * 4, self.config.hidden_dim)
        self.text_proj = nn.Linear(self.config.text_dim, self.config.hidden_dim) if self.config.text_dim else None
        self.time_proj = nn.Linear(1, self.config.hidden_dim)
        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(self.config.hidden_dim, self.config.hidden_dim, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(self.config.hidden_dim, self.config.hidden_dim, 3, padding=1),
                    nn.GELU(),
                )
                for _ in range(self.config.depth)
            ]
        )
        self.out_proj = nn.Conv2d(self.config.hidden_dim, self.config.max_colors, 1)

    def _cond(
        self,
        structure_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        text_embeddings: torch.Tensor | None,
        palette: torch.Tensor,
        palette_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        batch, height, width = structure_tokens.shape
        structure_pad = self.config.structure_vocab_size + 1
        safe_structure = structure_tokens.clamp(0, structure_pad).masked_fill(~valid_mask, structure_pad)
        structure = self.structure_embed(safe_structure).permute(0, 3, 1, 2)
        padded_palette = torch.zeros((batch, self.config.max_colors, 4), dtype=palette.dtype, device=palette.device)
        take = min(palette.shape[1], self.config.max_colors)
        padded_palette[:, :take] = palette[:, :take]
        padded_mask = torch.zeros((batch, self.config.max_colors), dtype=torch.bool, device=palette.device)
        padded_mask[:, : min(palette_mask.shape[1], self.config.max_colors)] = palette_mask[:, : self.config.max_colors]
        palette_vec = self.palette_proj((padded_palette.float().masked_fill(~padded_mask[:, :, None], 0) / 255).flatten(1))
        cond = palette_vec
        if self.text_proj is not None:
            if text_embeddings is None:
                text_embeddings = torch.zeros((batch, self.config.text_dim), device=palette.device)
            cond = cond + self.text_proj(text_embeddings.to(palette.device))
        cond = cond + self.time_proj(t.reshape(batch, 1).to(palette.device).float())
        return structure + cond[:, :, None, None].expand(-1, -1, height, width)

    def forward(
        self,
        logits: torch.Tensor,
        t: torch.Tensor,
        valid_mask: torch.Tensor,
        structure_tokens: torch.Tensor,
        text_embeddings: torch.Tensor | None,
        palette: torch.Tensor,
        palette_mask: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape[-2:] != valid_mask.shape[-2:] or logits.shape[0] != valid_mask.shape[0]:
            raise ValueError("logits and valid_mask must have matching batch and spatial shape")
        cond = self._cond(structure_tokens, valid_mask, text_embeddings, palette, palette_mask, t)
        x = self.in_proj(torch.cat([logits, cond], dim=1))
        return self.out_proj(self.blocks(x)).masked_fill(~valid_mask[:, None], 0)

    def loss(
        self,
        base_logits: torch.Tensor,
        indices: torch.Tensor,
        valid_mask: torch.Tensor,
        structure_tokens: torch.Tensor,
        text_embeddings: torch.Tensor | None,
        palette: torch.Tensor,
        palette_mask: torch.Tensor,
    ) -> torch.Tensor:
        base = base_logits.detach().clamp(-20, 20)
        target = F.one_hot(indices.clamp_min(0), self.config.max_colors).permute(0, 3, 1, 2).float()
        t = torch.rand((base_logits.shape[0],), device=base_logits.device)
        x_t = base * (1 - t[:, None, None, None]) + target * t[:, None, None, None]
        velocity = target - base
        pred = self(x_t, t, valid_mask, structure_tokens, text_embeddings, palette, palette_mask)
        channel_mask = _palette_mask_from_palette(_pad_palette_mask(palette_mask.to(base.device), self.config.max_colors), base.shape[-2], base.shape[-1])
        mask = channel_mask & valid_mask[:, None]
        return F.mse_loss(pred[mask], velocity[mask])

    @torch.no_grad()
    def refine(
        self,
        logits: torch.Tensor,
        steps: int,
        valid_mask: torch.Tensor,
        structure_tokens: torch.Tensor,
        text_embeddings: torch.Tensor | None,
        palette: torch.Tensor,
        palette_mask: torch.Tensor,
    ) -> torch.Tensor:
        steps = int(steps)
        if steps <= 0:
            return logits.masked_fill(~valid_mask[:, None], 0)
        x = logits.clamp(-20, 20)
        palette_mask = _pad_palette_mask(palette_mask.to(x.device), self.config.max_colors)
        for step in range(steps):
            t = torch.full((logits.shape[0],), step / steps, device=logits.device)
            x = x + self(x, t, valid_mask, structure_tokens, text_embeddings, palette, palette_mask) / steps
            x = x.masked_fill(~_palette_mask_from_palette(palette_mask.to(x.device), x.shape[-2], x.shape[-1]), -20)
        x = x.masked_fill(~_palette_mask_from_palette(palette_mask.to(x.device), x.shape[-2], x.shape[-1]), -1e9)
        return x.masked_fill(~valid_mask[:, None], 0)

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "PaletteLogitFlowRefiner":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(FlowRefinerConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model


@torch.no_grad()
def compare_refinement(
    refiner: PaletteLogitFlowRefiner,
    base_logits: torch.Tensor,
    indices: torch.Tensor,
    valid_mask: torch.Tensor,
    structure_tokens: torch.Tensor,
    text_embeddings: torch.Tensor | None,
    palette: torch.Tensor,
    palette_mask: torch.Tensor,
    steps: tuple[int, ...] = (0, 1, 2, 4),
) -> dict[int, dict[str, float | torch.Tensor]]:
    out: dict[int, dict[str, float | torch.Tensor]] = {}
    for step_count in steps:
        start = perf_counter()
        logits = refiner.refine(base_logits, step_count, valid_mask, structure_tokens, text_embeddings, palette, palette_mask)
        latency_ms = (perf_counter() - start) * 1000
        pred = logits.argmax(dim=1).masked_fill(~valid_mask, -1)
        accuracy = pred.eq(indices).logical_and(valid_mask).float().sum() / valid_mask.float().sum().clamp_min(1)
        valid_palette = pred.ge(0) & (pred < palette_mask.sum(1)[:, None, None].to(pred.device))
        palette_consistency = valid_palette.logical_or(~valid_mask).float().mean()
        target_transparent = indices.eq(0) & valid_mask
        pred_transparent = pred.eq(0) & valid_mask
        transparency = pred_transparent.eq(target_transparent).logical_and(valid_mask).float().sum() / valid_mask.float().sum().clamp_min(1)
        edge_target_h = indices[:, 1:] != indices[:, :-1]
        edge_pred_h = pred[:, 1:] != pred[:, :-1]
        edge_valid_h = valid_mask[:, 1:] & valid_mask[:, :-1]
        edge_target_w = indices[:, :, 1:] != indices[:, :, :-1]
        edge_pred_w = pred[:, :, 1:] != pred[:, :, :-1]
        edge_valid_w = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]
        edge_hits = edge_pred_h.eq(edge_target_h).logical_and(edge_valid_h).float().sum()
        edge_hits = edge_hits + edge_pred_w.eq(edge_target_w).logical_and(edge_valid_w).float().sum()
        edge_total = edge_valid_h.float().sum() + edge_valid_w.float().sum()
        edge_accuracy = edge_hits / edge_total.clamp_min(1)
        out[step_count] = {
            "indices": pred,
            "logits": logits,
            "index_accuracy": float(accuracy.item()),
            "edge_detail": float(edge_accuracy.item()),
            "palette_consistency": float(palette_consistency.item()),
            "transparency": float(transparency.item()),
            "latency_ms": latency_ms,
        }
    return out
