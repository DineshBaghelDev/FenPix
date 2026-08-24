from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchy import structure_region_weights
from .maskgit import random_mask_tokens, maskgit_loss


@dataclass(frozen=True)
class DirectStructureConfig:
    vocab_size: int = 128
    hidden_dim: int = 96
    depth: int = 8
    text_dim: int = 64
    max_height: int = 128
    max_width: int = 128

    @property
    def mask_token_id(self) -> int:
        return self.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.vocab_size + 1


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    edge = torch.zeros_like(mask)
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :, 1:] |= mask[:, :, 1:] != mask[:, :, :-1]
    edge[:, :, :-1] |= mask[:, :, 1:] != mask[:, :, :-1]
    return edge


def _masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid[:, None].float()
    return (x * mask).flatten(2).sum(2) / mask.flatten(2).sum(2).clamp_min(1)


class DilatedConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.dw = nn.Conv2d(channels, channels, 5, padding=2 * dilation, dilation=dilation, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.pw2 = nn.Conv2d(channels * 2, channels, 1)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(F.gelu(self.norm1(x)))
        y = self.pw2(F.gelu(self.pw1(self.norm2(y))))
        return x + y


class DirectStructureGenerator(nn.Module):
    def __init__(self, config: DirectStructureConfig | None = None):
        super().__init__()
        self.config = config or DirectStructureConfig()
        self.token_embed = nn.Embedding(self.config.vocab_size + 2, self.config.hidden_dim)
        self.row_embed = nn.Embedding(self.config.max_height, self.config.hidden_dim)
        self.col_embed = nn.Embedding(self.config.max_width, self.config.hidden_dim)
        self.text_proj = nn.Linear(self.config.text_dim, self.config.hidden_dim) if self.config.text_dim else None
        dilations = (1, 2, 4)
        self.blocks = nn.ModuleList(
            DilatedConvBlock(self.config.hidden_dim, dilations[i % len(dilations)])
            for i in range(self.config.depth)
        )
        self.norm = nn.GroupNorm(1, self.config.hidden_dim)
        self.structure_head = nn.Conv2d(self.config.hidden_dim, self.config.vocab_size, 1)
        self.occupancy_head = nn.Conv2d(self.config.hidden_dim, 1, 1)
        self.boundary_head = nn.Conv2d(self.config.hidden_dim, 1, 1)
        self.count_head = nn.Linear(self.config.hidden_dim, 65)

    def encode(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
        cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        if tokens.shape != valid_mask.shape:
            raise ValueError("tokens and valid_mask must have the same shape")
        batch, height, width = tokens.shape
        if height > self.config.max_height or width > self.config.max_width:
            raise ValueError("token grid exceeds configured positional embeddings")
        safe = tokens.clamp(0, self.config.pad_token_id).masked_fill(~valid_mask, self.config.pad_token_id)
        rows = torch.arange(height, device=tokens.device)
        cols = torch.arange(width, device=tokens.device)
        x = self.token_embed(safe)
        x = x + self.row_embed(rows)[None, :, None, :] + self.col_embed(cols)[None, None, :, :]
        if self.text_proj is not None:
            if text_embeddings is None:
                text_embeddings = torch.zeros((batch, self.config.text_dim), device=tokens.device)
            text_embeddings = text_embeddings.to(tokens.device)
            if cond_drop_prob > 0:
                keep = torch.rand((batch, 1), device=tokens.device).ge(cond_drop_prob)
                text_embeddings = text_embeddings * keep
            x = x + self.text_proj(text_embeddings)[:, None, None]
        x = x.permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x)
        return F.gelu(self.norm(x)).masked_fill(~valid_mask[:, None], 0)

    def forward(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
        cond_drop_prob: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        x = self.encode(tokens, valid_mask, text_embeddings, cond_drop_prob)
        return {
            "logits": self.structure_head(x).masked_fill(~valid_mask[:, None], -1e4),
            "occupancy_logits": self.occupancy_head(x).squeeze(1).masked_fill(~valid_mask, -1e4),
            "boundary_logits": self.boundary_head(x).squeeze(1).masked_fill(~valid_mask, -1e4),
            "count_logits": self.count_head(_masked_mean(x, valid_mask)),
        }

    def loss(
        self,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
        foreground_weight: float = 2.0,
        boundary_weight: float = 2.0,
        occupancy_loss_weight: float = 0.25,
        boundary_loss_weight: float = 0.25,
        count_loss_weight: float = 0.05,
        cond_drop_prob: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        targets = targets.clamp(0, self.config.vocab_size - 1)
        masked, labels = random_mask_tokens(targets, valid_mask, self.config.mask_token_id)
        out = self(masked, valid_mask, text_embeddings, cond_drop_prob)
        weights = structure_region_weights(targets, valid_mask, targets.shape[-2:], foreground_weight, boundary_weight)
        structure_loss = maskgit_loss(out["logits"], labels, weights)
        occupancy = targets.ne(0) & valid_mask
        occupancy_loss = F.binary_cross_entropy_with_logits(out["occupancy_logits"], occupancy.float(), reduction="none")
        occupancy_loss = (occupancy_loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1)
        boundary = _boundary(occupancy)
        boundary_loss = F.binary_cross_entropy_with_logits(out["boundary_logits"], boundary.float(), reduction="none")
        boundary_loss = (boundary_loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1)
        count_target = targets.masked_fill(~valid_mask, 0).amax(dim=(1, 2)).clamp_max(64)
        count_loss = F.cross_entropy(out["count_logits"], count_target)
        loss = structure_loss + occupancy_loss * occupancy_loss_weight + boundary_loss * boundary_loss_weight + count_loss * count_loss_weight
        return {
            "loss": loss,
            "structure_loss": structure_loss,
            "occupancy_loss": occupancy_loss,
            "boundary_loss": boundary_loss,
            "count_loss": count_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        valid_mask: torch.Tensor | None = None,
        text_embeddings: torch.Tensor | None = None,
        steps: int = 8,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        batch, height, width = shape
        device = next(self.parameters()).device
        valid = torch.ones(shape, dtype=torch.bool, device=device) if valid_mask is None else valid_mask.to(device)
        tokens = torch.full(shape, self.config.mask_token_id, dtype=torch.long, device=device).masked_fill(~valid, self.config.pad_token_id)
        for step in range(max(1, steps)):
            still_masked = tokens.eq(self.config.mask_token_id) & valid
            if not still_masked.any():
                break
            logits = self(tokens, valid, text_embeddings)["logits"]
            if text_embeddings is not None and guidance_scale != 1.0:
                uncond = self(tokens, valid, None)["logits"]
                logits = uncond + (logits - uncond) * guidance_scale
            probs = (logits.clamp(-50, 50) / max(temperature, 1e-6)).softmax(dim=1)
            sampled = torch.multinomial(probs.permute(0, 2, 3, 1).reshape(-1, self.config.vocab_size), 1).reshape(shape)
            confidence = probs.gather(1, sampled[:, None]).squeeze(1).masked_fill(~still_masked, -1)
            remaining = max(1, steps - step)
            flat_tokens = tokens.flatten(1)
            flat_sampled = sampled.flatten(1)
            flat_conf = confidence.flatten(1)
            flat_mask = still_masked.flatten(1)
            for b in range(batch):
                count = int(flat_mask[b].sum())
                take = max(1, (count + remaining - 1) // remaining)
                chosen = flat_conf[b].topk(take).indices
                flat_tokens[b, chosen] = flat_sampled[b, chosen]
        return tokens.masked_fill(~valid, self.config.pad_token_id)

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "DirectStructureGenerator":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(DirectStructureConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model
