from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MaskGITConfig:
    vocab_size: int = 128
    hidden_dim: int = 128
    depth: int = 4
    heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.0
    max_height: int = 32
    max_width: int = 32
    text_dim: int = 0
    cond_tokens: int = 1
    structure_cond: bool = False
    structure_vocab_size: int | None = None

    @property
    def mask_token_id(self) -> int:
        return self.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.vocab_size + 1


def random_mask_tokens(
    tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_token_id: int,
    min_ratio: float = 0.1,
    max_ratio: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.shape != valid_mask.shape:
        raise ValueError("tokens and valid_mask must have the same shape")
    if not 0 < min_ratio <= max_ratio <= 1:
        raise ValueError("mask ratios must satisfy 0 < min <= max <= 1")

    masked = tokens.clone()
    labels = torch.full_like(tokens, -100)
    flat_valid = valid_mask.flatten(1)
    flat_masked = masked.flatten(1)
    flat_labels = labels.flatten(1)
    flat_tokens = tokens.flatten(1)
    device = tokens.device

    for b in range(tokens.shape[0]):
        valid_indices = flat_valid[b].nonzero(as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            continue
        ratio = torch.empty((), device=device).uniform_(min_ratio, max_ratio, generator=generator)
        count = max(1, int(round(float(valid_indices.numel()) * float(ratio.item()))))
        order = torch.randperm(valid_indices.numel(), device=device, generator=generator)[:count]
        chosen = valid_indices[order]
        flat_masked[b, chosen] = mask_token_id
        flat_labels[b, chosen] = flat_tokens[b, chosen]
    return masked, labels


def maskgit_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]), labels.reshape(-1), ignore_index=-100)


class MaskGIT(nn.Module):
    def __init__(self, config: MaskGITConfig | None = None):
        super().__init__()
        self.config = config or MaskGITConfig()
        if self.config.hidden_dim % self.config.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.token_embed = nn.Embedding(self.config.vocab_size + 2, self.config.hidden_dim)
        self.structure_cond_embed = (
            nn.Embedding((self.config.structure_vocab_size or self.config.vocab_size) + 2, self.config.hidden_dim)
            if self.config.structure_cond
            else None
        )
        self.row_embed = nn.Embedding(self.config.max_height, self.config.hidden_dim)
        self.col_embed = nn.Embedding(self.config.max_width, self.config.hidden_dim)
        self.cond_proj = (
            nn.Linear(self.config.text_dim, self.config.cond_tokens * self.config.hidden_dim)
            if self.config.text_dim > 0
            else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.heads,
            dim_feedforward=self.config.hidden_dim * self.config.ff_mult,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.config.depth)
        self.norm = nn.LayerNorm(self.config.hidden_dim)
        self.head = nn.Linear(self.config.hidden_dim, self.config.vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
        structure_condition: torch.Tensor | None = None,
        cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        if tokens.shape != valid_mask.shape:
            raise ValueError("tokens and valid_mask must have the same shape")
        batch, height, width = tokens.shape
        if height > self.config.max_height or width > self.config.max_width:
            raise ValueError("token grid exceeds configured positional embeddings")

        x_tokens = tokens.masked_fill(~valid_mask, self.config.pad_token_id)
        rows = torch.arange(height, device=tokens.device)
        cols = torch.arange(width, device=tokens.device)
        x = self.token_embed(x_tokens)
        if self.structure_cond_embed is not None:
            structure_pad_token_id = self.structure_cond_embed.num_embeddings - 1
            if structure_condition is None:
                structure_condition = torch.full_like(tokens, structure_pad_token_id)
            if structure_condition.shape != tokens.shape:
                raise ValueError("structure_condition must match tokens shape")
            cond_tokens = structure_condition.masked_fill(~valid_mask, structure_pad_token_id)
            x = x + self.structure_cond_embed(cond_tokens)
        x = x + self.row_embed(rows)[None, :, None, :] + self.col_embed(cols)[None, None, :, :]
        x = x.reshape(batch, height * width, self.config.hidden_dim)
        padding = ~valid_mask.reshape(batch, height * width)
        if self.cond_proj is not None:
            if text_embeddings is None:
                text_embeddings = torch.zeros((batch, self.config.text_dim), device=tokens.device)
            text_embeddings = text_embeddings.to(tokens.device)
            if cond_drop_prob > 0:
                keep = torch.rand((batch, 1), device=tokens.device).ge(cond_drop_prob)
                text_embeddings = text_embeddings * keep
            cond = self.cond_proj(text_embeddings).reshape(batch, self.config.cond_tokens, self.config.hidden_dim)
            x = torch.cat([cond, x], dim=1)
            padding = torch.cat([torch.zeros((batch, self.config.cond_tokens), dtype=torch.bool, device=tokens.device), padding], dim=1)
        x = self.transformer(x, src_key_padding_mask=padding)
        x = x[:, -height * width :]
        logits = self.head(self.norm(x)).reshape(batch, height, width, self.config.vocab_size)
        return logits.permute(0, 3, 1, 2)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        valid_mask: torch.Tensor | None = None,
        steps: int = 8,
        temperature: float = 1.0,
        text_embeddings: torch.Tensor | None = None,
        structure_condition: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch, height, width = shape
        device = next(self.parameters()).device
        valid = torch.ones(shape, dtype=torch.bool, device=device) if valid_mask is None else valid_mask.to(device)
        tokens = torch.full(shape, self.config.mask_token_id, dtype=torch.long, device=device)
        tokens = tokens.masked_fill(~valid, self.config.pad_token_id)

        for step in range(max(1, steps)):
            still_masked = tokens.eq(self.config.mask_token_id) & valid
            if not still_masked.any():
                break
            logits = self(tokens, valid, text_embeddings, structure_condition)
            if text_embeddings is not None and guidance_scale != 1.0:
                uncond = self(tokens, valid, None, structure_condition)
                logits = uncond + (logits - uncond) * guidance_scale
            probs = (logits.clamp(-50, 50) / max(temperature, 1e-6)).softmax(dim=1)
            sampled = torch.multinomial(probs.permute(0, 2, 3, 1).reshape(-1, self.config.vocab_size), 1, generator=generator)
            sampled = sampled.reshape(batch, height, width)
            confidence = probs.gather(1, sampled[:, None]).squeeze(1).masked_fill(~still_masked, -1)
            remaining_steps = max(1, steps - step)
            flat_mask = still_masked.flatten(1)
            flat_conf = confidence.flatten(1)
            flat_tokens = tokens.flatten(1)
            flat_sampled = sampled.flatten(1)
            for b in range(batch):
                count = int(flat_mask[b].sum().item())
                take = max(1, (count + remaining_steps - 1) // remaining_steps)
                chosen = flat_conf[b].topk(take).indices
                flat_tokens[b, chosen] = flat_sampled[b, chosen]
        return tokens.masked_fill(~valid, self.config.pad_token_id)

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "MaskGIT":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(MaskGITConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model
