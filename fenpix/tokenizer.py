from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StructureTokenizerConfig:
    num_structure_classes: int = 65
    codebook_size: int = 128
    latent_dim: int = 128
    hidden_dim: int = 128
    downsample: int = 4
    commitment_cost: float = 0.25


def canonical_structure_indices(
    indices: torch.Tensor,
    palette: torch.Tensor,
    valid_mask: torch.Tensor,
    max_regions: int = 64,
) -> torch.Tensor:
    """Map image-local palette IDs to local region IDs; RGB values are ignored."""
    if indices.ndim != 3:
        raise ValueError("indices must be [batch,height,width]")
    out = torch.zeros_like(indices)
    for b in range(indices.shape[0]):
        valid = valid_mask[b] & (indices[b] >= 0)
        if not valid.any():
            continue
        alpha = palette[b, :, 3] if palette.ndim == 3 else torch.empty(0, device=indices.device)
        transparent = torch.zeros_like(valid)
        in_palette = valid & (indices[b] < len(alpha))
        if len(alpha):
            transparent[in_palette] = alpha[indices[b][in_palette]].eq(0)
        visible_values = indices[b][valid & ~transparent]
        seen: list[int] = []
        for value in visible_values.flatten().tolist():
            if value not in seen:
                seen.append(value)
            if len(seen) >= max_regions:
                break
        for region_id, value in enumerate(seen, start=1):
            out[b][valid & ~transparent & indices[b].eq(value)] = region_id
    return out.clamp_max(max_regions)


def structure_one_hot(targets: torch.Tensor, valid_mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    targets = targets.clamp(0, num_classes - 1)
    x = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    return x * valid_mask[:, None].float()


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits, targets.clamp_min(0), reduction="none")
    mask = valid_mask.float()
    return (loss * mask).sum() / mask.sum().clamp_min(1)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, latent_dim: int, commitment_cost: float):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.embedding.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)
        self.commitment_cost = commitment_cost

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        channels = z.shape[1]
        flat = z.permute(0, 2, 3, 1).reshape(-1, channels)
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.square().sum(dim=1)
        )
        codes = distances.argmin(dim=1)
        quantized = self.embedding(codes).view(z.shape[0], z.shape[2], z.shape[3], channels).permute(0, 3, 1, 2)
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        quantized = z + (quantized - z).detach()
        return quantized, codes.view(z.shape[0], z.shape[2], z.shape[3]), loss


class StructureTokenizer(nn.Module):
    def __init__(self, config: StructureTokenizerConfig | None = None):
        super().__init__()
        self.config = config or StructureTokenizerConfig()
        if self.config.downsample < 1 or self.config.downsample & (self.config.downsample - 1):
            raise ValueError("downsample must be a power of two")

        encoder: list[nn.Module] = [
            nn.Conv2d(self.config.num_structure_classes, self.config.hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        steps = self.config.downsample.bit_length() - 1
        for _ in range(steps):
            encoder += [
                nn.Conv2d(self.config.hidden_dim, self.config.hidden_dim, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
        encoder.append(nn.Conv2d(self.config.hidden_dim, self.config.latent_dim, 1))
        self.encoder = nn.Sequential(*encoder)
        self.quantizer = VectorQuantizer(
            self.config.codebook_size,
            self.config.latent_dim,
            self.config.commitment_cost,
        )

        decoder: list[nn.Module] = [nn.Conv2d(self.config.latent_dim, self.config.hidden_dim, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(steps):
            decoder += [
                nn.ConvTranspose2d(self.config.hidden_dim, self.config.hidden_dim, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
        decoder.append(nn.Conv2d(self.config.hidden_dim, self.config.num_structure_classes, 1))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        quantized, codes, vq_loss = self.quantizer(z)
        logits = self.decoder(quantized)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="nearest")
        return {"logits": logits, "codes": codes, "vq_loss": vq_loss}

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "StructureTokenizer":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(StructureTokenizerConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model


@torch.no_grad()
def tokenizer_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    codes: torch.Tensor,
    codebook_size: int,
) -> dict[str, float]:
    pred = logits.argmax(dim=1)
    mask = valid_mask.bool()
    accuracy = pred[mask].eq(targets[mask]).float().mean().item() if mask.any() else 0.0
    silhouette = pred.eq(0)[mask].eq(targets.eq(0)[mask]).float().mean().item() if mask.any() else 0.0
    counts = torch.bincount(codes.flatten(), minlength=codebook_size).float()
    probs = counts / counts.sum().clamp_min(1)
    used = int((counts > 0).sum().item())
    entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
    return {
        "accuracy": accuracy,
        "silhouette_accuracy": silhouette,
        "codes_used": float(used),
        "dead_codes": float(codebook_size - used),
        "perplexity": float(entropy.exp().item()),
    }
