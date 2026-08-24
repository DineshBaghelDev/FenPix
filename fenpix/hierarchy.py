from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .maskgit import MaskGIT, MaskGITConfig, maskgit_loss, random_mask_tokens
from .tokenizer import StructureTokenizer, canonical_structure_indices, structure_one_hot


STAGE_SIZES = (32, 64, 128)


@dataclass(frozen=True)
class HierarchicalMaskGITConfig:
    vocab_size: int = 128
    hidden_dim: int = 64
    depth: int = 2
    heads: int = 4
    text_dim: int = 64
    cond_tokens: int = 1
    downsample: int = 4
    stages: tuple[int, ...] = STAGE_SIZES


def _ceil_multiple(value: int, multiple: int) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def stage_native_shape(width: int, height: int, stage_size: int, downsample: int = 4) -> tuple[int, int]:
    scale = min(1.0, stage_size / max(width, height))
    out_w = min(stage_size, _ceil_multiple(max(1, round(width * scale)), downsample))
    out_h = min(stage_size, _ceil_multiple(max(1, round(height * scale)), downsample))
    return out_h, out_w


def condition_to_shape(tokens: torch.Tensor, valid: torch.Tensor, shape: tuple[int, int], pad_token_id: int) -> torch.Tensor:
    cond = F.interpolate(tokens[:, None].float(), size=shape, mode="nearest").squeeze(1).long()
    cond_valid = F.interpolate(valid[:, None].float(), size=shape, mode="nearest").squeeze(1).bool()
    return cond.masked_fill(~cond_valid, pad_token_id)


@torch.no_grad()
def stage_structure_from_batch(
    batch: dict[str, Any],
    stage_size: int,
    downsample: int = 4,
    device: torch.device | str = "cpu",
    connected_components: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    indices = batch["structure_indices"] if connected_components else canonical_structure_indices(
        batch["indices"],
        batch["palette"],
        batch["valid_mask"],
        connected_components=False,
    )
    valid = batch["valid_mask"]
    targets = indices.to(device)
    valid = valid.to(device)
    shapes = [stage_native_shape(int(size[0]), int(size[1]), stage_size, downsample) for size in batch["size"]]
    max_h = max(shape[0] for shape in shapes)
    max_w = max(shape[1] for shape in shapes)
    stage_targets = torch.zeros((len(shapes), max_h, max_w), dtype=torch.long, device=device)
    stage_valid = torch.zeros((len(shapes), max_h, max_w), dtype=torch.bool, device=device)

    for row, (height, width) in enumerate(shapes):
        src_h = int(batch["size"][row][1])
        src_w = int(batch["size"][row][0])
        crop = targets[row : row + 1, None, :src_h, :src_w].float()
        crop_valid = valid[row : row + 1, None, :src_h, :src_w].float()
        stage_targets[row, :height, :width] = F.interpolate(crop, size=(height, width), mode="nearest").squeeze().long()
        stage_valid[row, :height, :width] = F.interpolate(crop_valid, size=(height, width), mode="nearest").squeeze().bool()

    return stage_targets, stage_valid


@torch.no_grad()
def stage_tokens_from_batch(
    batch: dict[str, Any],
    tokenizer: StructureTokenizer,
    stage_size: int,
    device: torch.device | str = "cpu",
    connected_components: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    stage_targets, stage_valid = stage_structure_from_batch(batch, stage_size, tokenizer.config.downsample, device, connected_components)
    stage_targets = stage_targets.clamp_max(tokenizer.config.num_structure_classes - 1)
    out = tokenizer(structure_one_hot(stage_targets, stage_valid, tokenizer.config.num_structure_classes))
    token_valid = F.interpolate(stage_valid[:, None].float(), size=out["codes"].shape[-2:], mode="nearest").squeeze(1).bool()
    return out["codes"], token_valid


def structure_region_weights(targets: torch.Tensor, valid: torch.Tensor, shape: tuple[int, int], foreground_weight: float, boundary_weight: float) -> torch.Tensor:
    foreground = targets.ne(0) & valid
    boundary = torch.zeros_like(foreground)
    boundary[:, 1:] |= foreground[:, 1:] != foreground[:, :-1]
    boundary[:, :-1] |= foreground[:, 1:] != foreground[:, :-1]
    boundary[:, :, 1:] |= foreground[:, :, 1:] != foreground[:, :, :-1]
    boundary[:, :, :-1] |= foreground[:, :, 1:] != foreground[:, :, :-1]
    weights = torch.ones_like(targets, dtype=torch.float32) + foreground.float() * foreground_weight + boundary.float() * boundary_weight
    weights = F.interpolate(weights[:, None], size=shape, mode="nearest").squeeze(1)
    valid_small = F.interpolate(valid[:, None].float(), size=shape, mode="nearest").squeeze(1).bool()
    return weights.masked_fill(~valid_small, 0.0)


def foreground_decode_loss(
    code_logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    tokenizer: StructureTokenizer,
    weights: torch.Tensor,
) -> torch.Tensor:
    probs = code_logits.softmax(dim=1).permute(0, 2, 3, 1)
    latent = torch.matmul(probs, tokenizer.quantizer.embedding.weight.to(dtype=probs.dtype)).permute(0, 3, 1, 2)
    decoded = tokenizer.decoder(latent)
    if decoded.shape[-2:] != targets.shape[-2:]:
        decoded = F.interpolate(decoded, size=targets.shape[-2:], mode="nearest")
    fg_logits = torch.logsumexp(decoded[:, 1:], dim=1) - decoded[:, 0]
    loss = F.binary_cross_entropy_with_logits(fg_logits, targets.ne(0).float(), reduction="none")
    return (loss * weights * valid.float()).sum() / (weights * valid.float()).sum().clamp_min(1)


class HierarchicalMaskGIT(nn.Module):
    def __init__(self, config: HierarchicalMaskGITConfig | None = None):
        super().__init__()
        self.config = config or HierarchicalMaskGITConfig()
        self.stages = tuple(self.config.stages)
        self.models = nn.ModuleDict()
        for stage in self.stages:
            self.models[str(stage)] = MaskGIT(
                MaskGITConfig(
                    vocab_size=self.config.vocab_size,
                    hidden_dim=self.config.hidden_dim,
                    depth=self.config.depth,
                    heads=self.config.heads,
                    max_height=stage // self.config.downsample,
                    max_width=stage // self.config.downsample,
                    text_dim=self.config.text_dim,
                    cond_tokens=self.config.cond_tokens,
                    structure_cond=stage != self.stages[0],
                )
            )

    def stage_loss(
        self,
        stage: int,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
        lower_tokens: torch.Tensor | None = None,
        lower_valid: torch.Tensor | None = None,
        cond_drop_prob: float = 0.0,
        target_structure: torch.Tensor | None = None,
        target_structure_valid: torch.Tensor | None = None,
        tokenizer: StructureTokenizer | None = None,
        foreground_weight: float = 0.0,
        boundary_weight: float = 0.0,
        foreground_loss_weight: float = 0.0,
    ) -> torch.Tensor:
        model = self.models[str(stage)]
        masked, labels = random_mask_tokens(tokens, valid, model.config.mask_token_id)
        cond = None
        if lower_tokens is not None and lower_valid is not None:
            cond = condition_to_shape(lower_tokens, lower_valid, tokens.shape[-2:], model.config.pad_token_id)
        logits = model(masked, valid, text_embeddings, cond, cond_drop_prob)
        token_weights = None
        pixel_weights = None
        if target_structure is not None and target_structure_valid is not None:
            token_weights = structure_region_weights(target_structure, target_structure_valid, tokens.shape[-2:], foreground_weight, boundary_weight)
            pixel_weights = structure_region_weights(target_structure, target_structure_valid, target_structure.shape[-2:], foreground_weight, boundary_weight)
        loss = maskgit_loss(logits, labels, token_weights)
        if foreground_loss_weight > 0 and tokenizer is not None and target_structure is not None and target_structure_valid is not None and pixel_weights is not None:
            loss = loss + foreground_decode_loss(logits, target_structure.clamp_max(tokenizer.config.num_structure_classes - 1), target_structure_valid, tokenizer, pixel_weights) * foreground_loss_weight
        return loss

    @torch.no_grad()
    def sample(
        self,
        width: int,
        height: int,
        prompts: list[str],
        text_embeddings: torch.Tensor | None = None,
        steps: int = 8,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        batch = len(prompts)
        device = next(self.parameters()).device
        previous: tuple[torch.Tensor, torch.Tensor] | None = None
        out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for stage in self.stages:
            native_h, native_w = stage_native_shape(width, height, stage, self.config.downsample)
            shape = (batch, native_h // self.config.downsample, native_w // self.config.downsample)
            valid = torch.ones(shape, dtype=torch.bool, device=device)
            model = self.models[str(stage)]
            cond = condition_to_shape(previous[0], previous[1], shape[-2:], model.config.pad_token_id) if previous else None
            tokens = model.sample(
                shape,
                valid,
                steps=steps,
                temperature=temperature,
                text_embeddings=text_embeddings,
                structure_condition=cond,
                guidance_scale=guidance_scale,
            )
            previous = (tokens, valid)
            out[stage] = previous
        return out

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "HierarchicalMaskGIT":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = checkpoint["config"]
        config["stages"] = tuple(config["stages"])
        model = cls(HierarchicalMaskGITConfig(**config))
        model.load_state_dict(checkpoint["state_dict"])
        return model
