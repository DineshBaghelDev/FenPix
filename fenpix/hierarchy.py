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
def stage_tokens_from_batch(
    batch: dict[str, Any],
    tokenizer: StructureTokenizer,
    stage_size: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    indices = batch["structure_indices"].to(device)
    palette = batch["palette"].to(device)
    valid = batch["valid_mask"].to(device)
    targets = canonical_structure_indices(indices, palette, valid, max_regions=tokenizer.config.num_structure_classes - 1)
    shapes = [stage_native_shape(int(size[0]), int(size[1]), stage_size, tokenizer.config.downsample) for size in batch["size"]]
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

    out = tokenizer(structure_one_hot(stage_targets, stage_valid, tokenizer.config.num_structure_classes))
    token_valid = F.interpolate(stage_valid[:, None].float(), size=out["codes"].shape[-2:], mode="nearest").squeeze(1).bool()
    return out["codes"], token_valid


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
    ) -> torch.Tensor:
        model = self.models[str(stage)]
        masked, labels = random_mask_tokens(tokens, valid, model.config.mask_token_id)
        cond = None
        if lower_tokens is not None and lower_valid is not None:
            cond = condition_to_shape(lower_tokens, lower_valid, tokens.shape[-2:], model.config.pad_token_id)
        return maskgit_loss(model(masked, valid, text_embeddings, cond, cond_drop_prob), labels)

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
