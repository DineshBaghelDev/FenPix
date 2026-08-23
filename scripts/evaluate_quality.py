from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenpix.dataset import PixelArtDataset, dataset_quality_report, pixel_art_collate
from fenpix.evaluation import PROMPT_EVAL_SET, compare_color_and_refiner, save_comparison_gallery, save_metrics
from fenpix.refiner import FlowRefinerConfig, PaletteLogitFlowRefiner
from fenpix.text import FrozenPretrainedTextEncoder, TextEncoderConfig


def _base_logits(indices: torch.Tensor, max_colors: int, noise: float) -> torch.Tensor:
    logits = F.one_hot(indices.clamp_min(0), max_colors).permute(0, 3, 1, 2).float() * 5
    return logits + torch.randn_like(logits) * noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--metrics", type=Path, default=Path("runs/m8_metrics.json"))
    parser.add_argument("--gallery", type=Path, default=Path("runs/m8_gallery.png"))
    parser.add_argument("--refiner", type=Path)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-colors", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = PixelArtDataset(args.data, max_colors=args.max_colors)
    subset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
    batch = next(iter(DataLoader(subset, batch_size=len(subset), collate_fn=pixel_art_collate)))
    prompts = list(PROMPT_EVAL_SET[: len(subset)])
    metadata = [meta | {"path": path} for meta, path in zip(batch["metadata"], batch["path"])]
    text = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=64)).encode(prompts)

    indices = batch["indices"]
    valid = batch["valid_mask"]
    structure = batch["structure_indices"]
    base = _base_logits(indices, args.max_colors, args.noise)
    refiner = (
        PaletteLogitFlowRefiner.load_checkpoint(args.refiner)
        if args.refiner
        else PaletteLogitFlowRefiner(FlowRefinerConfig(max_colors=args.max_colors, text_dim=64))
    )
    metrics = compare_color_and_refiner(
        base,
        indices,
        valid,
        structure,
        text,
        batch["palette"],
        batch["palette_mask"],
        prompts,
        metadata,
        refiner,
    ) | {"dataset": dataset_quality_report(subset)}

    outputs = {"target": indices, "baseline_color": base.argmax(1).masked_fill(~valid, -1)}
    for steps in (1, 2, 4):
        outputs[f"flow_{steps}"] = refiner.refine(base, steps, valid, structure, text, batch["palette"], batch["palette_mask"]).argmax(1).masked_fill(~valid, -1)
    save_metrics(metrics, args.metrics)
    save_comparison_gallery(outputs, batch["palette"], batch["palette_mask"], args.gallery, prompts)
    print(f"wrote {args.metrics} and {args.gallery}")


if __name__ == "__main__":
    main()
