from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenpix.color import IndexedColorModel
from fenpix.dataset import BucketBatchSampler, PixelArtDataset, filtered_indices, pixel_art_collate, split_report, train_val_test_split
from fenpix.evaluation import compute_quality_metrics, render_indexed_batch, save_comparison_gallery, save_metrics, timed
from fenpix.hierarchy import HierarchicalMaskGIT, condition_to_shape, stage_tokens_from_batch
from fenpix.text import FrozenPretrainedTextEncoder, FrozenVisionLanguageEncoder, TextEncoderConfig
from fenpix.tokenizer import StructureTokenizer


def _captions(batch) -> list[str]:
    out = []
    for meta, path in zip(batch["metadata"], batch["path"]):
        caption = meta.get("caption") or meta.get("category") or Path(path).stem
        category = meta.get("category")
        out.append(f"{caption} {category}".strip() if category and category not in str(caption) else str(caption))
    return out


def _test_split(args: argparse.Namespace):
    dataset = PixelArtDataset(args.data, max_colors=args.max_colors, cache=args.cache)
    keep = filtered_indices(dataset, max_bucket_size=args.max_size, include_lossy=args.include_lossy, limit=args.limit)
    train, validation, test = train_val_test_split(Subset(dataset, keep), args.validation_fraction, args.test_fraction, args.seed)
    return train, validation, test


def _heldout_structure(batch, tokenizer, args, device):
    tokens, valid = stage_tokens_from_batch(batch, tokenizer, min(args.stage, args.max_size), device)
    return condition_to_shape(tokens, valid, batch["indices"].shape[-2:], tokenizer.config.codebook_size + 1)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--color-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path)
    parser.add_argument("--use-heldout-structure", action="store_true")
    parser.add_argument("--metrics", type=Path, default=Path("runs/m8_1_metrics.json"))
    parser.add_argument("--gallery", type=Path, default=Path("runs/m8_1_gallery.png"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--include-lossy", action="store_true")
    parser.add_argument("--max-colors", type=int, default=64)
    parser.add_argument("--max-size", type=int, default=128)
    parser.add_argument("--stage", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--structure-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--text-provider", choices=["clip", "tiny"], default="clip")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.hierarchy and not args.use_heldout_structure:
        raise ValueError("pass --hierarchy or explicitly pass --use-heldout-structure")
    if args.text_provider != "clip":
        raise ValueError("M8.1 held-out evaluation requires --text-provider clip")

    device = torch.device(args.device)
    train, validation, test = _test_split(args)
    loader = DataLoader(test, batch_sampler=BucketBatchSampler(test, args.batch_size, seed=args.seed), collate_fn=pixel_art_collate)
    tokenizer = StructureTokenizer.load_checkpoint(args.tokenizer, map_location=device).to(device).eval()
    color = IndexedColorModel.load_checkpoint(args.color_checkpoint, map_location=device).to(device).eval()
    hierarchy = HierarchicalMaskGIT.load_checkpoint(args.hierarchy, map_location=device).to(device).eval() if args.hierarchy else None
    text_encoder = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=color.config.text_dim, provider="clip", device=args.device))
    vlm = FrozenVisionLanguageEncoder(TextEncoderConfig(provider="clip", device=args.device))

    targets = []
    generated = []
    prompts = []
    total_latency = 0.0
    for batch in loader:
        batch_prompts = _captions(batch)
        text = text_encoder.encode(batch_prompts).to(device)

        def generate():
            if hierarchy is not None:
                width = int(batch["size"][0][0])
                height = int(batch["size"][0][1])
                stages = hierarchy.sample(width, height, batch_prompts, text, args.structure_steps, args.temperature, args.guidance_scale)
                tokens, valid = stages[max(stages)]
                structure = condition_to_shape(tokens, valid, batch["indices"].shape[-2:], hierarchy.config.vocab_size + 1)
            else:
                structure = _heldout_structure(batch, tokenizer, args, device)
            return color.sample(structure, batch["valid_mask"].to(device), text, args.steps, args.temperature, args.guidance_scale)

        out, latency = timed(generate)
        total_latency += latency
        generated.extend(render_indexed_batch(out["indices"].cpu(), out["palette"].cpu(), out["palette_mask"].cpu(), batch["valid_mask"]))
        targets.extend(render_indexed_batch(batch["indices"], batch["palette"], batch["palette_mask"], batch["valid_mask"]))
        prompts.extend(batch_prompts)

    metrics = compute_quality_metrics(generated, targets, prompts, encoder=vlm, latency_ms=total_latency / max(len(generated), 1)).__dict__
    report = {"metrics": metrics, "split": split_report(train, validation, test), "used_heldout_structure": bool(args.use_heldout_structure and hierarchy is None)}
    save_metrics(report, args.metrics)
    save_comparison_gallery(targets, generated, args.gallery, prompts)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
