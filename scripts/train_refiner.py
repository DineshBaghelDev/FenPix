from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenpix.color import IndexedColorModel, reconstruct_indexed_png
from fenpix.dataset import BucketBatchSampler, PixelArtDataset, pixel_art_collate
from fenpix.hierarchy import HierarchicalMaskGIT, condition_to_shape, stage_tokens_from_batch
from fenpix.refiner import FlowRefinerConfig, PaletteLogitFlowRefiner, compare_refinement
from fenpix.text import FrozenPretrainedTextEncoder, TextEmbeddingCache, TextEncoderConfig
from fenpix.tokenizer import StructureTokenizer
from fenpix.training import append_jsonl, load_training_checkpoint, save_training_checkpoint, set_deterministic


def _captions(batch) -> list[str]:
    out = []
    for meta in batch["metadata"]:
        caption = meta.get("caption") or meta.get("category") or "pixel art"
        category = meta.get("category")
        out.append(f"{caption} {category}".strip() if category and category not in str(caption) else str(caption))
    return out


def _dataset(args: argparse.Namespace):
    dataset = PixelArtDataset(args.data, max_colors=args.max_colors, cache=args.cache)
    keep = [i for i in range(len(dataset)) if int(dataset[i]["bucket_size"]) <= args.max_size]
    if args.limit:
        keep = keep[: args.limit]
    if not keep:
        raise ValueError(f"no <={args.max_size} PNGs found")
    return Subset(dataset, keep)


def _structure_for_batch(batch, tokenizer: StructureTokenizer, stage: int, device: torch.device) -> torch.Tensor:
    tokens, token_valid = stage_tokens_from_batch(batch, tokenizer, stage, device)
    return condition_to_shape(tokens, token_valid, batch["indices"].shape[-2:], tokenizer.config.codebook_size + 1)


def _base_logits(color: IndexedColorModel, batch, structure, text, device: torch.device, guidance_scale: float) -> torch.Tensor:
    tokens = torch.full_like(batch["indices"].to(device), color.index_model.config.mask_token_id)
    tokens = tokens.masked_fill(~batch["valid_mask"].to(device), color.index_model.config.pad_token_id)
    return color.index_logits(
        tokens,
        batch["valid_mask"].to(device),
        structure,
        text,
        batch["palette"].to(device),
        batch["palette_mask"].to(device),
        guidance_scale,
    ).detach()


def _save_compare_viz(batch, palette, palette_mask, metrics, path: Path, max_items: int = 4) -> None:
    rows = []
    step_keys = sorted(metrics)
    for b in range(min(max_items, batch["indices"].shape[0])):
        h, w = batch["valid_mask"][b].shape
        panels = [np.asarray(reconstruct_indexed_png(batch["indices"][b, :h, :w].masked_fill(~batch["valid_mask"][b], 0), palette[b][palette_mask[b]]))]
        for steps in step_keys:
            pred = metrics[steps]["indices"][b, :h, :w].detach().cpu().masked_fill(~batch["valid_mask"][b], 0)
            panels.append(np.asarray(reconstruct_indexed_png(pred, palette[b][palette_mask[b]])))
        rows.append(np.concatenate(panels, axis=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0).astype(np.uint8), "RGBA").save(path)


def train(args: argparse.Namespace) -> dict[str, float]:
    set_deterministic(args.seed)
    device = torch.device(args.device)
    tokenizer = StructureTokenizer.load_checkpoint(args.tokenizer, map_location=device).to(device).eval()
    color = IndexedColorModel.load_checkpoint(args.color_checkpoint, map_location=device).to(device).eval()
    text_encoder = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=color.config.text_dim, provider=args.text_provider, device=args.device))
    text_cache = TextEmbeddingCache(args.embedding_cache, text_encoder) if args.embedding_cache else None
    dataset = _dataset(args)
    loader = DataLoader(dataset, batch_sampler=BucketBatchSampler(dataset, args.batch_size, seed=args.seed), collate_fn=pixel_art_collate)
    refiner = PaletteLogitFlowRefiner(
        FlowRefinerConfig(
            max_colors=color.config.max_colors,
            structure_vocab_size=tokenizer.config.codebook_size,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            text_dim=color.config.text_dim,
        )
    ).to(device)
    opt = torch.optim.AdamW(refiner.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_epoch = load_training_checkpoint(args.resume, refiner, opt, device) + 1 if args.resume else 1
    last: dict[str, float] = {}

    for epoch in range(start_epoch, args.epochs + 1):
        total = 0.0
        batches = 0
        refiner.train()
        opt.zero_grad(set_to_none=True)
        for batch in loader:
            text = (text_cache.encode(_captions(batch)) if text_cache else text_encoder.encode(_captions(batch))).to(device)
            structure = _structure_for_batch(batch, tokenizer, min(args.stage, args.max_size), device)
            base = _base_logits(color, batch, structure, text, device, args.guidance_scale)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss = refiner.loss(
                    base,
                    batch["indices"].to(device),
                    batch["valid_mask"].to(device),
                    structure,
                    text,
                    batch["palette"].to(device),
                    batch["palette_mask"].to(device),
                )
            scaler.scale(loss / args.grad_accum).backward()
            if (batches + 1) % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += float(loss.item())
            batches += 1
        if batches % args.grad_accum:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        last = {"epoch": float(epoch), "loss": total / max(batches, 1)}
        append_jsonl(args.log, last)
        print(json.dumps(last, sort_keys=True))

    eval_metrics = evaluate(args, refiner=refiner, color=color, tokenizer=tokenizer, loader=loader, text_encoder=text_encoder)
    if args.checkpoint:
        save_training_checkpoint(args.checkpoint, refiner, opt, args.epochs, {"train": last, "eval": eval_metrics})
    return last


@torch.no_grad()
def evaluate(args: argparse.Namespace, refiner=None, color=None, tokenizer=None, loader=None, text_encoder=None) -> dict[str, dict[str, float]]:
    device = torch.device(args.device)
    tokenizer = tokenizer or StructureTokenizer.load_checkpoint(args.tokenizer, map_location=device).to(device).eval()
    color = color or IndexedColorModel.load_checkpoint(args.color_checkpoint, map_location=device).to(device).eval()
    refiner = refiner or PaletteLogitFlowRefiner.load_checkpoint(args.checkpoint, map_location=device).to(device).eval()
    text_encoder = text_encoder or FrozenPretrainedTextEncoder(TextEncoderConfig(dim=color.config.text_dim, provider=args.text_provider, device=args.device))
    if loader is None:
        dataset = _dataset(args)
        loader = DataLoader(dataset, batch_sampler=BucketBatchSampler(dataset, args.batch_size, seed=args.seed), collate_fn=pixel_art_collate)

    totals: dict[int, dict[str, float]] = {}
    counts = 0
    first_viz = None
    for batch in loader:
        text = text_encoder.encode(_captions(batch)).to(device)
        structure = _structure_for_batch(batch, tokenizer, min(args.stage, args.max_size), device)
        base = _base_logits(color, batch, structure, text, device, args.guidance_scale)
        metrics = compare_refinement(
            refiner,
            base,
            batch["indices"].to(device),
            batch["valid_mask"].to(device),
            structure,
            text,
            batch["palette"].to(device),
            batch["palette_mask"].to(device),
            steps=tuple(args.refine_steps),
        )
        if first_viz is None:
            first_viz = (batch, metrics)
        for steps, row in metrics.items():
            acc = totals.setdefault(steps, {k: 0.0 for k in ("index_accuracy", "edge_detail", "palette_consistency", "transparency", "latency_ms")})
            for key in acc:
                acc[key] += float(row[key])
        counts += 1

    summary = {str(steps): {key: value / max(counts, 1) for key, value in row.items()} for steps, row in totals.items()}
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.viz and first_viz is not None:
        batch, metrics = first_viz
        _save_compare_viz(batch, batch["palette"], batch["palette_mask"], metrics, args.viz)
    print(json.dumps(summary, sort_keys=True))
    return summary


@torch.no_grad()
def sample(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    color = IndexedColorModel.load_checkpoint(args.color_checkpoint, map_location=device).to(device).eval()
    refiner = PaletteLogitFlowRefiner.load_checkpoint(args.checkpoint, map_location=device).to(device).eval()
    prompts = args.prompts or ["pixel art"]
    prompts = (prompts * ((args.samples + len(prompts) - 1) // len(prompts)))[: args.samples]
    text = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=color.config.text_dim, provider=args.text_provider, device=args.device)).encode(prompts).to(device) if color.config.text_dim else None

    if args.hierarchy:
        hierarchy = HierarchicalMaskGIT.load_checkpoint(args.hierarchy, map_location=device).to(device).eval()
        stages = hierarchy.sample(args.width, args.height, prompts, text, args.structure_steps, args.temperature, args.guidance_scale)
        tokens, valid = stages[max(stages)]
        structure = condition_to_shape(tokens, valid, (args.height, args.width), hierarchy.config.vocab_size + 1)
    else:
        structure = torch.zeros((args.samples, args.height, args.width), dtype=torch.long, device=device)
    valid = torch.ones_like(structure, dtype=torch.bool)
    pred = color.predict_palette(structure, valid, text)
    base_indices = color._sample_indices(structure, valid, color._index_text(text, pred["palette"].float(), pred["palette_mask"]), pred["palette_mask"], args.steps, args.temperature, args.guidance_scale)
    base_logits = color.index_logits(base_indices, valid, structure, text, pred["palette"].float(), pred["palette_mask"], args.guidance_scale)
    refined = refiner.refine(base_logits, args.refine_steps, valid, structure, text, pred["palette"].float(), pred["palette_mask"])
    indices = refined.argmax(dim=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    image = reconstruct_indexed_png(indices[0, : args.height, : args.width], pred["palette"][0][pred["palette_mask"][0]])
    image.save(args.out)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p):
        p.add_argument("--color-checkpoint", type=Path, required=True)
        p.add_argument("--checkpoint", type=Path, default=Path("runs/m7_refiner.pt"))
        p.add_argument("--device", default="cpu")
        p.add_argument("--guidance-scale", type=float, default=2.0)
        p.add_argument("--text-provider", choices=["clip", "tiny"], default="clip")

    train_parser = sub.add_parser("train")
    train_parser.add_argument("data", type=Path)
    train_parser.add_argument("--tokenizer", type=Path, required=True)
    train_parser.add_argument("--viz", type=Path, default=Path("runs/m7_refiner_compare.png"))
    train_parser.add_argument("--metrics", type=Path, default=Path("runs/m7_refiner_metrics.json"))
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--limit", type=int, default=32)
    train_parser.add_argument("--max-colors", type=int, default=64)
    train_parser.add_argument("--max-size", type=int, default=128)
    train_parser.add_argument("--stage", type=int, default=128)
    train_parser.add_argument("--hidden-dim", type=int, default=64)
    train_parser.add_argument("--depth", type=int, default=2)
    train_parser.add_argument("--embedding-cache", type=Path, default=Path("runs/m7_refiner_text_cache.pt"))
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--refine-steps", type=int, nargs="+", default=[0, 1, 2, 4])
    train_parser.add_argument("--cache", action="store_true")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--log", type=Path, default=Path("runs/m7_refiner_train.jsonl"))
    train_parser.add_argument("--amp", action="store_true")
    train_parser.add_argument("--grad-accum", type=int, default=1)
    shared(train_parser)
    train_parser.set_defaults(func=train)

    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("data", type=Path)
    eval_parser.add_argument("--tokenizer", type=Path, required=True)
    eval_parser.add_argument("--viz", type=Path, default=Path("runs/m7_refiner_compare.png"))
    eval_parser.add_argument("--metrics", type=Path, default=Path("runs/m7_refiner_metrics.json"))
    eval_parser.add_argument("--batch-size", type=int, default=4)
    eval_parser.add_argument("--limit", type=int, default=32)
    eval_parser.add_argument("--max-colors", type=int, default=64)
    eval_parser.add_argument("--max-size", type=int, default=128)
    eval_parser.add_argument("--stage", type=int, default=128)
    eval_parser.add_argument("--refine-steps", type=int, nargs="+", default=[0, 1, 2, 4])
    eval_parser.add_argument("--cache", action="store_true")
    eval_parser.add_argument("--seed", type=int, default=0)
    shared(eval_parser)
    eval_parser.set_defaults(func=evaluate)

    sample_parser = sub.add_parser("sample")
    sample_parser.add_argument("--hierarchy", type=Path)
    sample_parser.add_argument("--out", type=Path, default=Path("runs/m7_refined_sample.png"))
    sample_parser.add_argument("--samples", type=int, default=1)
    sample_parser.add_argument("--width", type=int, default=128)
    sample_parser.add_argument("--height", type=int, default=128)
    sample_parser.add_argument("--steps", type=int, default=8)
    sample_parser.add_argument("--refine-steps", type=int, default=2)
    sample_parser.add_argument("--structure-steps", type=int, default=8)
    sample_parser.add_argument("--temperature", type=float, default=1.0)
    sample_parser.add_argument("--prompts", nargs="*")
    shared(sample_parser)
    sample_parser.set_defaults(func=sample)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
