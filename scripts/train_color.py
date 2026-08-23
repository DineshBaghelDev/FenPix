from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenpix.color import IndexedColorConfig, IndexedColorModel, reconstruct_indexed_png
from fenpix.dataset import BucketBatchSampler, PixelArtDataset, filtered_indices, pixel_art_collate, split_report, train_val_test_split
from fenpix.hierarchy import HierarchicalMaskGIT, condition_to_shape, stage_tokens_from_batch
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


def _splits(args: argparse.Namespace):
    dataset = PixelArtDataset(args.data, max_colors=args.max_colors, cache=args.cache)
    keep = filtered_indices(dataset, max_bucket_size=args.max_size, include_lossy=args.include_lossy, limit=args.limit)
    if not keep:
        raise ValueError(f"no <={args.max_size} PNGs found")
    return train_val_test_split(Subset(dataset, keep), args.validation_fraction, args.test_fraction, args.seed)


def _structure_for_batch(batch, tokenizer: StructureTokenizer, stage: int, device: torch.device) -> torch.Tensor:
    tokens, token_valid = stage_tokens_from_batch(batch, tokenizer, stage, device)
    return condition_to_shape(tokens, token_valid, batch["indices"].shape[-2:], tokenizer.config.codebook_size + 1)


def _structure_panel(tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    panel = (tokens.float() / max(1, int(tokens.max().item())) * 255).byte().masked_fill(~valid.cpu(), 255)
    return panel


def _palette_panel(palette: torch.Tensor, palette_mask: torch.Tensor, width: int, height: int) -> torch.Tensor:
    colors = palette[palette_mask].cpu().numpy().astype(np.uint8)
    if len(colors) == 0:
        colors = np.array([[0, 0, 0, 0]], dtype=np.uint8)
    swatch_w = max(1, width // len(colors))
    panel = np.zeros((height, width, 4), dtype=np.uint8)
    panel[..., 3] = 255
    for i, color in enumerate(colors):
        panel[:, i * swatch_w : (i + 1) * swatch_w] = color
    panel[:, len(colors) * swatch_w :] = colors[-1]
    return torch.from_numpy(panel)


def _save_viz(structure: torch.Tensor, valid: torch.Tensor, sample: dict[str, torch.Tensor], path: Path, max_items: int = 4) -> None:
    rows = []
    for b in range(min(max_items, structure.shape[0])):
        h, w = valid[b].shape
        struct = _structure_panel(structure[b].cpu(), valid[b]).numpy()
        struct = np.repeat(struct[:, :, None], 4, axis=2)
        struct[..., 3] = 255
        palette = _palette_panel(sample["palette"][b], sample["palette_mask"][b], w, max(4, h // 8)).numpy()
        indices = sample["indices"][b, :h, :w].detach().cpu().masked_fill(~valid[b].cpu(), 0)
        index_img = (indices.float() / max(1, int(indices.max().item())) * 255).byte().numpy()
        index_img = np.repeat(index_img[:, :, None], 4, axis=2)
        index_img[..., 3] = 255
        final = np.asarray(reconstruct_indexed_png(indices, sample["palette"][b][sample["palette_mask"][b]]))
        rows.append(np.concatenate([struct, index_img, final], axis=1))
        rows.append(np.pad(palette, ((0, 0), (0, w * 2), (0, 0)), constant_values=255))
    canvas = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas.astype(np.uint8), "RGBA").save(path)


def train(args: argparse.Namespace) -> dict[str, float]:
    set_deterministic(args.seed)
    device = torch.device(args.device)
    tokenizer = StructureTokenizer.load_checkpoint(args.tokenizer, map_location=device).to(device).eval()
    text_encoder = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=args.text_dim, provider=args.text_provider, device=args.device))
    text_cache = TextEmbeddingCache(args.embedding_cache, text_encoder) if args.embedding_cache else None
    train_set, validation_set, test_set = _splits(args)
    loader = DataLoader(train_set, batch_sampler=BucketBatchSampler(train_set, args.batch_size, seed=args.seed), collate_fn=pixel_art_collate)
    validation_loader = DataLoader(validation_set, batch_sampler=BucketBatchSampler(validation_set, args.batch_size, seed=args.seed), collate_fn=pixel_art_collate)
    config = IndexedColorConfig(
        max_colors=args.max_colors,
        min_colors=args.min_colors,
        structure_vocab_size=tokenizer.config.codebook_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        text_dim=args.text_dim,
        cond_tokens=args.cond_tokens,
        max_height=args.max_size,
        max_width=args.max_size,
    )
    model = IndexedColorModel(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_epoch = load_training_checkpoint(args.resume, model, opt, device) + 1 if args.resume else 1
    last = {}

    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        images = 0
        totals = {"loss": 0.0, "palette_loss": 0.0, "index_loss": 0.0}
        steps = 0
        model.train()
        opt.zero_grad(set_to_none=True)
        for batch in loader:
            texts = _captions(batch)
            text = (text_cache.encode(texts) if text_cache else text_encoder.encode(texts)).to(device)
            structure = _structure_for_batch(batch, tokenizer, min(args.stage, args.max_size), device)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                losses = model.loss(
                    batch["indices"].to(device),
                    batch["valid_mask"].to(device),
                    structure,
                    text,
                    batch["palette"].to(device),
                    batch["palette_mask"].to(device),
                    batch["palette_size"].to(device),
                    args.cond_dropout,
                )
            scaler.scale(losses["loss"] / args.grad_accum).backward()
            if (steps + 1) % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            for key in totals:
                totals[key] += float(losses[key].item())
            steps += 1
            images += int(batch["indices"].shape[0])
        if steps % args.grad_accum:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        seconds = perf_counter() - started
        val = _validation_loss(model, validation_loader, tokenizer, text_encoder, args, device)
        last = {
            "epoch": epoch,
            "split": split_report(train_set, validation_set, test_set),
            "seconds": seconds,
            "images_per_second": images / max(seconds, 1e-9),
            "peak_vram_mb": torch.cuda.max_memory_reserved(device) / 1024 / 1024 if device.type == "cuda" else 0.0,
            "validation_loss": val,
        } | {key: value / max(steps, 1) for key, value in totals.items()}
        append_jsonl(args.log, last)
        print(json.dumps(last, sort_keys=True))

    if args.checkpoint:
        save_training_checkpoint(args.checkpoint, model, opt, args.epochs, last)
    if args.viz:
        batch = next(iter(loader))
        text = text_encoder.encode(_captions(batch)).to(device)
        structure = _structure_for_batch(batch, tokenizer, min(args.stage, args.max_size), device)
        sample = model.eval().sample(structure, batch["valid_mask"].to(device), text, args.steps, args.temperature, args.guidance_scale)
        _save_viz(structure.cpu(), batch["valid_mask"], sample, args.viz)
    return last


@torch.no_grad()
def _validation_loss(model, loader, tokenizer, text_encoder, args, device: torch.device) -> float:
    model.eval()
    total = 0.0
    steps = 0
    for batch in loader:
        text = text_encoder.encode(_captions(batch)).to(device)
        structure = _structure_for_batch(batch, tokenizer, min(args.stage, args.max_size), device)
        losses = model.loss(
            batch["indices"].to(device),
            batch["valid_mask"].to(device),
            structure,
            text,
            batch["palette"].to(device),
            batch["palette_mask"].to(device),
            batch["palette_size"].to(device),
            0.0,
        )
        total += float(losses["loss"].item())
        steps += 1
    return total / max(steps, 1)


def sample(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    color = IndexedColorModel.load_checkpoint(args.checkpoint, map_location=device).to(device).eval()
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
    out = color.sample(structure, valid, text, args.steps, args.temperature, args.guidance_scale)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    image = reconstruct_indexed_png(out["indices"][0, : args.height, : args.width], out["palette"][0][out["palette_mask"][0]])
    image.save(args.out)
    if args.viz:
        _save_viz(structure.cpu(), valid.cpu(), out, args.viz, max_items=args.samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("data", type=Path)
    train_parser.add_argument("--tokenizer", type=Path, required=True)
    train_parser.add_argument("--checkpoint", type=Path, default=Path("runs/m7_color.pt"))
    train_parser.add_argument("--viz", type=Path, default=Path("runs/m7_color_viz.png"))
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--limit", type=int, default=32)
    train_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_parser.add_argument("--test-fraction", type=float, default=0.2)
    train_parser.add_argument("--include-lossy", action="store_true")
    train_parser.add_argument("--max-colors", type=int, default=64)
    train_parser.add_argument("--min-colors", type=int, default=8)
    train_parser.add_argument("--max-size", type=int, default=128)
    train_parser.add_argument("--stage", type=int, default=128)
    train_parser.add_argument("--hidden-dim", type=int, default=64)
    train_parser.add_argument("--depth", type=int, default=2)
    train_parser.add_argument("--heads", type=int, default=4)
    train_parser.add_argument("--text-dim", type=int, default=64)
    train_parser.add_argument("--text-provider", choices=["clip", "tiny"], default="clip")
    train_parser.add_argument("--cond-tokens", type=int, default=1)
    train_parser.add_argument("--cond-dropout", type=float, default=0.1)
    train_parser.add_argument("--embedding-cache", type=Path, default=Path("runs/m7_text_cache.pt"))
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--steps", type=int, default=8)
    train_parser.add_argument("--temperature", type=float, default=1.0)
    train_parser.add_argument("--guidance-scale", type=float, default=2.0)
    train_parser.add_argument("--cache", action="store_true")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--log", type=Path, default=Path("runs/m7_color_train.jsonl"))
    train_parser.add_argument("--amp", action="store_true")
    train_parser.add_argument("--grad-accum", type=int, default=1)
    train_parser.set_defaults(func=train)

    sample_parser = sub.add_parser("sample")
    sample_parser.add_argument("--checkpoint", type=Path, required=True)
    sample_parser.add_argument("--hierarchy", type=Path)
    sample_parser.add_argument("--out", type=Path, default=Path("runs/m7_sample.png"))
    sample_parser.add_argument("--viz", type=Path, default=Path("runs/m7_sample_viz.png"))
    sample_parser.add_argument("--device", default="cpu")
    sample_parser.add_argument("--samples", type=int, default=1)
    sample_parser.add_argument("--width", type=int, default=128)
    sample_parser.add_argument("--height", type=int, default=128)
    sample_parser.add_argument("--steps", type=int, default=8)
    sample_parser.add_argument("--structure-steps", type=int, default=8)
    sample_parser.add_argument("--temperature", type=float, default=1.0)
    sample_parser.add_argument("--guidance-scale", type=float, default=2.0)
    sample_parser.add_argument("--text-provider", choices=["clip", "tiny"], default="clip")
    sample_parser.add_argument("--prompts", nargs="*")
    sample_parser.set_defaults(func=sample)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
