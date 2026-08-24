from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenpix.dataset import BucketBatchSampler, PixelArtDataset, filtered_indices, pixel_art_collate, split_report, train_val_test_split
from fenpix.direct_structure import DirectStructureConfig, DirectStructureGenerator
from fenpix.text import FrozenPretrainedTextEncoder, TextEmbeddingCache, TextEncoderConfig
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


def _targets(batch, device: torch.device, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["structure_indices"].to(device).clamp(0, vocab_size - 1), batch["valid_mask"].to(device)


def _save_viz(tokens: torch.Tensor, valid: torch.Tensor, path: Path, max_items: int = 4) -> None:
    panels = []
    vmax = max(1, int(tokens.clamp_min(0).max().item()))
    for row in range(min(max_items, tokens.shape[0])):
        panel = (tokens[row].detach().cpu().clamp_min(0).float() / vmax * 255).byte()
        panels.append(panel.masked_fill(~valid[row].detach().cpu(), 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(torch.cat(panels, dim=0).numpy(), "L").save(path)


def train(args: argparse.Namespace) -> dict[str, float]:
    set_deterministic(args.seed)
    device = torch.device(args.device)
    text_encoder = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=args.text_dim, provider=args.text_provider, device=args.device))
    text_cache = TextEmbeddingCache(args.embedding_cache, text_encoder) if args.embedding_cache else None
    train_set, validation_set, test_set = _splits(args)
    loader = DataLoader(train_set, batch_sampler=BucketBatchSampler(train_set, args.batch_size, seed=args.seed), collate_fn=pixel_art_collate)
    validation_loader = DataLoader(validation_set, batch_sampler=BucketBatchSampler(validation_set, args.batch_size, seed=args.seed), collate_fn=pixel_art_collate)
    config = DirectStructureConfig(
        vocab_size=args.structure_vocab_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        text_dim=args.text_dim,
        max_height=args.max_size,
        max_width=args.max_size,
    )
    model = DirectStructureGenerator(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_epoch = load_training_checkpoint(args.resume, model, opt, device) + 1 if args.resume else 1
    last = {}

    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        totals = {key: 0.0 for key in ("loss", "structure_loss", "occupancy_loss", "boundary_loss", "count_loss")}
        images = 0
        steps = 0
        model.train()
        opt.zero_grad(set_to_none=True)
        for batch in loader:
            if steps == 0 or (steps + 1) % 50 == 0:
                print(f"train epoch {epoch} batch {steps + 1}/{len(loader)}", file=sys.stderr, flush=True)
            text = (text_cache.encode(_captions(batch)) if text_cache else text_encoder.encode(_captions(batch))).to(device)
            targets, valid = _targets(batch, device, config.vocab_size)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                losses = model.loss(
                    targets,
                    valid,
                    text,
                    args.foreground_weight,
                    args.boundary_weight,
                    args.occupancy_loss_weight,
                    args.boundary_loss_weight,
                    args.count_loss_weight,
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
            images += int(targets.shape[0])
        if steps % args.grad_accum:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        seconds = perf_counter() - started
        last = {
            "epoch": epoch,
            **_validation(model, validation_loader, text_encoder, args, device),
            "split": split_report(train_set, validation_set, test_set),
            "seconds": seconds,
            "images_per_second": images / max(seconds, 1e-9),
            "peak_vram_mb": torch.cuda.max_memory_reserved(device) / 1024 / 1024 if device.type == "cuda" else 0.0,
        } | {key: value / max(steps, 1) for key, value in totals.items()}
        append_jsonl(args.log, last)
        print(json.dumps(last, sort_keys=True))

    if args.checkpoint:
        save_training_checkpoint(args.checkpoint, model, opt, args.epochs, last)
    if args.viz:
        sample_args = argparse.Namespace(**(vars(args) | {"out": args.viz}))
        sample(sample_args)
    return last


@torch.no_grad()
def _validation(model, loader, text_encoder, args, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0.0
    error = 0.0
    steps = 0
    for batch in loader:
        text = text_encoder.encode(_captions(batch)).to(device)
        targets, valid = _targets(batch, device, model.config.vocab_size)
        losses = model.loss(
            targets,
            valid,
            text,
            args.foreground_weight,
            args.boundary_weight,
            args.occupancy_loss_weight,
            args.boundary_loss_weight,
            args.count_loss_weight,
            0.0,
        )
        masked = torch.full_like(targets, model.config.mask_token_id)
        pred = model(masked, valid, text)["logits"].argmax(dim=1)
        total += float(losses["loss"].item())
        error += float(pred[valid].ne(targets[valid]).float().mean().item() if valid.any() else 0.0)
        steps += 1
    return {"validation_loss": total / max(steps, 1), "validation_token_error": error / max(steps, 1)}


@torch.no_grad()
def sample(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = DirectStructureGenerator.load_checkpoint(args.checkpoint, map_location=device).to(device).eval()
    prompts = args.prompts or ["red potion icon", "stone house", "grass tile", "small tree"]
    prompts = (prompts * ((args.samples + len(prompts) - 1) // len(prompts)))[: args.samples]
    text = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=model.config.text_dim, provider=args.text_provider, device=args.device)).encode(prompts).to(device) if model.config.text_dim else None
    valid = torch.ones((args.samples, args.height, args.width), dtype=torch.bool, device=device)
    tokens = model.sample(valid.shape, valid, text, args.steps, args.temperature, args.guidance_scale)
    _save_viz(tokens, valid, args.out, args.samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("data", type=Path)
    train_parser.add_argument("--checkpoint", type=Path, default=Path("runs/m8_7_direct_structure.pt"))
    train_parser.add_argument("--viz", type=Path, default=Path("runs/m8_7_direct_structure.png"))
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--limit", type=int, default=32)
    train_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_parser.add_argument("--test-fraction", type=float, default=0.2)
    train_parser.add_argument("--include-lossy", action="store_true")
    train_parser.add_argument("--max-colors", type=int, default=64)
    train_parser.add_argument("--max-size", type=int, default=128)
    train_parser.add_argument("--structure-vocab-size", type=int, default=128)
    train_parser.add_argument("--hidden-dim", type=int, default=96)
    train_parser.add_argument("--depth", type=int, default=8)
    train_parser.add_argument("--text-dim", type=int, default=64)
    train_parser.add_argument("--text-provider", choices=["clip", "tiny"], default="clip")
    train_parser.add_argument("--cond-dropout", type=float, default=0.1)
    train_parser.add_argument("--foreground-weight", type=float, default=2.0)
    train_parser.add_argument("--boundary-weight", type=float, default=2.0)
    train_parser.add_argument("--occupancy-loss-weight", type=float, default=0.25)
    train_parser.add_argument("--boundary-loss-weight", type=float, default=0.25)
    train_parser.add_argument("--count-loss-weight", type=float, default=0.05)
    train_parser.add_argument("--embedding-cache", type=Path, default=Path("runs/m8_7_text_cache.pt"))
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--samples", type=int, default=4)
    train_parser.add_argument("--steps", type=int, default=8)
    train_parser.add_argument("--width", type=int, default=128)
    train_parser.add_argument("--height", type=int, default=128)
    train_parser.add_argument("--temperature", type=float, default=1.0)
    train_parser.add_argument("--guidance-scale", type=float, default=2.0)
    train_parser.add_argument("--prompts", nargs="*")
    train_parser.add_argument("--cache", action="store_true")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--log", type=Path, default=Path("runs/m8_7_direct_structure.jsonl"))
    train_parser.add_argument("--amp", action="store_true")
    train_parser.add_argument("--grad-accum", type=int, default=1)
    train_parser.set_defaults(func=train)

    sample_parser = sub.add_parser("sample")
    sample_parser.add_argument("--checkpoint", type=Path, required=True)
    sample_parser.add_argument("--out", type=Path, default=Path("runs/m8_7_direct_structure.png"))
    sample_parser.add_argument("--device", default="cpu")
    sample_parser.add_argument("--samples", type=int, default=4)
    sample_parser.add_argument("--width", type=int, default=128)
    sample_parser.add_argument("--height", type=int, default=128)
    sample_parser.add_argument("--steps", type=int, default=8)
    sample_parser.add_argument("--temperature", type=float, default=1.0)
    sample_parser.add_argument("--guidance-scale", type=float, default=2.0)
    sample_parser.add_argument("--text-provider", choices=["clip", "tiny"], default="clip")
    sample_parser.add_argument("--prompts", nargs="*")
    sample_parser.set_defaults(func=sample)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
