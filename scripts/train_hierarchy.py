from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenpix.dataset import BucketBatchSampler, PixelArtDataset, pixel_art_collate
from fenpix.hierarchy import HierarchicalMaskGIT, HierarchicalMaskGITConfig, stage_tokens_from_batch
from fenpix.text import FrozenPretrainedTextEncoder, TextEmbeddingCache, TextEncoderConfig
from fenpix.tokenizer import StructureTokenizer


def _captions(batch) -> list[str]:
    out = []
    for meta in batch["metadata"]:
        caption = meta.get("caption") or meta.get("category") or "pixel art"
        category = meta.get("category")
        out.append(f"{caption} {category}".strip() if category and category not in str(caption) else str(caption))
    return out


def _dataset(args: argparse.Namespace):
    dataset = PixelArtDataset(args.data, max_colors=args.max_colors, cache=args.cache)
    keep = [i for i in range(len(dataset)) if int(dataset[i]["bucket_size"]) <= max(args.stages)]
    if args.limit:
        keep = keep[: args.limit]
    if not keep:
        raise ValueError("no <=128 PNGs found")
    return Subset(dataset, keep)


def _save_stage_grid(samples: dict[int, tuple[torch.Tensor, torch.Tensor]], path: Path, vocab_size: int) -> None:
    scale = max(1, 255 // max(1, vocab_size - 1))
    rows = []
    batch = next(iter(samples.values()))[0].shape[0]
    for b in range(batch):
        panels = []
        for stage in sorted(samples):
            tokens, valid = samples[stage]
            img = (tokens[b].cpu().clamp(0, vocab_size - 1) * scale).clamp_max(255).byte()
            img = img.masked_fill(~valid[b].cpu(), 255)
            panels.append(F.interpolate(img[None, None].float(), size=(32, 32), mode="nearest").squeeze().byte())
        rows.append(torch.cat(panels, dim=1))
    canvas = torch.cat(rows, dim=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas.numpy(), "L").save(path)


def train(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(args.device)
    tokenizer = StructureTokenizer.load_checkpoint(args.tokenizer, map_location=device).to(device).eval()
    text_encoder = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=args.text_dim, provider=args.text_provider, device=args.device))
    text_cache = TextEmbeddingCache(args.embedding_cache, text_encoder) if args.embedding_cache else None
    dataset = _dataset(args)
    loader = DataLoader(
        dataset,
        batch_sampler=BucketBatchSampler(dataset, args.batch_size, seed=args.seed),
        collate_fn=pixel_art_collate,
    )
    config = HierarchicalMaskGITConfig(
        vocab_size=tokenizer.config.codebook_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        text_dim=args.text_dim,
        cond_tokens=args.cond_tokens,
        downsample=tokenizer.config.downsample,
        stages=tuple(args.stages),
    )
    model = HierarchicalMaskGIT(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    last = {}

    for epoch in range(1, args.epochs + 1):
        totals = {stage: 0.0 for stage in config.stages}
        steps = 0
        model.train()
        for batch in loader:
            texts = _captions(batch)
            text_embeddings = (text_cache.encode(texts) if text_cache else text_encoder.encode(texts)).to(device)
            tokens = {stage: stage_tokens_from_batch(batch, tokenizer, stage, device) for stage in config.stages}
            loss = torch.zeros((), device=device)
            previous = None
            for stage in config.stages:
                stage_loss = model.stage_loss(
                    stage,
                    tokens[stage][0],
                    tokens[stage][1],
                    text_embeddings,
                    *(previous or (None, None)),
                    cond_drop_prob=args.cond_dropout,
                )
                totals[stage] += float(stage_loss.item())
                loss = loss + stage_loss
                previous = tokens[stage]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
        last = {"epoch": epoch} | {f"loss_{stage}": totals[stage] / max(steps, 1) for stage in config.stages}
        print(json.dumps(last, sort_keys=True))

    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(args.checkpoint, extra=last)
    if args.viz:
        sample_args = argparse.Namespace(**(vars(args) | {"out": args.viz}))
        sample(sample_args)
    return last


def sample(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = HierarchicalMaskGIT.load_checkpoint(args.checkpoint, map_location=device).to(device).eval()
    prompts = args.prompts or ["red potion icon", "stone house", "grass tile", "small tree"]
    prompts = (prompts * ((args.samples + len(prompts) - 1) // len(prompts)))[: args.samples]
    text_embeddings = None
    if model.config.text_dim > 0:
        text_embeddings = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=model.config.text_dim, provider=args.text_provider, device=args.device)).encode(prompts).to(device)
    samples = model.sample(
        args.width,
        args.height,
        prompts,
        text_embeddings=text_embeddings,
        steps=args.steps,
        temperature=args.temperature,
        guidance_scale=args.guidance_scale,
    )
    _save_stage_grid(samples, args.out, model.config.vocab_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("data", type=Path)
    train_parser.add_argument("--tokenizer", type=Path, required=True)
    train_parser.add_argument("--checkpoint", type=Path, default=Path("runs/m6_hierarchy.pt"))
    train_parser.add_argument("--viz", type=Path, default=Path("runs/m6_stages.png"))
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--limit", type=int, default=32)
    train_parser.add_argument("--max-colors", type=int, default=64)
    train_parser.add_argument("--hidden-dim", type=int, default=64)
    train_parser.add_argument("--depth", type=int, default=2)
    train_parser.add_argument("--heads", type=int, default=4)
    train_parser.add_argument("--text-dim", type=int, default=64)
    train_parser.add_argument("--text-provider", choices=["clip", "tiny"], default="clip")
    train_parser.add_argument("--cond-tokens", type=int, default=1)
    train_parser.add_argument("--cond-dropout", type=float, default=0.1)
    train_parser.add_argument("--embedding-cache", type=Path, default=Path("runs/m6_text_cache.pt"))
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--stages", type=int, nargs="+", default=[32, 64, 128])
    train_parser.add_argument("--samples", type=int, default=4)
    train_parser.add_argument("--steps", type=int, default=8)
    train_parser.add_argument("--width", type=int, default=128)
    train_parser.add_argument("--height", type=int, default=128)
    train_parser.add_argument("--temperature", type=float, default=1.0)
    train_parser.add_argument("--guidance-scale", type=float, default=2.0)
    train_parser.add_argument("--prompts", nargs="*")
    train_parser.add_argument("--cache", action="store_true")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.set_defaults(func=train)

    sample_parser = sub.add_parser("sample")
    sample_parser.add_argument("--checkpoint", type=Path, required=True)
    sample_parser.add_argument("--out", type=Path, default=Path("runs/m6_stages.png"))
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
