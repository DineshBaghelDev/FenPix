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

from fenpix.dataset import PixelArtDataset, pixel_art_collate
from fenpix.maskgit import MaskGIT, MaskGITConfig, maskgit_loss, random_mask_tokens
from fenpix.text import FrozenPretrainedTextEncoder, TextEmbeddingCache, TextEncoderConfig
from fenpix.tokenizer import StructureTokenizer, canonical_structure_indices, structure_one_hot


@torch.no_grad()
def _tokens_from_batch(batch, tokenizer: StructureTokenizer, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    indices = batch["structure_indices"].to(device)
    palette = batch["palette"].to(device)
    valid = batch["valid_mask"].to(device)
    targets = canonical_structure_indices(indices, palette, valid, max_regions=tokenizer.config.num_structure_classes - 1)
    out = tokenizer(structure_one_hot(targets, valid, tokenizer.config.num_structure_classes))
    token_valid = F.interpolate(valid[:, None].float(), size=out["codes"].shape[-2:], mode="nearest").squeeze(1).bool()
    return out["codes"], token_valid


def _save_token_grid(tokens: torch.Tensor, valid: torch.Tensor, path: Path, vocab_size: int, max_items: int = 16) -> None:
    tokens = tokens.cpu()
    valid = valid.cpu()
    rows = []
    scale = max(1, 255 // max(1, vocab_size - 1))
    for i in range(min(max_items, tokens.shape[0])):
        if valid[i].any():
            h, w = valid[i].nonzero().amax(dim=0).add(1).tolist()
        else:
            h, w = tokens.shape[-2:]
        img = (tokens[i, :h, :w].clamp(0, vocab_size - 1) * scale).clamp_max(255).byte()
        img = img.masked_fill(~valid[i, :h, :w], 255)
        rows.append(img)
    tile_h = max(row.shape[0] for row in rows)
    tile_w = max(row.shape[1] for row in rows)
    cols = min(4, len(rows))
    canvas = torch.full((tile_h * ((len(rows) + cols - 1) // cols), tile_w * cols), 255, dtype=torch.uint8)
    for i, row in enumerate(rows):
        y = (i // cols) * tile_h
        x = (i % cols) * tile_w
        canvas[y : y + row.shape[0], x : x + row.shape[1]] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas.numpy(), "L").save(path)


def _captions(batch) -> list[str]:
    out = []
    for meta in batch["metadata"]:
        caption = meta.get("caption") or meta.get("category") or "pixel art"
        category = meta.get("category")
        out.append(f"{caption} {category}".strip() if category and category not in str(caption) else str(caption))
    return out


def _dataset(args: argparse.Namespace):
    dataset = PixelArtDataset(args.data, max_colors=args.max_colors, cache=args.cache)
    keep = [i for i in range(len(dataset)) if int(dataset[i]["bucket_size"]) == 32]
    if args.limit:
        keep = keep[: args.limit]
    if not keep:
        raise ValueError("no 32x32-class PNGs found")
    return Subset(dataset, keep)


def train(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(args.device)
    tokenizer = StructureTokenizer.load_checkpoint(args.tokenizer, map_location=device).to(device).eval()
    text_encoder = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=args.text_dim, provider=args.text_provider, device=args.device))
    text_cache = TextEmbeddingCache(args.embedding_cache, text_encoder) if args.embedding_cache else None
    loader = DataLoader(_dataset(args), batch_size=args.batch_size, shuffle=True, collate_fn=pixel_art_collate)
    first_tokens, _ = _tokens_from_batch(next(iter(loader)), tokenizer, device)
    config = MaskGITConfig(
        vocab_size=tokenizer.config.codebook_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        max_height=max(args.max_grid, first_tokens.shape[-2]),
        max_width=max(args.max_grid, first_tokens.shape[-1]),
        text_dim=args.text_dim,
        cond_tokens=args.cond_tokens,
    )
    model = MaskGIT(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    last = {}

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        steps = 0
        model.train()
        for batch in loader:
            tokens, valid = _tokens_from_batch(batch, tokenizer, device)
            texts = _captions(batch)
            text_embeddings = (text_cache.encode(texts) if text_cache else text_encoder.encode(texts)).to(device)
            masked, labels = random_mask_tokens(tokens, valid, config.mask_token_id)
            loss = maskgit_loss(model(masked, valid, text_embeddings, cond_drop_prob=args.cond_dropout), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        last = {"epoch": epoch, "loss": total / max(steps, 1)}
        print(json.dumps(last, sort_keys=True))

    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(args.checkpoint, extra=last)
    if args.viz:
        model.eval()
        with torch.no_grad():
            valid = torch.ones((args.samples, first_tokens.shape[-2], first_tokens.shape[-1]), dtype=torch.bool, device=device)
            prompts = (args.prompts or ["red potion icon", "stone house", "grass tile", "small tree"])[: args.samples]
            prompts = (prompts * ((args.samples + len(prompts) - 1) // len(prompts)))[: args.samples]
            text_embeddings = text_encoder.encode(prompts).to(device)
            samples = model.sample(tuple(valid.shape), valid, steps=args.sample_steps, text_embeddings=text_embeddings, guidance_scale=args.guidance_scale)
        _save_token_grid(samples, valid, Path(args.viz), config.vocab_size)
    return last


def sample(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = MaskGIT.load_checkpoint(args.checkpoint, map_location=device).to(device).eval()
    prompts = args.prompts or ["red potion icon", "stone house", "grass tile", "small tree"]
    prompts = (prompts * ((args.samples + len(prompts) - 1) // len(prompts)))[: args.samples]
    valid = torch.ones((len(prompts), args.height, args.width), dtype=torch.bool, device=device)
    text_embeddings = None
    if model.config.text_dim > 0:
        text_embeddings = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=model.config.text_dim, provider=args.text_provider, device=args.device)).encode(prompts).to(device)
    tokens = model.sample(
        tuple(valid.shape),
        valid,
        steps=args.steps,
        temperature=args.temperature,
        text_embeddings=text_embeddings,
        guidance_scale=args.guidance_scale,
    )
    _save_token_grid(tokens, valid, args.out, model.config.vocab_size, max_items=args.samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("data", type=Path)
    train_parser.add_argument("--tokenizer", type=Path, required=True)
    train_parser.add_argument("--checkpoint", type=Path, default=Path("runs/m5_maskgit.pt"))
    train_parser.add_argument("--viz", type=Path, default=Path("runs/m5_samples.png"))
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--limit", type=int, default=32)
    train_parser.add_argument("--max-colors", type=int, default=64)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--depth", type=int, default=4)
    train_parser.add_argument("--heads", type=int, default=4)
    train_parser.add_argument("--max-grid", type=int, default=32)
    train_parser.add_argument("--text-dim", type=int, default=64)
    train_parser.add_argument("--text-provider", choices=["clip", "tiny"], default="clip")
    train_parser.add_argument("--cond-tokens", type=int, default=1)
    train_parser.add_argument("--cond-dropout", type=float, default=0.1)
    train_parser.add_argument("--embedding-cache", type=Path, default=Path("runs/m5_text_cache.pt"))
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--samples", type=int, default=16)
    train_parser.add_argument("--sample-steps", type=int, default=8)
    train_parser.add_argument("--guidance-scale", type=float, default=2.0)
    train_parser.add_argument("--prompts", nargs="*")
    train_parser.add_argument("--cache", action="store_true")
    train_parser.set_defaults(func=train)

    sample_parser = sub.add_parser("sample")
    sample_parser.add_argument("--checkpoint", type=Path, required=True)
    sample_parser.add_argument("--out", type=Path, default=Path("runs/m5_samples.png"))
    sample_parser.add_argument("--device", default="cpu")
    sample_parser.add_argument("--samples", type=int, default=16)
    sample_parser.add_argument("--height", type=int, default=8)
    sample_parser.add_argument("--width", type=int, default=8)
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
