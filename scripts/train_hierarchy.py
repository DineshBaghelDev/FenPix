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
from fenpix.hierarchy import HierarchicalMaskGIT, HierarchicalMaskGITConfig, condition_to_shape, stage_structure_from_batch, stage_tokens_from_batch
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
    keep = filtered_indices(dataset, max_bucket_size=max(args.stages), include_lossy=args.include_lossy, limit=args.limit)
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


def _noisy_condition(tokens: torch.Tensor, valid: torch.Tensor, vocab_size: int, noise_prob: float) -> tuple[torch.Tensor, torch.Tensor]:
    if noise_prob <= 0:
        return tokens, valid
    replace = torch.rand(tokens.shape, device=tokens.device).lt(noise_prob) & valid
    noise = torch.randint(0, vocab_size, tokens.shape, device=tokens.device)
    return tokens.masked_scatter(replace, noise[replace]), valid


@torch.no_grad()
def _sample_stage_condition(
    model: HierarchicalMaskGIT,
    stage: int,
    tokens: torch.Tensor,
    valid: torch.Tensor,
    text_embeddings: torch.Tensor,
    lower: tuple[torch.Tensor, torch.Tensor] | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    stage_model = model.models[str(stage)]
    cond = condition_to_shape(lower[0], lower[1], tokens.shape[-2:], stage_model.config.pad_token_id) if lower else None
    sampled = stage_model.sample(
        tokens.shape,
        valid,
        steps=args.sampled_lower_steps,
        temperature=args.temperature,
        text_embeddings=text_embeddings,
        structure_condition=cond,
        guidance_scale=args.guidance_scale,
    )
    return sampled, valid


def _mixed_condition(
    previous_gt: tuple[torch.Tensor, torch.Tensor] | None,
    previous_sampled: tuple[torch.Tensor, torch.Tensor] | None,
    args: argparse.Namespace,
    vocab_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if previous_gt is None:
        return None, None
    roll = torch.rand(()).item()
    if previous_sampled is not None and roll < args.sampled_lower_prob:
        return previous_sampled
    if roll < args.sampled_lower_prob + args.corrupt_lower_prob:
        return _noisy_condition(*previous_gt, vocab_size, args.lower_noise_prob)
    return previous_gt


def train(args: argparse.Namespace) -> dict[str, float]:
    set_deterministic(args.seed)
    device = torch.device(args.device)
    tokenizer = StructureTokenizer.load_checkpoint(args.tokenizer, map_location=device).to(device).eval()
    for param in tokenizer.parameters():
        param.requires_grad_(False)
    text_encoder = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=args.text_dim, provider=args.text_provider, device=args.device))
    text_cache = TextEmbeddingCache(args.embedding_cache, text_encoder) if args.embedding_cache else None
    dataset = _dataset(args)
    train_set, validation_set, test_set = train_val_test_split(dataset, args.validation_fraction, args.test_fraction, args.seed)
    loader = DataLoader(
        train_set,
        batch_sampler=BucketBatchSampler(train_set, args.batch_size, seed=args.seed),
        collate_fn=pixel_art_collate,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_sampler=BucketBatchSampler(validation_set, args.batch_size, seed=args.seed),
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
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_epoch = load_training_checkpoint(args.resume, model, opt, device) + 1 if args.resume else 1
    last = {}

    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        images = 0
        totals = {stage: 0.0 for stage in config.stages}
        steps = 0
        model.train()
        opt.zero_grad(set_to_none=True)
        for batch in loader:
            if steps == 0 or (steps + 1) % 50 == 0:
                print(f"train epoch {epoch} batch {steps + 1}/{len(loader)}", file=sys.stderr, flush=True)
            texts = _captions(batch)
            text_embeddings = (text_cache.encode(texts) if text_cache else text_encoder.encode(texts)).to(device)
            tokens = {stage: stage_tokens_from_batch(batch, tokenizer, stage, device, args.component_targets) for stage in config.stages}
            structures = {stage: stage_structure_from_batch(batch, stage, config.downsample, device, args.component_targets) for stage in config.stages}
            loss = torch.zeros((), device=device)
            previous_gt = None
            previous_sampled = None
            for stage in config.stages:
                cond = _mixed_condition(previous_gt, previous_sampled, args, config.vocab_size)
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    stage_loss = model.stage_loss(
                        stage,
                        tokens[stage][0],
                        tokens[stage][1],
                        text_embeddings,
                        *cond,
                        cond_drop_prob=args.cond_dropout,
                        target_structure=structures[stage][0],
                        target_structure_valid=structures[stage][1],
                        tokenizer=tokenizer,
                        foreground_weight=args.foreground_weight,
                        boundary_weight=args.boundary_weight,
                        foreground_loss_weight=args.foreground_loss_weight,
                    )
                totals[stage] += float(stage_loss.item())
                loss = loss + stage_loss
                lower_for_sample = previous_sampled or previous_gt
                previous_sampled = _sample_stage_condition(model, stage, tokens[stage][0], tokens[stage][1], text_embeddings, lower_for_sample, args) if args.sampled_lower_prob > 0 else None
                previous_gt = tokens[stage]
            scaler.scale(loss / args.grad_accum).backward()
            if (steps + 1) % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            steps += 1
            images += int(batch["indices"].shape[0])
        if steps % args.grad_accum:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        seconds = perf_counter() - started
        last = {
            "epoch": epoch,
            **_validation_loss(model, validation_loader, tokenizer, text_encoder, config, args, device),
            "split": split_report(train_set, validation_set, test_set),
            "seconds": seconds,
            "images_per_second": images / max(seconds, 1e-9),
            "peak_vram_mb": torch.cuda.max_memory_reserved(device) / 1024 / 1024 if device.type == "cuda" else 0.0,
        } | {f"loss_{stage}": totals[stage] / max(steps, 1) for stage in config.stages}
        append_jsonl(args.log, last)
        print(json.dumps(last, sort_keys=True))

    if args.checkpoint:
        save_training_checkpoint(args.checkpoint, model, opt, args.epochs, last)
    if args.viz:
        sample_args = argparse.Namespace(**(vars(args) | {"out": args.viz}))
        sample(sample_args)
    return last


@torch.no_grad()
def _validation_loss(model, loader, tokenizer, text_encoder, config, args, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0.0
    steps = 0
    errors = {f"validation_token_error_{a}_{b}": 0.0 for a, b in zip(config.stages, config.stages[1:])}
    errors |= {f"validation_loss_{a}_{b}": 0.0 for a, b in zip(config.stages, config.stages[1:])}
    for batch in loader:
        text_embeddings = text_encoder.encode(_captions(batch)).to(device)
        tokens = {stage: stage_tokens_from_batch(batch, tokenizer, stage, device, args.component_targets) for stage in config.stages}
        structures = {stage: stage_structure_from_batch(batch, stage, config.downsample, device, args.component_targets) for stage in config.stages}
        loss = torch.zeros((), device=device)
        previous = None
        for stage in config.stages:
            stage_loss = model.stage_loss(
                stage,
                tokens[stage][0],
                tokens[stage][1],
                text_embeddings,
                *(previous or (None, None)),
                target_structure=structures[stage][0],
                target_structure_valid=structures[stage][1],
                tokenizer=tokenizer,
                foreground_weight=args.foreground_weight,
                boundary_weight=args.boundary_weight,
                foreground_loss_weight=args.foreground_loss_weight,
            )
            loss = loss + stage_loss
            if previous is not None:
                stage_model = model.models[str(stage)]
                cond = condition_to_shape(previous[0], previous[1], tokens[stage][0].shape[-2:], stage_model.config.pad_token_id)
                masked = torch.full_like(tokens[stage][0], stage_model.config.mask_token_id)
                pred = stage_model(masked, tokens[stage][1], text_embeddings, cond).argmax(dim=1)
                valid = tokens[stage][1]
                errors[f"validation_token_error_{previous_stage}_{stage}"] += float(pred[valid].ne(tokens[stage][0][valid]).float().mean().item() if valid.any() else 0.0)
                errors[f"validation_loss_{previous_stage}_{stage}"] += float(stage_loss.item())
            previous = tokens[stage]
            previous_stage = stage
        total += float(loss.item())
        steps += 1
    out = {"validation_loss": total / max(steps, 1)}
    out.update({key: value / max(steps, 1) for key, value in errors.items()})
    return out


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
    train_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_parser.add_argument("--test-fraction", type=float, default=0.2)
    train_parser.add_argument("--max-colors", type=int, default=64)
    train_parser.add_argument("--include-lossy", action="store_true")
    train_parser.add_argument("--palette-region-targets", dest="component_targets", action="store_false")
    train_parser.set_defaults(component_targets=True)
    train_parser.add_argument("--hidden-dim", type=int, default=64)
    train_parser.add_argument("--depth", type=int, default=2)
    train_parser.add_argument("--heads", type=int, default=4)
    train_parser.add_argument("--text-dim", type=int, default=64)
    train_parser.add_argument("--text-provider", choices=["clip", "tiny"], default="clip")
    train_parser.add_argument("--cond-tokens", type=int, default=1)
    train_parser.add_argument("--cond-dropout", type=float, default=0.1)
    train_parser.add_argument("--sampled-lower-prob", type=float, default=0.25)
    train_parser.add_argument("--corrupt-lower-prob", type=float, default=0.25)
    train_parser.add_argument("--sampled-lower-steps", type=int, default=4)
    train_parser.add_argument("--lower-noise-prob", type=float, default=0.15)
    train_parser.add_argument("--foreground-weight", type=float, default=2.0)
    train_parser.add_argument("--boundary-weight", type=float, default=2.0)
    train_parser.add_argument("--foreground-loss-weight", type=float, default=0.25)
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
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--log", type=Path, default=Path("runs/m6_train.jsonl"))
    train_parser.add_argument("--amp", action="store_true")
    train_parser.add_argument("--grad-accum", type=int, default=1)
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
