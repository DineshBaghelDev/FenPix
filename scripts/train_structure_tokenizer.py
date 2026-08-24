from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenpix.dataset import PixelArtDataset, filtered_indices, pixel_art_collate, split_report, train_val_test_split
from fenpix.training import append_jsonl, load_training_checkpoint, save_training_checkpoint, set_deterministic
from fenpix.tokenizer import (
    StructureTokenizer,
    StructureTokenizerConfig,
    masked_cross_entropy,
    structure_one_hot,
    tokenizer_metrics,
)


def _batch_to_model(batch, device: torch.device, num_classes: int):
    indices = batch["structure_indices"]
    valid = batch["valid_mask"]
    targets = indices.clamp_max(num_classes - 1)
    return structure_one_hot(targets, valid, num_classes).to(device), targets.to(device), valid.to(device)


def _save_grid(targets: torch.Tensor, logits: torch.Tensor, valid: torch.Tensor, path: Path, max_items: int = 8) -> None:
    pred = logits.argmax(dim=1).cpu()
    targets = targets.cpu()
    valid = valid.cpu()
    rows = []
    for i in range(min(max_items, targets.shape[0])):
        h, w = valid[i].nonzero().amax(dim=0).add(1).tolist()
        target = (targets[i, :h, :w].clamp_max(63) * 4).byte()
        recon = (pred[i, :h, :w].clamp_max(63) * 4).byte()
        blank = torch.full((h, 2), 255, dtype=torch.uint8)
        rows.append(torch.cat([target, blank, recon], dim=1))
    height = sum(row.shape[0] for row in rows)
    width = max(row.shape[1] for row in rows)
    canvas = torch.full((height, width), 255, dtype=torch.uint8)
    y = 0
    for row in rows:
        canvas[y : y + row.shape[0], : row.shape[1]] = row
        y += row.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas.numpy(), "L").save(path)


def train(args: argparse.Namespace) -> dict[str, float]:
    set_deterministic(args.seed)
    device = torch.device(args.device)
    dataset = PixelArtDataset(args.data, max_colors=args.max_colors, cache=args.cache)
    keep = filtered_indices(dataset, max_bucket_size=args.max_size, include_lossy=args.include_lossy, limit=args.limit)
    train_set, validation_set, test_set = train_val_test_split(Subset(dataset, keep), args.validation_fraction, args.test_fraction, args.seed)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=pixel_art_collate)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, collate_fn=pixel_art_collate)

    config = StructureTokenizerConfig(
        num_structure_classes=args.structure_classes,
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        downsample=args.downsample,
    )
    model = StructureTokenizer(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_epoch = load_training_checkpoint(args.resume, model, opt, device) + 1 if args.resume else 1
    last_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        images = 0
        model.train()
        total_loss = 0.0
        steps = 0
        opt.zero_grad(set_to_none=True)
        for batch in loader:
            x, targets, valid = _batch_to_model(batch, device, config.num_structure_classes)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                out = model(x)
                recon_loss = masked_cross_entropy(out["logits"], targets, valid)
                loss = (recon_loss + out["vq_loss"]) / args.grad_accum
            scaler.scale(loss).backward()
            if (steps + 1) % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total_loss += float(loss.item())
            steps += 1
            images += int(batch["indices"].shape[0])
        if steps % args.grad_accum:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        seconds = perf_counter() - started
        last_metrics = _validation_metrics(model, validation_loader, device, config)
        last_metrics["loss"] = total_loss * args.grad_accum / max(steps, 1)
        last_metrics["seconds"] = seconds
        last_metrics["images_per_second"] = images / max(seconds, 1e-9)
        last_metrics["peak_vram_mb"] = torch.cuda.max_memory_reserved(device) / 1024 / 1024 if device.type == "cuda" else 0.0
        last_metrics["split"] = split_report(train_set, validation_set, test_set)
        row = {"epoch": epoch, **last_metrics}
        append_jsonl(args.log, row)
        print(json.dumps(row, sort_keys=True))

    if args.checkpoint:
        save_training_checkpoint(args.checkpoint, model, opt, args.epochs, last_metrics)
    if args.viz:
        model.eval()
        batch = next(iter(loader))
        x, targets, valid = _batch_to_model(batch, device, config.num_structure_classes)
        with torch.no_grad():
            out = model(x)
        _save_grid(targets, out["logits"], valid, Path(args.viz))
    return last_metrics


@torch.no_grad()
def _validation_metrics(model, loader, device: torch.device, config: StructureTokenizerConfig) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    steps = 0
    last_batch = None
    for batch in loader:
        x, targets, valid = _batch_to_model(batch, device, config.num_structure_classes)
        out = model(x)
        metrics = tokenizer_metrics(out["logits"], targets, valid, out["codes"], config.codebook_size)
        pred = out["logits"].argmax(dim=1)
        metrics["reconstruction_quality"] = metrics["accuracy"]
        for bucket_size in batch["bucket_size"].unique().tolist():
            rows = batch["bucket_size"].eq(bucket_size)
            metrics[f"boundary_f1_{int(bucket_size)}"] = _boundary_f1_tokens(pred[rows], targets[rows], valid[rows])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        steps += 1
        last_batch = (targets, valid, out)
    if last_batch is None:
        return {}
    return {f"validation_{key}": value / max(steps, 1) for key, value in totals.items()}


def _boundary_f1_tokens(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    def edge(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        e = torch.zeros_like(m)
        e[1:] |= (x[1:] != x[:-1]) & m[1:] & m[:-1]
        e[:-1] |= (x[1:] != x[:-1]) & m[1:] & m[:-1]
        e[:, 1:] |= (x[:, 1:] != x[:, :-1]) & m[:, 1:] & m[:, :-1]
        e[:, :-1] |= (x[:, 1:] != x[:, :-1]) & m[:, 1:] & m[:, :-1]
        return e

    pred_edges = []
    target_edges = []
    for p, t, m in zip(pred.cpu(), target.cpu(), valid.cpu()):
        pred_edges.append(edge(p, m))
        target_edges.append(edge(t, m))
    pred_edge = torch.stack(pred_edges)
    target_edge = torch.stack(target_edges)
    tp = (pred_edge & target_edge).sum().item()
    fp = (pred_edge & ~target_edge).sum().item()
    fn = (~pred_edge & target_edge).sum().item()
    denom = 2 * tp + fp + fn
    return float(1.0 if denom == 0 else (2 * tp) / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/m3_structure_tokenizer.pt"))
    parser.add_argument("--viz", type=Path, default=Path("runs/m3_recon.png"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--include-lossy", action="store_true")
    parser.add_argument("--max-colors", type=int, default=64)
    parser.add_argument("--max-size", type=int, default=128)
    parser.add_argument("--structure-classes", type=int, default=65)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log", type=Path, default=Path("runs/m3_train.jsonl"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=1)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
