from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenpix.dataset import PixelArtDataset, filtered_indices, pixel_art_collate, split_report, train_val_test_split
from fenpix.tokenizer import (
    StructureTokenizer,
    StructureTokenizerConfig,
    canonical_structure_indices,
    masked_cross_entropy,
    structure_one_hot,
    tokenizer_metrics,
)


def _batch_to_model(batch, device: torch.device, num_classes: int):
    indices = batch["structure_indices"].to(device)
    palette = batch["palette"].to(device)
    valid = batch["valid_mask"].to(device)
    targets = canonical_structure_indices(indices, palette, valid, max_regions=num_classes - 1)
    return structure_one_hot(targets, valid, num_classes), targets, valid


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
    last_metrics: dict[str, float] = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            x, targets, valid = _batch_to_model(batch, device, config.num_structure_classes)
            out = model(x)
            recon_loss = masked_cross_entropy(out["logits"], targets, valid)
            loss = recon_loss + out["vq_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            steps += 1

        last_metrics = _validation_metrics(model, validation_loader, device, config)
        last_metrics["loss"] = total_loss / max(steps, 1)
        last_metrics["split"] = split_report(train_set, validation_set, test_set)
        print(json.dumps({"epoch": epoch, **last_metrics}, sort_keys=True))

    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(args.checkpoint, extra={"metrics": last_metrics})
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
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        steps += 1
        last_batch = (targets, valid, out)
    if last_batch is None:
        return {}
    return {f"validation_{key}": value / max(steps, 1) for key, value in totals.items()}


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
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
