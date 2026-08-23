from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenpix.dataset import BucketBatchSampler, PixelArtDataset, filtered_indices, pixel_art_collate


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FenPix dataloader throughput and VRAM by bucket.")
    parser.add_argument("data", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-colors", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = PixelArtDataset(args.data, max_colors=args.max_colors, cache=args.cache)
    keep = filtered_indices(dataset, include_lossy=False, limit=args.limit)
    subset = Subset(dataset, keep)
    loader = DataLoader(subset, batch_sampler=BucketBatchSampler(subset, args.batch_size, seed=args.seed), collate_fn=pixel_art_collate)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    seen = 0
    start = perf_counter()
    bucket_counts: dict[str, int] = {}
    for batch in loader:
        indices = batch["indices"].to(device, non_blocking=True)
        palette = batch["palette"].to(device, non_blocking=True)
        _ = (indices.float().mean() + palette.float().mean()).item()
        seen += int(indices.shape[0])
        for bucket in batch["bucket"]:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    elapsed = max(perf_counter() - start, 1e-9)
    throughput = seen / elapsed
    report = {
        "samples": seen,
        "seconds": elapsed,
        "samples_per_second": throughput,
        "vram_peak_mb": torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else 0.0,
        "epoch_hours_500k": 500_000 / throughput / 3600,
        "epoch_hours_1m": 1_000_000 / throughput / 3600,
        "buckets": dict(sorted(bucket_counts.items())),
        "mixed_precision_available": bool(device.type == "cuda"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
