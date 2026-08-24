from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenpix.corpus import prepare_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a licensed multi-source FenPix corpus.")
    parser.add_argument("config", type=Path, help="JSON config with a sources array.")
    parser.add_argument("output", type=Path, help="Output corpus directory.")
    parser.add_argument("--target-count", type=int, default=100_000)
    parser.add_argument("--min-count", type=int, default=50_000)
    parser.add_argument("--max-size", type=int, default=128)
    parser.add_argument("--max-colors", type=int, default=64)
    parser.add_argument("--max-pixel-art-colors", type=int, default=256)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--per-bucket-cap", type=int)
    parser.add_argument("--compose-scenes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh-downloads", action="store_true")
    args = parser.parse_args()

    report = prepare_corpus(
        args.config,
        args.output,
        target_count=args.target_count,
        min_count=args.min_count,
        max_size=args.max_size,
        max_colors=args.max_colors,
        max_pixel_art_colors=args.max_pixel_art_colors,
        near_duplicate_hamming=args.near_duplicate_hamming,
        per_bucket_cap=args.per_bucket_cap,
        compose_scenes=args.compose_scenes,
        seed=args.seed,
        refresh_downloads=args.refresh_downloads,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
