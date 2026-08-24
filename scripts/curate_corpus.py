from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenpix.corpus import curate_provisional_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a provisional curated FenPix corpus manifest.")
    parser.add_argument("input", type=Path, help="Candidate asset root.")
    parser.add_argument("output", type=Path, help="Derived curation output root.")
    parser.add_argument("--max-size", type=int, default=128)
    parser.add_argument("--max-colors", type=int, default=64)
    parser.add_argument("--max-pixel-art-colors", type=int, default=256)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    args = parser.parse_args()

    report = curate_provisional_corpus(
        args.input,
        args.output,
        max_size=args.max_size,
        max_colors=args.max_colors,
        max_pixel_art_colors=args.max_pixel_art_colors,
        near_duplicate_hamming=args.near_duplicate_hamming,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
