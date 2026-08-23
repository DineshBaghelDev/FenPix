from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenpix.dataset import build_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an indexed FenPix dataset manifest.")
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-colors", type=int, default=64)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    args = parser.parse_args()

    out = args.out or (args.data / "manifest.jsonl")
    report = build_dataset_manifest(
        args.data,
        out,
        max_colors=args.max_colors,
        near_duplicate_hamming=args.near_duplicate_hamming,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
