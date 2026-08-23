from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from fenpix.palette import image_to_indices, reconstruct_rgba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("png")
    parser.add_argument("--out", default="roundtrip.png")
    parser.add_argument("--max-colors", type=int, default=64)
    args = parser.parse_args()

    with Image.open(args.png) as image:
        encoding = image_to_indices(image, max_colors=args.max_colors)

    out = Path(args.out)
    reconstruct_rgba(encoding.indices, encoding.palette).save(out)
    print(f"wrote {out} ({encoding.width}x{encoding.height}, {len(encoding.palette)} colors)")


if __name__ == "__main__":
    main()
