from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenpix.palette import image_to_indices


DEFAULT_SOURCE_URL = "https://kenney.nl/assets/tiny-town"
DEFAULT_LICENSE = "Creative Commons CC0"


def _tags(path: Path) -> list[str]:
    words = [path.stem, *[part for part in path.parts[:-1] if part.lower() not in {"png", "images"}]]
    tags: set[str] = set()
    for word in words:
        for token in word.replace("-", "_").replace(" ", "_").lower().split("_"):
            if token and not token.isdigit() and token not in {"tile", "kenney"}:
                tags.add(token)
    return sorted(tags)


def _has_transparency(image: Image.Image) -> bool:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return bool((rgba[..., 3] < 255).any())


def _reject(reason: str, report: dict[str, Any]) -> None:
    report["total_rejected"] += 1
    report["rejection_reasons"][reason] = report["rejection_reasons"].get(reason, 0) + 1


def _extract_zip(archive_path: Path, target: str | PathLike[str]) -> None:
    root = Path(target).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (root / member.filename).resolve()
            if not member_path.is_relative_to(root):
                raise ValueError(f"unsafe zip path: {member.filename}")
            archive.extract(member, root)


def _scan_root(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if input_path.is_dir():
        return input_path, None
    if input_path.is_file() and zipfile.is_zipfile(input_path):
        tmp = tempfile.TemporaryDirectory()
        try:
            _extract_zip(input_path, tmp.name)
        except Exception:
            tmp.cleanup()
            raise
        return Path(tmp.name), tmp
    raise ValueError(f"input must be a directory or zip archive: {input_path}")


def prepare_kenney(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source: str = "Kenney",
    source_url: str = DEFAULT_SOURCE_URL,
    license_text: str = DEFAULT_LICENSE,
    max_size: int = 128,
    max_colors: int = 64,
    max_pixel_art_colors: int = 256,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(f"output directory must be empty: {output_path}")
    root, tmp = _scan_root(input_path)
    report: dict[str, Any] = {
        "total_scanned": 0,
        "total_accepted": 0,
        "total_rejected": 0,
        "rejection_reasons": {},
        "size_distribution": {},
        "palette_size_distribution": {},
        "transparency_counts": {"true": 0, "false": 0},
    }

    try:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            report["total_scanned"] += 1
            if path.suffix.lower() != ".png":
                _reject("unsupported_format", report)
                continue

            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    if width <= 0 or height <= 0 or width > max_size or height > max_size:
                        _reject("oversized", report)
                        continue

                    encoding = image_to_indices(image, max_colors=max_colors)
                    if encoding.unique_color_count > max_pixel_art_colors:
                        _reject("non_pixel_art", report)
                        continue

                    transparency = _has_transparency(image)
            except (OSError, UnidentifiedImageError, ValueError):
                _reject("corrupt", report)
                continue

            rel = path.relative_to(root)
            target = output_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

            metadata = {
                "source": source,
                "source_url": source_url,
                "license": license_text,
                "provenance": str(rel).replace("\\", "/"),
                "width": width,
                "height": height,
                "transparency": transparency,
                "unique_color_count": encoding.unique_color_count,
                "palette_size_used": len(encoding.palette),
                "lossy": encoding.lossy,
                "tags": _tags(Path(root.name) / rel),
            }
            target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

            report["total_accepted"] += 1
            size_key = f"{width}x{height}"
            palette_key = str(len(encoding.palette))
            report["size_distribution"][size_key] = report["size_distribution"].get(size_key, 0) + 1
            report["palette_size_distribution"][palette_key] = report["palette_size_distribution"].get(palette_key, 0) + 1
            report["transparency_counts"][str(transparency).lower()] += 1
    finally:
        if tmp is not None:
            tmp.cleanup()

    output_path.mkdir(parents=True, exist_ok=True)
    report["rejection_reasons"] = dict(sorted(report["rejection_reasons"].items()))
    report["size_distribution"] = dict(sorted(report["size_distribution"].items()))
    report["palette_size_distribution"] = dict(sorted(report["palette_size_distribution"].items(), key=lambda item: int(item[0])))
    (output_path / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Kenney PNG assets as a FenPix dataset.")
    parser.add_argument("input", help="Raw Kenney folder or downloaded zip archive.")
    parser.add_argument("output", help="Processed FenPix dataset directory.")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--license", default=DEFAULT_LICENSE)
    args = parser.parse_args()

    report = prepare_kenney(args.input, args.output, source_url=args.source_url, license_text=args.license)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
