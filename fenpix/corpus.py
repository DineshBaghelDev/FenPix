from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .dataset import (
    _PhashIndex,
    _aspect_bucket,
    _bucket_size,
    _palette_manifest_stats,
    _perceptual_hash,
    build_dataset_manifest,
    bucket_id,
    dataset_manifest_report,
    load_dataset_manifest,
    save_dataset_manifest,
)
from .palette import image_to_indices


ALLOWED_LICENSES = {
    "cc0",
    "creative commons cc0",
    "cc0-1.0",
    "cc0 1.0",
    "public domain",
    "pdm",
    "cc-by",
    "cc by",
    "cc-by-3.0",
    "cc-by-4.0",
    "oga-by",
    "oga-by-3.0",
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "mixed permissive",
}

DENIED_LICENSE_PARTS = ("nc", "nd", "noncommercial", "no derivatives", "all rights reserved", "unknown")


@dataclass(frozen=True)
class CorpusPaths:
    root: Path
    downloads: Path
    extracted: Path
    assets: Path
    final_manifest: Path
    report: Path


def corpus_paths(output: str | Path) -> CorpusPaths:
    root = Path(output)
    return CorpusPaths(
        root=root,
        downloads=root / "_downloads",
        extracted=root / "_extracted",
        assets=root / "assets",
        final_manifest=root / "manifest.jsonl",
        report=root / "report.json",
    )


def prepare_corpus(
    config_path: str | Path,
    output: str | Path,
    *,
    target_count: int = 100_000,
    min_count: int = 50_000,
    max_size: int = 128,
    max_colors: int = 64,
    max_pixel_art_colors: int = 256,
    near_duplicate_hamming: int = 4,
    per_bucket_cap: int | None = None,
    compose_scenes: int = 0,
    seed: int = 0,
    refresh_downloads: bool = False,
) -> dict[str, Any]:
    paths = corpus_paths(output)
    paths.root.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "sources": [],
        "total_scanned": 0,
        "total_accepted": 0,
        "total_rejected": 0,
        "rejection_reasons": {},
        "target_count": target_count,
        "min_count": min_count,
    }

    for source in config.get("sources", []):
        if source.get("enabled", True) is False:
            report["sources"].append(
                {
                    "name": str(source["name"]),
                    "scanned": 0,
                    "accepted": 0,
                    "rejected": 0,
                    "rejection_reasons": {"disabled": 1},
                    "blocked_reason": source.get("blocked_reason"),
                }
            )
            continue
        source_report = ingest_source(
            source,
            paths,
            max_size=max_size,
            max_colors=max_colors,
            max_pixel_art_colors=max_pixel_art_colors,
            refresh_downloads=refresh_downloads,
        )
        report["sources"].append(source_report)
        report["total_scanned"] += source_report["scanned"]
        report["total_accepted"] += source_report["accepted"]
        report["total_rejected"] += source_report["rejected"]
        for reason, count in source_report["rejection_reasons"].items():
            report["rejection_reasons"][reason] = report["rejection_reasons"].get(reason, 0) + count

    if compose_scenes:
        compose_report = compose_scene_samples(paths.assets, compose_scenes, seed=seed)
        report["composition"] = compose_report
        report["total_accepted"] += compose_report["created"]

    raw_manifest = paths.root / "manifest.raw.jsonl"
    manifest_report = build_dataset_manifest(paths.assets, raw_manifest, max_colors=max_colors, near_duplicate_hamming=near_duplicate_hamming)
    rows = load_dataset_manifest(raw_manifest, root=paths.assets)
    curated_rows, curation = curate_manifest_rows(rows, target_count=target_count, per_bucket_cap=per_bucket_cap, seed=seed)
    save_dataset_manifest(curated_rows, paths.final_manifest)
    save_dataset_manifest(curated_rows, paths.assets / "manifest.jsonl")
    report["manifest"] = manifest_report
    report["curation"] = curation | dataset_manifest_report(curated_rows)
    report["status"] = "ready" if len(curated_rows) >= min_count else "below_min_count"
    paths.report.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report


def ingest_source(
    source: dict[str, Any],
    paths: CorpusPaths,
    *,
    max_size: int,
    max_colors: int,
    max_pixel_art_colors: int,
    refresh_downloads: bool = False,
) -> dict[str, Any]:
    name = str(source["name"])
    license_text = str(source.get("license") or "")
    if not is_allowed_license(license_text):
        return {"name": name, "scanned": 0, "accepted": 0, "rejected": 0, "rejection_reasons": {"license_not_allowed": 1}}

    source_root, tmp = resolve_source_root(source, paths, refresh_downloads=refresh_downloads)
    if source.get("extract_archives"):
        source_root = extract_archives(source_root, paths.extracted / f"{slug(name)}_archives")
    extracted_parquet = extract_parquet_images(source_root, paths.extracted / f"{slug(name)}_parquet", source) if source.get("extract_parquet") else None
    if extracted_parquet is not None:
        source_root = extracted_parquet
    report = {"name": name, "scanned": 0, "accepted": 0, "rejected": 0, "rejection_reasons": {}}
    try:
        for path in sorted(p for p in source_root.rglob("*.png") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(source_root).parts)):
            report["scanned"] += 1
            if report["scanned"] == 1 or report["scanned"] % 1000 == 0:
                print(f"{name}: scanned={report['scanned']} accepted={report['accepted']} rejected={report['rejected']}", file=sys.stderr, flush=True)
            rel = path.relative_to(source_root)
            target = unique_target(paths.assets / slug(name) / rel)
            if target.exists() and target.with_suffix(".json").exists():
                report["accepted"] += 1
                continue
            reason = validate_png(path, max_size=max_size, max_colors=max_colors, max_pixel_art_colors=max_pixel_art_colors)
            if reason:
                _reject(report, reason)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            metadata = source_metadata(source, rel, target)
            target.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            report["accepted"] += 1
    finally:
        if tmp is not None:
            tmp.cleanup()
    return report


def resolve_source_root(source: dict[str, Any], paths: CorpusPaths, *, refresh_downloads: bool) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if "hf_repo" in source:
        return huggingface_source_root(source, paths, refresh_downloads=refresh_downloads), None
    if "url" in source:
        paths.downloads.mkdir(parents=True, exist_ok=True)
        url = str(source["url"])
        target = paths.downloads / (source.get("file_name") or Path(urllib.parse.urlparse(url).path).name or f"{slug(source['name'])}.download")
        if refresh_downloads or not target.exists():
            urllib.request.urlretrieve(url, target)
        return archive_or_file_root(target, paths.extracted / slug(str(source["name"])))
    return archive_or_file_root(Path(source["path"]), paths.extracted / slug(str(source["name"])))


def huggingface_source_root(source: dict[str, Any], paths: CorpusPaths, *, refresh_downloads: bool) -> Path:
    repo_id = str(source["hf_repo"])
    out = paths.downloads / "huggingface" / slug(repo_id)
    if out.exists() and not refresh_downloads:
        return out
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for hf_repo sources") from exc
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=out,
        allow_patterns=source.get("allow_patterns"),
        ignore_patterns=source.get("ignore_patterns"),
    )
    return out


def extract_parquet_images(root: Path, out: Path, source: dict[str, Any]) -> Path:
    marker = out / ".complete"
    out.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for extract_parquet sources") from exc
    count = 0
    parquet_paths = sorted(root.rglob("*.parquet"))
    expected = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_paths)
    palette = parquet_palette(root) if source.get("extract_grid") else None
    if marker.exists():
        try:
            if int(marker.read_text(encoding="utf-8").strip()) >= expected:
                return out
        except ValueError:
            pass
    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        columns = parquet_image_columns(parquet.schema.names, include_grid=palette is not None)
        row_index = 0
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                payload = first_image_payload(row)
                if payload is None and palette is not None:
                    payload = grid_image_payload(row, palette)
                if payload is not None:
                    target = out / parquet_path.stem / f"{row_index:08d}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        target.write_bytes(payload)
                    metadata = source_metadata(source, Path(parquet_path.name) / target.name, target)
                    caption = row.get("text") or row.get("prompt") or row.get("caption")
                    if caption:
                        metadata["caption"] = str(caption)
                    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    count += 1
                    if count == 1 or count % 1000 == 0:
                        print(f"{source['name']}: parquet_images={count}", file=sys.stderr, flush=True)
                row_index += 1
    if count == 0:
        raise ValueError(f"no image payloads found in parquet source {root}")
    marker.write_text(f"{count}\n", encoding="utf-8")
    return out


def parquet_image_columns(names: list[str], *, include_grid: bool = False) -> list[str] | None:
    wanted = [name for name in ("preview", "image", "img", "prompt", "text", "caption") if name in names]
    if "bytes" in names:
        wanted.append("preview")
    if include_grid:
        wanted.append("grid")
    return wanted or None


def parquet_palette(root: Path) -> np.ndarray | None:
    palettes = sorted(root.rglob("palette.npy"))
    if not palettes:
        return None
    palette = np.load(palettes[0])
    if palette.ndim != 2 or palette.shape[1] not in (3, 4):
        raise ValueError(f"unsupported palette shape in {palettes[0]}: {palette.shape}")
    return palette.astype(np.uint8)


def grid_image_payload(row: dict[str, Any], palette: np.ndarray) -> bytes | None:
    grid = row.get("grid")
    if not grid:
        return None
    side = int(len(grid) ** 0.5)
    if side * side != len(grid):
        return None
    indexed = np.asarray(grid, dtype=np.uint8).reshape(side, side)
    image = Image.fromarray(palette[indexed], "RGBA" if palette.shape[1] == 4 else "RGB")
    with tempfile.SpooledTemporaryFile() as buffer:
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.read()


def first_image_payload(row: dict[str, Any]) -> bytes | None:
    for value in row.values():
        if isinstance(value, (bytes, bytearray)) and value[:8] == b"\x89PNG\r\n\x1a\n":
            return bytes(value)
        if isinstance(value, dict):
            raw = value.get("bytes")
            if isinstance(raw, (bytes, bytearray)) and raw[:8] == b"\x89PNG\r\n\x1a\n":
                return bytes(raw)
    return None


def archive_or_file_root(path: Path, extract_root: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path.is_dir():
        return path, None
    if path.is_file() and zipfile.is_zipfile(path):
        if extract_root.exists():
            return extract_root, None
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                member_path = (extract_root / member.filename).resolve()
                if not member_path.is_relative_to(extract_root.resolve()):
                    raise ValueError(f"unsafe zip path: {member.filename}")
                archive.extract(member, extract_root)
        return extract_root, None
    if path.is_file() and path.suffix.lower() == ".png":
        tmp = tempfile.TemporaryDirectory()
        target = Path(tmp.name) / path.name
        shutil.copy2(path, target)
        return Path(tmp.name), tmp
    raise ValueError(f"source path must be a directory, zip, or png: {path}")


def extract_archives(root: Path, out: Path) -> Path:
    marker = out / ".complete"
    if marker.exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    found = False
    for archive_path in sorted(root.rglob("*.zip")):
        found = True
        target_root = out / archive_path.stem
        target_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = (target_root / member.filename).resolve()
                if not member_path.is_relative_to(target_root.resolve()):
                    raise ValueError(f"unsafe zip path: {member.filename}")
                archive.extract(member, target_root)
    if found:
        marker.write_text("ok\n", encoding="utf-8")
    return out if found else root


def validate_png(path: Path, *, max_size: int, max_colors: int, max_pixel_art_colors: int) -> str | None:
    try:
        with Image.open(path) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width > max_size or height > max_size:
                return "oversized"
            image.load()
            encoding = image_to_indices(image, max_colors=max_colors)
            if encoding.unique_color_count > max_pixel_art_colors:
                return "non_pixel_art"
    except Image.DecompressionBombError:
        return "oversized"
    except (OSError, UnidentifiedImageError, ValueError):
        return "corrupt"
    return None


def source_metadata(source: dict[str, Any], rel: Path, target: Path) -> dict[str, Any]:
    tags = sorted(set(source.get("tags") or []) | {token for token in rel.stem.replace("-", "_").lower().split("_") if token and not token.isdigit()})
    return {
        "source": source["name"],
        "source_url": source.get("source_url") or source.get("url") or source.get("path"),
        "license": source.get("license"),
        "author": source.get("author"),
        "category": source.get("category"),
        "tags": tags,
        "provenance": str(rel).replace("\\", "/"),
        "corpus_rel_path": str(target.relative_to(target.parents[2])).replace("\\", "/") if len(target.parents) > 2 else target.name,
    }


def curate_manifest_rows(rows: list[dict[str, Any]], *, target_count: int, per_bucket_cap: int | None, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed = [
        row for row in rows
        if not row.get("lossy")
        and not row.get("duplicate_of")
        and not row.get("near_duplicate_of")
        and is_allowed_license(str(row.get("license") or ""))
    ]
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in allowed:
        meta = row.get("metadata") or {}
        key = f"{meta.get('category') or 'uncategorized'}|{row.get('bucket')}"
        by_group.setdefault(key, []).append(row)
    rng = random.Random(seed)
    for group in by_group.values():
        rng.shuffle(group)
    cap = per_bucket_cap or max(1, (target_count + max(1, len(by_group)) - 1) // max(1, len(by_group)))
    balanced = []
    for key in sorted(by_group):
        balanced.extend(by_group[key][:cap])
    if len(balanced) > target_count:
        rng.shuffle(balanced)
        balanced = balanced[:target_count]
    balanced.sort(key=lambda row: row["rel_path"])
    return balanced, {
        "eligible": len(allowed),
        "selected": len(balanced),
        "dropped_lossy": sum(1 for row in rows if row.get("lossy")),
        "dropped_duplicates": sum(1 for row in rows if row.get("duplicate_of")),
        "dropped_near_duplicates": sum(1 for row in rows if row.get("near_duplicate_of")),
        "groups": {key: len(value) for key, value in sorted(by_group.items())},
        "per_group_cap": cap,
    }


def curate_provisional_corpus(
    input_root: str | Path,
    output_root: str | Path,
    *,
    max_size: int = 128,
    max_colors: int = 64,
    max_pixel_art_colors: int = 256,
    near_duplicate_hamming: int = 4,
) -> dict[str, Any]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    training_root = output_root / "training_pool"
    holdout_root = output_root / "holdout"
    accepted_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    seen_bytes: dict[str, str] = {}
    seen_phash = _PhashIndex()
    report: dict[str, Any] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "policy": {
            "max_size": max_size,
            "max_colors": max_colors,
            "max_pixel_art_colors": max_pixel_art_colors,
            "near_duplicate_hamming": near_duplicate_hamming,
        },
        "scanned": 0,
        "accepted": 0,
        "sprite_sheets_split": 0,
        "split_children": 0,
        "duplicates": 0,
        "near_duplicates": 0,
        "non_pixel_art_smooth": 0,
        "oversized_high_res": 0,
        "lossy_gt_64_colors": 0,
        "corrupt_unsupported": 0,
        "rejection_reasons": {},
        "flags": {},
    }

    output_root.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(p for p in input_root.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(input_root).parts)):
        if source_path.suffix.lower() == ".json":
            continue
        report["scanned"] += 1
        if report["scanned"] == 1 or report["scanned"] % 1000 == 0:
            print(
                f"curate: scanned={report['scanned']} accepted={len(accepted_rows)} holdout={len(holdout_rows)}",
                file=sys.stderr,
                flush=True,
            )
        rel = source_path.relative_to(input_root)
        metadata = _read_sidecar(source_path)
        if source_path.suffix.lower() != ".png":
            _holdout_file(source_path, holdout_root / "corrupt_unsupported" / _safe_output_rel(rel), "unsupported_format", holdout_rows, report, input_root, metadata)
            continue

        try:
            with Image.open(source_path) as image:
                image.load()
                rgba = image.convert("RGBA")
        except (OSError, UnidentifiedImageError, ValueError):
            _holdout_file(source_path, holdout_root / "corrupt_unsupported" / _safe_output_rel(rel), "corrupt", holdout_rows, report, input_root, metadata)
            continue

        pieces = _sprite_sheet_pieces(rgba, max_size=max_size) if max(rgba.size) > max_size else []
        if pieces:
            report["sprite_sheets_split"] += 1
            report["split_children"] += len(pieces)
            for index, (box, piece) in enumerate(pieces):
                piece_rel = Path("_splits") / rel.with_suffix("") / f"{index:04d}_{box[0]}_{box[1]}_{box[2] - box[0]}x{box[3] - box[1]}.png"
                piece_meta = dict(metadata)
                piece_meta["split_from"] = str(rel).replace("\\", "/")
                piece_meta["split_box"] = list(box)
                _process_candidate(piece, piece_rel, training_root, holdout_root, accepted_rows, holdout_rows, report, seen_bytes, seen_phash, max_size, max_colors, max_pixel_art_colors, near_duplicate_hamming, piece_meta)
            continue

        _process_candidate(rgba, rel, training_root, holdout_root, accepted_rows, holdout_rows, report, seen_bytes, seen_phash, max_size, max_colors, max_pixel_art_colors, near_duplicate_hamming, metadata, source_path=source_path)

    save_dataset_manifest(accepted_rows, output_root / "manifest.provisional.jsonl")
    save_dataset_manifest(holdout_rows, output_root / "holdout_manifest.jsonl")
    report["accepted"] = len(accepted_rows)
    report["holdout"] = len(holdout_rows)
    report["rejection_reasons"] = dict(sorted(report["rejection_reasons"].items()))
    report["flags"] = dict(sorted(report["flags"].items()))
    (output_root / "report.provisional.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _process_candidate(
    image: Image.Image,
    rel: Path,
    training_root: Path,
    holdout_root: Path,
    accepted_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    report: dict[str, Any],
    seen_bytes: dict[str, str],
    seen_phash: _PhashIndex,
    max_size: int,
    max_colors: int,
    max_pixel_art_colors: int,
    near_duplicate_hamming: int,
    metadata: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> None:
    output_rel = _safe_output_rel(rel)
    width, height = image.size
    reason, unique_color_count, palette_size = _candidate_rejection(image, max_size=max_size, max_colors=max_colors, max_pixel_art_colors=max_pixel_art_colors)
    if reason:
        _holdout_image(image, holdout_root / reason / output_rel, reason, holdout_rows, report, rel, metadata, unique_color_count, palette_size, source_path=source_path)
        return

    payload = source_path.read_bytes() if source_path is not None else _png_bytes(image)
    digest = hashlib.sha256(payload).hexdigest()
    phash = _perceptual_hash(image)
    duplicate_of = seen_bytes.get(digest)
    if duplicate_of:
        report["duplicates"] += 1
        _holdout_image(image, holdout_root / "duplicate" / output_rel, "duplicate", holdout_rows, report, rel, metadata, unique_color_count, palette_size, duplicate_of=duplicate_of, source_path=source_path)
        return
    near_of = seen_phash.find(phash, max_hamming=near_duplicate_hamming)
    if near_of:
        report["near_duplicates"] += 1
        _holdout_image(image, holdout_root / "near_duplicate" / output_rel, "near_duplicate", holdout_rows, report, rel, metadata, unique_color_count, palette_size, near_duplicate_of=near_of, source_path=source_path)
        return

    target = training_root / output_rel
    _write_candidate(image, target, source_path=source_path)
    meta = _curated_metadata(metadata, "accepted", source_path, unique_color_count, palette_size)
    target.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    row = _manifest_row(target, training_root.parent, rel, width, height, digest, phash, unique_color_count, palette_size, meta)
    accepted_rows.append(row)
    seen_bytes[digest] = str(target)
    seen_phash.add(phash, str(target))


def _candidate_rejection(image: Image.Image, *, max_size: int, max_colors: int, max_pixel_art_colors: int) -> tuple[str | None, int, int]:
    width, height = image.size
    if width <= 0 or height <= 0 or width > max_size or height > max_size:
        return "oversized_high_res", 0, 0
    unique_color_count, palette_size, lossy = _palette_manifest_stats(image, max_colors=max_colors)
    if unique_color_count > max_pixel_art_colors:
        return "non_pixel_art_smooth", unique_color_count, palette_size
    if lossy:
        return "lossy_gt_64_colors", unique_color_count, palette_size
    return None, unique_color_count, palette_size


def _sprite_sheet_pieces(image: Image.Image, *, max_size: int) -> list[tuple[tuple[int, int, int, int], Image.Image]]:
    if image.width * image.height > 1_000_000:
        return []
    arr = np.asarray(image, dtype=np.uint8)
    alpha_mask = arr[..., 3] > 0
    if alpha_mask.any() and not alpha_mask.all():
        boxes = _component_boxes(alpha_mask, max_size=max_size)
    else:
        background = arr[0, 0]
        flat_mask = np.any(arr != background, axis=2)
        boxes = _component_boxes(flat_mask, max_size=max_size)
    if len(boxes) < 2:
        return []
    return [(box, image.crop(box)) for box in boxes]


def _component_boxes(mask: np.ndarray, *, max_size: int) -> list[tuple[int, int, int, int]]:
    try:
        import cv2

        labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        boxes = []
        for label in range(1, labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= 4 and width <= max_size and height <= max_size:
                boxes.append((x, y, x + width, y + height))
        return sorted(boxes, key=lambda box: (box[1], box[0]))
    except Exception:
        pass

    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=np.bool_)
    boxes: list[tuple[int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for ny in range(cy - 1, cy + 2):
                    for nx in range(cx - 1, cx + 2):
                        if (nx, ny) == (cx, cy):
                            continue
                        if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((nx, ny))
            box = (min_x, min_y, max_x + 1, max_y + 1)
            if area >= 4 and box[2] - box[0] <= max_size and box[3] - box[1] <= max_size:
                boxes.append(box)
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def _holdout_file(source: Path, target: Path, reason: str, rows: list[dict[str, Any]], report: dict[str, Any], root: Path, metadata: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _link_or_copy(source, target)
    row = {
        "path": str(target),
        "rel_path": str(target.relative_to(target.parents[2])).replace("\\", "/") if len(target.parents) > 2 else target.name,
        "source_path": str(source),
        "source_rel_path": str(source.relative_to(root)).replace("\\", "/"),
        "reason": reason,
        "metadata": _curated_metadata(metadata, reason, source, 0, 0),
    }
    rows.append(row)
    _count_rejection(report, reason)


def _holdout_image(
    image: Image.Image,
    target: Path,
    reason: str,
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    rel: Path,
    metadata: dict[str, Any],
    unique_color_count: int,
    palette_size: int,
    *,
    duplicate_of: str | None = None,
    near_duplicate_of: str | None = None,
    source_path: Path | None = None,
) -> None:
    _write_candidate(image, target, source_path=source_path)
    meta = _curated_metadata(metadata, reason, source_path, unique_color_count, palette_size)
    target.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(source_path.read_bytes() if source_path is not None else target.read_bytes()).hexdigest()
    phash = _perceptual_hash(image)
    rows.append(
        {
            "path": str(target),
            "rel_path": str(target.relative_to(target.parents[2])).replace("\\", "/") if len(target.parents) > 2 else target.name,
            "source_path": str(source_path) if source_path else None,
            "source_rel_path": str(rel).replace("\\", "/"),
            "reason": reason,
            "width": image.width,
            "height": image.height,
            "sha256": digest,
            "phash": phash,
            "duplicate_of": duplicate_of,
            "near_duplicate_of": near_duplicate_of,
            "unique_color_count": unique_color_count,
            "palette_size": palette_size,
            "metadata": meta,
        }
    )
    _count_rejection(report, reason)


def _manifest_row(
    path: Path,
    root: Path,
    rel: Path,
    width: int,
    height: int,
    digest: str,
    phash: str,
    unique_color_count: int,
    palette_size: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "path": str(path),
        "rel_path": str(path.relative_to(root)).replace("\\", "/"),
        "source_rel_path": str(rel).replace("\\", "/"),
        "width": width,
        "height": height,
        "bucket_size": _bucket_size(width, height),
        "aspect_bucket": _aspect_bucket(width, height),
        "bucket": bucket_id(width, height),
        "sha256": digest,
        "phash": phash,
        "duplicate_of": None,
        "near_duplicate_of": None,
        "source": metadata.get("source"),
        "source_url": metadata.get("source_url"),
        "license": metadata.get("license"),
        "metadata": metadata,
        "lossy": False,
        "unique_color_count": unique_color_count,
        "palette_size": palette_size,
    }


def _curated_metadata(metadata: dict[str, Any], status: str, source_path: Path | None, unique_color_count: int, palette_size: int) -> dict[str, Any]:
    out = dict(metadata)
    out["curation_status"] = status
    out["source_path"] = str(source_path) if source_path else out.get("source_path")
    out["unique_color_count"] = unique_color_count
    out["palette_size_used"] = palette_size
    return out


def _write_candidate(image: Image.Image, target: Path, *, source_path: Path | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_path is not None:
        _link_or_copy(source_path, target)
    else:
        image.save(target)


def _safe_output_rel(rel: Path) -> Path:
    rel_text = str(rel).replace("\\", "/")
    if len(rel_text) <= 180 and all(len(part) <= 80 for part in rel.parts):
        return rel
    digest = hashlib.sha256(rel_text.encode("utf-8")).hexdigest()[:24]
    suffix = rel.suffix.lower() or ".bin"
    stem = slug(rel.stem)[:48]
    return Path("_long_paths") / f"{stem}_{digest}{suffix}"


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except FileExistsError:
        return
    except OSError:
        shutil.copy2(source, target)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _read_sidecar(path: Path) -> dict[str, Any]:
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _count_rejection(report: dict[str, Any], reason: str) -> None:
    report["rejection_reasons"][reason] = report["rejection_reasons"].get(reason, 0) + 1
    report["flags"][reason] = report["flags"].get(reason, 0) + 1
    if reason in report:
        report[reason] += 1
    if reason in {"corrupt", "unsupported_format"}:
        report["corrupt_unsupported"] += 1


def compose_scene_samples(root: Path, count: int, *, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    out_dir = root / "procedural_scenes"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        out_dir / f"scene_{index:06d}.png"
        for index in range(count)
        if (out_dir / f"scene_{index:06d}.png").exists() and (out_dir / f"scene_{index:06d}.json").exists()
    ]
    if len(existing) == count:
        return {"created": count, "new": 0, "reused": count, "source_assets": None}
    candidates = [path for path in root.rglob("*.png") if path.with_suffix(".json").exists()]
    transparent = []
    for path in candidates:
        with Image.open(path) as image:
            alpha = np.asarray(image.convert("RGBA"))[..., 3]
        if (alpha < 255).any():
            transparent.append(path)
    if len(transparent) < 2:
        return {"created": 0, "reason": "need at least two transparent assets"}
    created = 0
    reused = 0
    for index in range(count):
        target = out_dir / f"scene_{index:06d}.png"
        if target.exists() and target.with_suffix(".json").exists():
            reused += 1
            continue
        canvas = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        chosen = rng.sample(transparent, min(6, len(transparent)))
        parents = []
        parent_licenses = []
        for asset in chosen:
            with Image.open(asset) as image:
                sprite = image.convert("RGBA")
            x = rng.randrange(0, max(1, 128 - sprite.width + 1))
            y = rng.randrange(0, max(1, 128 - sprite.height + 1))
            canvas.alpha_composite(sprite, (x, y))
            parents.append(str(asset.relative_to(root)).replace("\\", "/"))
            metadata = json.loads(asset.with_suffix(".json").read_text(encoding="utf-8"))
            parent_licenses.append(str(metadata.get("license") or ""))
        canvas.save(target)
        licenses = sorted(set(parent_licenses))
        target.with_suffix(".json").write_text(
            json.dumps(
                {
                    "source": "procedural_composition",
                    "source_url": "local",
                    "license": licenses[0] if len(licenses) == 1 else "mixed permissive",
                    "category": "scene",
                    "tags": ["scene", "procedural"],
                    "provenance": parents,
                    "parent_licenses": licenses,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        created += 1
    return {"created": created + reused, "new": created, "reused": reused, "source_assets": len(transparent)}


def is_allowed_license(text: str) -> bool:
    norm = normalize_license(text)
    if not norm:
        return False
    if any(part in norm for part in DENIED_LICENSE_PARTS):
        return False
    return norm in ALLOWED_LICENSES or any(norm.startswith(prefix) for prefix in ("cc-by ", "cc by "))


def normalize_license(text: str) -> str:
    return " ".join(text.strip().lower().replace("_", "-").split())


def unique_target(path: Path) -> Path:
    return path


def slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    return out or "source"


def _reject(report: dict[str, Any], reason: str) -> None:
    report["rejected"] += 1
    report["rejection_reasons"][reason] = report["rejection_reasons"].get(reason, 0) + 1
