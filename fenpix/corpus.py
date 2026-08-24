from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .dataset import build_dataset_manifest, dataset_manifest_report, load_dataset_manifest, save_dataset_manifest
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
    report = {"name": name, "scanned": 0, "accepted": 0, "rejected": 0, "rejection_reasons": {}}
    try:
        for path in sorted(p for p in source_root.rglob("*.png") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(source_root).parts)):
            report["scanned"] += 1
            reason = validate_png(path, max_size=max_size, max_colors=max_colors, max_pixel_art_colors=max_pixel_art_colors)
            if reason:
                _reject(report, reason)
                continue
            rel = path.relative_to(source_root)
            target = unique_target(paths.assets / slug(name) / rel)
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
    if "url" in source:
        paths.downloads.mkdir(parents=True, exist_ok=True)
        url = str(source["url"])
        target = paths.downloads / (source.get("file_name") or Path(urllib.parse.urlparse(url).path).name or f"{slug(source['name'])}.download")
        if refresh_downloads or not target.exists():
            urllib.request.urlretrieve(url, target)
        return archive_or_file_root(target, paths.extracted / slug(str(source["name"])))
    return archive_or_file_root(Path(source["path"]), paths.extracted / slug(str(source["name"])))


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


def validate_png(path: Path, *, max_size: int, max_colors: int, max_pixel_art_colors: int) -> str | None:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if width <= 0 or height <= 0 or width > max_size or height > max_size:
                return "oversized"
            encoding = image_to_indices(image, max_colors=max_colors)
            if encoding.unique_color_count > max_pixel_art_colors:
                return "non_pixel_art"
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


def compose_scene_samples(root: Path, count: int, *, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    candidates = [path for path in root.rglob("*.png") if path.with_suffix(".json").exists()]
    transparent = []
    for path in candidates:
        with Image.open(path) as image:
            alpha = np.asarray(image.convert("RGBA"))[..., 3]
        if (alpha < 255).any():
            transparent.append(path)
    if len(transparent) < 2:
        return {"created": 0, "reason": "need at least two transparent assets"}
    out_dir = root / "procedural_scenes"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
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
        target = out_dir / f"scene_{index:06d}.png"
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
    return {"created": count, "source_assets": len(transparent)}


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
    if not path.exists() and not path.with_suffix(".json").exists():
        return path
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return path.with_name(f"{path.stem}_{digest}{path.suffix}")


def slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    return out or "source"


def _reject(report: dict[str, Any], reason: str) -> None:
    report["rejected"] += 1
    report["rejection_reasons"][reason] = report["rejection_reasons"].get(reason, 0) + 1
