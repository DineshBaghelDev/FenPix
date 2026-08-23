from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler, Subset

from .palette import image_to_indices


BUCKET_SIZES = (32, 64, 128)


def _bucket_size(width: int, height: int, bucket_sizes: tuple[int, ...] = BUCKET_SIZES) -> int:
    longest = max(width, height)
    for size in bucket_sizes:
        if longest <= size:
            return size
    raise ValueError(f"image {width}x{height} is larger than supported buckets {bucket_sizes}")


def _aspect_bucket(width: int, height: int) -> str:
    ratio = width / height
    if ratio >= 4 / 3:
        return "landscape"
    if ratio <= 3 / 4:
        return "portrait"
    return "square"


def bucket_id(width: int, height: int, bucket_sizes: tuple[int, ...] = BUCKET_SIZES) -> str:
    return f"{_bucket_size(width, height, bucket_sizes)}:{_aspect_bucket(width, height)}"


class PixelArtDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        max_colors: int = 64,
        *,
        cache_dir: str | Path | None = None,
        cache: bool = False,
        bucket_sizes: tuple[int, ...] = BUCKET_SIZES,
    ):
        self.root = Path(root)
        self.max_colors = max_colors
        self.bucket_sizes = tuple(sorted(bucket_sizes))
        self.cache_dir = Path(cache_dir) if cache_dir is not None else (self.root / ".fenpix_cache" if cache else None)
        self.paths = sorted(self.root.rglob("*.png"))
        if not self.paths:
            raise ValueError(f"no PNG files found under {self.root}")
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.paths)

    def _cache_path(self, path: Path) -> Path:
        stat = path.stat()
        key = f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{self.max_colors}"
        return self.cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.npz"  # type: ignore[operator]

    def _encoding(self, path: Path):
        if self.cache_dir is None:
            with Image.open(path) as image:
                return image_to_indices(image, max_colors=self.max_colors)

        cache_path = self._cache_path(path)
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            return {
                "indices": cached["indices"],
                "palette": cached["palette"],
                "width": int(cached["width"]),
                "height": int(cached["height"]),
                "lossy": bool(cached["lossy"]),
                "unique_color_count": int(cached["unique_color_count"]),
            }

        with Image.open(path) as image:
            encoding = image_to_indices(image, max_colors=self.max_colors)
        np.savez_compressed(
            cache_path,
            indices=encoding.indices,
            palette=encoding.palette,
            width=np.array(encoding.width, dtype=np.int32),
            height=np.array(encoding.height, dtype=np.int32),
            lossy=np.array(encoding.lossy, dtype=np.bool_),
            unique_color_count=np.array(encoding.unique_color_count, dtype=np.int32),
        )
        return encoding

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        encoding = self._encoding(path)

        meta_path = path.with_suffix(".json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        indices = encoding["indices"] if isinstance(encoding, dict) else encoding.indices
        palette = encoding["palette"] if isinstance(encoding, dict) else encoding.palette
        width = encoding["width"] if isinstance(encoding, dict) else encoding.width
        height = encoding["height"] if isinstance(encoding, dict) else encoding.height
        lossy = encoding["lossy"] if isinstance(encoding, dict) else encoding.lossy
        unique_color_count = (
            encoding["unique_color_count"] if isinstance(encoding, dict) else encoding.unique_color_count
        )
        bucket_size = _bucket_size(width, height, self.bucket_sizes)
        aspect_bucket = _aspect_bucket(width, height)

        return {
            "path": str(path),
            "indices": torch.from_numpy(indices).long(),
            "structure_indices": torch.from_numpy(indices).long(),
            "palette": torch.from_numpy(palette).to(torch.uint8),
            "size": torch.tensor([width, height], dtype=torch.int32),
            "dimensions": {"width": width, "height": height},
            "palette_size": torch.tensor(len(palette), dtype=torch.int16),
            "unique_color_count": torch.tensor(unique_color_count, dtype=torch.int16),
            "lossy": bool(lossy),
            "bucket_size": torch.tensor(bucket_size, dtype=torch.int16),
            "aspect_bucket": aspect_bucket,
            "bucket": f"{bucket_size}:{aspect_bucket}",
            "valid_mask": torch.ones((height, width), dtype=torch.bool),
            "metadata": metadata,
        }


def train_validation_split(
    dataset: Dataset,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[Subset, Subset]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    validation_count = max(1, round(len(order) * validation_fraction))
    return Subset(dataset, order[validation_count:]), Subset(dataset, order[:validation_count])


def train_val_test_split(
    dataset: Dataset,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[Subset, Subset, Subset]:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be positive and sum to less than 1")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    test_count = max(1, round(len(order) * test_fraction))
    validation_count = max(1, round(len(order) * validation_fraction))
    return (
        Subset(dataset, order[test_count + validation_count :]),
        Subset(dataset, order[test_count : test_count + validation_count]),
        Subset(dataset, order[:test_count]),
    )


def filtered_indices(
    dataset: Dataset,
    *,
    max_bucket_size: int | None = None,
    include_lossy: bool = False,
    limit: int | None = None,
) -> list[int]:
    keep = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if max_bucket_size is not None and int(sample["bucket_size"]) > max_bucket_size:
            continue
        if sample["lossy"] and not include_lossy:
            continue
        keep.append(index)
    return keep[:limit] if limit else keep


def split_report(train: Dataset, validation: Dataset, test: Dataset) -> dict[str, Any]:
    def row(dataset: Dataset) -> dict[str, int]:
        lossy = sum(1 for index in range(len(dataset)) if dataset[index]["lossy"])
        return {"count": len(dataset), "lossy": lossy, "lossless": len(dataset) - lossy}

    return {"train": row(train), "validation": row(validation), "test": row(test)}


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: Dataset, batch_size: int, seed: int = 0, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last

    def __iter__(self):
        buckets: dict[str, list[int]] = {}
        for index in range(len(self.dataset)):
            sample = self.dataset[index]
            buckets.setdefault(sample["bucket"], []).append(index)

        generator = torch.Generator().manual_seed(self.seed)
        batches: list[list[int]] = []
        for key in sorted(buckets):
            order = torch.randperm(len(buckets[key]), generator=generator).tolist()
            shuffled = [buckets[key][i] for i in order]
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        order = torch.randperm(len(batches), generator=generator).tolist()
        for index in order:
            yield batches[index]

    def __len__(self) -> int:
        buckets: dict[str, int] = {}
        for index in range(len(self.dataset)):
            sample = self.dataset[index]
            buckets[sample["bucket"]] = buckets.get(sample["bucket"], 0) + 1
        if self.drop_last:
            return sum(count // self.batch_size for count in buckets.values())
        return sum((count + self.batch_size - 1) // self.batch_size for count in buckets.values())


def pixel_art_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")

    height = max(sample["indices"].shape[0] for sample in samples)
    width = max(sample["indices"].shape[1] for sample in samples)
    palette_size = max(int(sample["palette_size"]) for sample in samples)
    batch_size = len(samples)

    indices = torch.full((batch_size, height, width), -1, dtype=torch.long)
    masks = torch.zeros((batch_size, height, width), dtype=torch.bool)
    palettes = torch.zeros((batch_size, palette_size, 4), dtype=torch.uint8)
    palette_masks = torch.zeros((batch_size, palette_size), dtype=torch.bool)

    for row, sample in enumerate(samples):
        sample_height, sample_width = sample["indices"].shape
        sample_palette_size = int(sample["palette_size"])
        indices[row, :sample_height, :sample_width] = sample["indices"]
        masks[row, :sample_height, :sample_width] = sample["valid_mask"]
        palettes[row, :sample_palette_size] = sample["palette"]
        palette_masks[row, :sample_palette_size] = True

    return {
        "path": [sample["path"] for sample in samples],
        "indices": indices,
        "structure_indices": indices.clone(),
        "palette": palettes,
        "palette_mask": palette_masks,
        "valid_mask": masks,
        "size": torch.stack([sample["size"] for sample in samples]),
        "palette_size": torch.stack([sample["palette_size"] for sample in samples]),
        "unique_color_count": torch.stack([sample["unique_color_count"] for sample in samples]),
        "lossy": torch.tensor([sample["lossy"] for sample in samples], dtype=torch.bool),
        "bucket_size": torch.stack([sample["bucket_size"] for sample in samples]),
        "aspect_bucket": [sample["aspect_bucket"] for sample in samples],
        "bucket": [sample["bucket"] for sample in samples],
        "dimensions": [sample["dimensions"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
    }


def quality_score(sample: dict[str, Any]) -> float:
    indices = sample["indices"].numpy() if isinstance(sample["indices"], torch.Tensor) else sample["indices"]
    palette_size = int(sample["palette_size"])
    unique = int(sample["unique_color_count"])
    height, width = indices.shape
    edge_h = indices[1:] != indices[:-1]
    edge_w = indices[:, 1:] != indices[:, :-1]
    edge_density = (edge_h.sum() + edge_w.sum()) / max(1, edge_h.size + edge_w.size)
    palette_fit = min(1.0, unique / max(1, palette_size))
    size_fit = min(1.0, max(height, width) / max(BUCKET_SIZES))
    lossy_penalty = 0.25 if sample["lossy"] else 0.0
    return float(max(0.0, min(1.0, 0.35 * palette_fit + 0.35 * edge_density + 0.30 * size_fit - lossy_penalty)))


def dataset_quality_report(dataset: Dataset) -> dict[str, Any]:
    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    scores: list[float] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        path = Path(sample["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            duplicates.append({"path": str(path), "duplicate_of": seen[digest]})
        else:
            seen[digest] = str(path)
        scores.append(quality_score(sample))
    return {
        "count": len(dataset),
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "mean_quality": float(sum(scores) / max(1, len(scores))),
        "min_quality": float(min(scores) if scores else 0.0),
        "max_quality": float(max(scores) if scores else 0.0),
    }
