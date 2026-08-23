# FenPix

Tiny all-rounder pixel-art generation research repo.

This first checkpoint does not train a model yet. It establishes the image
representation the model will use:

- load native PNG/RGBA files without resizing
- extract a compact palette, including transparency
- convert pixels to palette indices
- reconstruct exact RGBA PNGs when the source has <=64 unique RGBA colors
- batch variable-size indexed images with padding masks and buckets

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Run Tests

```bash
python -m unittest discover -s tests
```

## Visualize A Roundtrip

```bash
python examples/visualize_roundtrip.py path\to\image.png --out roundtrip.png
```

## Dataset Shape

Put PNG files under a dataset folder. Optional metadata can sit next to each
image as `same-name.json`.

```text
data/
  sprite.png
  sprite.json
```

Example metadata:

```json
{
  "caption": "tiny red knight idle sprite",
  "category": "sprite",
  "perspective": "side",
  "palette_size": 16,
  "transparency": true,
  "source": "cc0"
}
```

`PixelArtDataset` returns one native-size sample per PNG:

```python
{
    "path": "...",
    "indices": LongTensor[height, width],
    "structure_indices": LongTensor[height, width],
    "palette": UInt8Tensor[colors, 4],
    "size": IntTensor[width, height],
    "valid_mask": BoolTensor[height, width],
    "bucket": "64:landscape",
    "bucket_size": IntTensor[],
    "aspect_bucket": "landscape",
    "palette_size": IntTensor[],
    "metadata": dict,
}
```

Variable sizes are preserved. Use `pixel_art_collate` to stack batches; it pads
only to the largest native size in the batch and exposes `valid_mask` plus
`palette_mask`. Use `BucketBatchSampler` for deterministic 32/64/128
resolution and aspect-ratio buckets, and `train_validation_split` for seeded
train/validation subsets.

## Prepare Kenney Assets

```bash
python scripts/prepare_kenney.py data/raw data/processed
```

The processed directory must be empty. It mirrors the input folder or zip paths,
copies accepted PNG files unchanged, writes matching `.json` sidecars, and
writes `report.json`.

## Roadmap

See [MILESTONES.md](MILESTONES.md) for the full milestone map.

1. Dataset pipeline
2. Palette extraction + indexing
3. Structure tokenizer/VQ, separate from color assignment
4. Palette-conditioned MaskGIT
5. Text conditioning
6. Variable aspect ratios
7. Hierarchical 32 -> 64 -> 128 generation
8. Flow/logit refiner ablation
9. Joint fine-tuning
10. Teacher model
11. Distillation
12. Quantization + inference optimization
