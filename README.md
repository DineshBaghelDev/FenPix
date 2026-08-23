# FenPix

Tiny all-rounder pixel-art generation research repo.

This first checkpoint does not train a model yet. It establishes the image
representation the model will use:

- load native PNG/RGBA files without resizing
- extract a compact palette, including transparency
- convert pixels to palette indices
- reconstruct exact RGBA PNGs when the palette budget covers the image
- keep variable sizes through a PyTorch `Dataset`

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

`PixelArtDataset` returns one sample per PNG:

```python
{
    "path": "...",
    "indices": LongTensor[height, width],
    "palette": UInt8Tensor[colors, 4],
    "size": IntTensor[width, height],
    "palette_size": IntTensor[],
    "metadata": dict,
}
```

Variable sizes are preserved, so use `pixel_art_collate` when batching.

## Roadmap

See [MILESTONES.md](MILESTONES.md) for the full milestone map.

1. Dataset pipeline
2. Palette extraction + indexing
3. Structure tokenizer/VQ
4. Basic MaskGIT
5. Text conditioning
6. Variable aspect ratios
7. Hierarchical 32 -> 64 -> 128 generation
8. Flow refiner over palette/index logits
9. Joint fine-tuning
10. Teacher model
11. Distillation
12. Quantization + inference optimization
