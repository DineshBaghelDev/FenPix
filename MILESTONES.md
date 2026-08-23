# FenPix Milestone Map

FenPix is a tiny text-to-pixel-art generator. The goal is not a general image
model that happens to output pixel art. It should generate native indexed pixel
art: crisp RGBA images, flexible sizes up to 128x128, palette-aware output, and
fast inference on small consumer hardware.

## What We Are Building

The model predicts pixel art as palette indices, not blurred RGB images.

```text
prompt
  -> frozen text encoder
  -> coarse structure tokens
  -> dynamic palette
  -> hierarchical MaskGIT indices
  -> tiny flow refiner over logits
  -> argmax palette index
  -> exact RGBA PNG
```

The important bet is simple: keep the generation space discrete and
palette-aware so the final image stays sharp.

## M0: Representation Pipeline

Status: started

Goal: prove every image can become model-ready data and roundtrip back to PNG.

Deliverables:

- Load native PNG/RGBA files without resizing.
- Extract 8-64 color palettes.
- Convert RGBA pixels to palette-index tensors.
- Reconstruct indexed tensors back to RGBA PNG.
- Preserve width, height, transparency, and metadata.
- Add tests for exact roundtrip and variable sizes.

Done when:

- A folder of PNGs can be scanned.
- Each image produces `indices`, `palette`, `size`, and `metadata`.
- Reconstruction works for images inside the palette budget.

## M1: Dataset Ingestion

Status: next

Goal: create the first useful local dataset in the expected format.

Deliverables:

- Download or import one clean CC0/licensed source.
- Store original PNG/RGBA files unchanged.
- Generate sidecar metadata JSON files.
- Compute palette size, transparency, dimensions, and basic tags.
- Reject corrupt, huge, or non-pixel-art files.
- Produce a small dataset report.

First source:

- Kenney assets, because licensing is clean and structure is predictable.

Done when:

- `python scripts/prepare_kenney.py data/raw data/processed` creates a usable
  prototype dataset.

## M2: Batch Format

Status: planned

Goal: make variable-size indexed pixel art trainable.

Deliverables:

- Pad or pack variable-size samples.
- Add masks for valid pixels.
- Add aspect-ratio buckets.
- Add train/validation split.
- Cache palette-index encodings.

Done when:

- A `DataLoader` can produce stable batches for 32x32, 64x64, and 128x128
  targets.

## M3: Tiny Structure Tokenizer

Status: planned

Goal: learn compact 16x16-ish structure tokens from indexed pixel art.

Deliverables:

- Small VQ/tokenizer model.
- Reconstruction loss over palette/index targets.
- Checkpoint save/load.
- Reconstruction visualization grid.

Target size:

- 10-15M trainable parameters.

Done when:

- Reconstructions preserve silhouette, transparency, and major color regions.

## M4: Basic MaskGIT

Status: planned

Goal: generate palette indices from masked tokens without text conditioning.

Deliverables:

- Tiny transformer over indexed tokens.
- Masked-token training.
- Iterative decode sampler.
- 32x32 generation first.

Target size:

- 25-40M trainable parameters.

Done when:

- Unconditional samples look like plausible pixel-art objects or tiles.

## M5: Text Conditioning

Status: planned

Goal: make prompts control category, object, style, and composition.

Deliverables:

- Frozen small text encoder.
- Precomputed text embeddings.
- Cross-attention or conditioning tokens.
- Caption/category training fields.

Done when:

- Prompts like `red potion icon`, `isometric stone house`, and `grass tile`
  produce visibly different outputs.

## M6: Hierarchical Generation

Status: planned

Goal: generate progressively from 32x32 to 64x64 to 128x128.

Deliverables:

- 32x32 base stage.
- 64x64 refinement stage.
- 128x128 refinement stage.
- Aspect-ratio buckets.

Done when:

- The same prompt can produce icons, sprites, tiles, and small scenes at
  different native sizes.

## M7: Palette-Logit Flow Refiner

Status: planned

Goal: improve final indexed logits without RGB blur.

Deliverables:

- 1-4 step flow-matching refiner.
- Operates on palette/index logits only.
- Compare samples with and without refinement.

Target size:

- 5-10M trainable parameters.

Done when:

- Refined outputs improve edges, details, and palette consistency without
  anti-aliasing.

## M8: Quality Pass

Status: planned

Goal: improve quality-per-parameter before scaling.

Deliverables:

- Better captions/tags.
- Dataset quality scoring.
- Duplicate removal.
- Prompt eval set.
- Sample gallery.

Done when:

- The small model has a repeatable eval loop and visible quality trend.

## M9: Teacher And Distillation

Status: later

Goal: train a larger teacher, then compress behavior into the tiny student.

Deliverables:

- 150-300M teacher.
- Distillation targets.
- Student fine-tuning.

Done when:

- Student quality improves without exceeding the deployment budget.

## M10: Quantized Inference

Status: later

Goal: make FenPix cheap to run.

Deliverables:

- Quantized checkpoint.
- CPU/GPU inference script.
- Memory and latency benchmark.
- Export format decision.

Targets:

- 40-80 MB quantized.
- Ideally under 1 GB VRAM.

Done when:

- A user can run one command and generate a PNG from a prompt.

## Immediate Next Step

Build M1: a Kenney-first ingestion script that downloads/imports assets, keeps
native PNGs, writes sidecar metadata, and verifies the output with the existing
palette pipeline.
