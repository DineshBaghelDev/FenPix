# M8.7 Context: Direct Structure Generation

Date: 2026-08-24

## Goal

Replace the 32->64->128 generated structure hierarchy with a direct target-resolution
structure generator, avoiding hierarchy exposure bias while preserving FenPix's
discrete/palette-compatible color contract.

Chosen architecture:

text + valid native canvas -> direct masked conv structure generator -> discrete
canonical structure map -> color stage

## Implemented

- Added `fenpix/direct_structure.py`.
- Added `DirectStructureGenerator` with:
  - discrete structure token embeddings
  - row/column positional embeddings
  - small dilated convolutional residual blocks
  - masked structure-token head
  - auxiliary occupancy, boundary, and component-count heads
- Structure representation stays simple:
  - `0` = transparent/background
  - `1..N` = canonical connected foreground regions from `pixel_art_collate`
  - `vocab_size` = mask
  - `vocab_size + 1` = pad
- Added `scripts/train_direct_structure.py` for direct structure train/sample.
- Added `--direct-structure` to `scripts/evaluate_quality.py`.
- Added `--direct-structure-targets` and `--direct-structure` sampling support to
  `scripts/train_color.py`, so color can be retrained fairly against direct
  canonical structure maps.
- Exported `DirectStructureConfig` and `DirectStructureGenerator` from
  `fenpix/__init__.py`.
- Added tests in `tests/test_direct_structure.py`.

## Losses

- Masked structure CE with foreground/boundary weighting.
- Occupancy BCE auxiliary loss.
- Boundary BCE auxiliary loss.
- Component-count CE auxiliary loss.
- No hard occupancy gate in sampling; M8.6 showed hard occupancy greatly reduced
  fragmentation but damaged silhouette/boundary quality.

Default small-model shape:

- `hidden_dim=96`
- `depth=8`
- `text_dim=64`
- `structure_vocab_size=128`
- max native canvas `128x128`

## Commands Run

Focused tests:

```powershell
python -m pytest tests\test_direct_structure.py -q
python -m pytest tests\test_hierarchy.py tests\test_color.py tests\test_direct_structure.py -q
```

Full tests:

```powershell
python -m pytest -q
```

Result: `75 passed`.

Tiny direct-structure training smoke:

```powershell
python scripts\train_direct_structure.py train data\processed_m8_2 --checkpoint runs\m8_7_direct_smoke.pt --log runs\m8_7_direct_smoke.jsonl --viz runs\m8_7_direct_smoke.png --device cpu --epochs 1 --batch-size 2 --limit 8 --max-size 64 --hidden-dim 16 --depth 1 --text-dim 8 --text-provider tiny --embedding-cache runs\m8_7_direct_smoke_text_cache.pt --samples 2 --width 32 --height 32 --steps 1
```

Result:

- `validation_loss`: 5.626181602478027
- `validation_token_error`: 0.99755859375
- smoke only; not a quality result

Tiny end-to-end direct-structure eval smoke:

```powershell
python scripts\evaluate_quality.py data\processed_m8_2 --color-checkpoint runs\m8_3_64_color.pt --direct-structure runs\m8_7_direct_smoke.pt --metrics runs\m8_7_direct_smoke_metrics.json --gallery runs\m8_7_direct_smoke_gallery.png --device cpu --limit 2 --max-size 64 --width 32 --height 32 --batch-size 1 --steps 1 --structure-steps 1 --text-provider tiny --alignment-provider tiny --split validation
```

Result:

- completed end to end with `structure_source=direct`
- smoke only; old M3-conditioned color checkpoint is not a fair quality pairing

Tiny direct-structure color-target training smoke:

```powershell
python scripts\train_color.py train data\processed_m8_2 --direct-structure-targets --checkpoint runs\m8_7_direct_color_smoke.pt --log runs\m8_7_direct_color_smoke.jsonl --viz runs\m8_7_direct_color_smoke.png --device cpu --epochs 1 --batch-size 2 --limit 8 --max-size 64 --hidden-dim 16 --depth 1 --text-dim 8 --text-provider tiny --embedding-cache runs\m8_7_direct_color_smoke_text_cache.pt --steps 1
```

Result:

- `validation_loss`: 6.482417106628418
- smoke only; proves color can train on direct structure targets

## 64px Quality Validation

Ran the full 5k lossless 64px validation gate on CUDA with seed `0`, CLIP text
conditioning, CLIP alignment, and no GT structure leakage.

Training:

```powershell
python scripts\train_direct_structure.py train data\processed_m8_2 --checkpoint runs\m8_7_direct_structure_64.pt --log runs\m8_7_direct_structure_64.jsonl --viz runs\m8_7_direct_structure_64.png --device cuda --epochs 20 --batch-size 4 --limit 5000 --max-size 64 --hidden-dim 96 --depth 8 --text-dim 64 --text-provider clip --embedding-cache runs\m8_5_mixed_only_text_cache.pt --samples 4 --width 64 --height 64 --steps 8 --seed 0 --cache
python scripts\train_color.py train data\processed_m8_2 --direct-structure-targets --checkpoint runs\m8_7_direct_color_64.pt --log runs\m8_7_direct_color_64.jsonl --viz runs\m8_7_direct_color_64.png --device cuda --epochs 20 --batch-size 4 --limit 5000 --max-size 64 --stage 64 --hidden-dim 64 --depth 2 --heads 4 --text-dim 64 --text-provider clip --embedding-cache runs\m8_5_mixed_only_text_cache.pt --steps 8 --seed 0 --cache
```

Training results:

- Direct structure epoch 20: `validation_loss=0.5709010775089264`,
  `validation_token_error=0.34470832186937334`, `peak_vram_mb=1220.0`.
- Direct-target color epoch 20: `validation_loss=1.4244115733355285`,
  `peak_vram_mb=738.0`.

Prompt-only validation comparison:

| run | transparency IoU | boundary F1 | component count error | component consistency | largest-component IoU | CLIP alignment | latency ms | VRAM MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M8.5 best hierarchy | 0.16821024655745542 | 0.05796293893505675 | 2.558 | 0.4721391882425374 | 0.14554796298143804 | 0.26004207134246826 | 80.62070129987842 | 1359.9423828125 |
| M8.6 occupancy hierarchy | 0.15840413283124166 | 0.07850395402302514 | 3.938 | 0.6130586807666317 | 0.13671902847692452 | 0.25810956954956055 | 61.29677659989102 | 1360.2734375 |
| M8.7 direct structure | 0.09996727240540365 | 0.10486492741462107 | 44.003 | 0.07297167478198983 | 0.0705007371058077 | 0.2342495173215866 | 79.48551489997772 | 1355.38330078125 |

Validation artifacts:

- `runs/m8_7_compare_m8_5_all_64_metrics.json`
- `runs/m8_7_compare_m8_5_all_64_gallery.png`
- `runs/m8_7_compare_m8_6_occ_64_metrics.json`
- `runs/m8_7_compare_m8_6_occ_64_gallery.png`
- `runs/m8_7_direct_64_validation_metrics.json`
- `runs/m8_7_direct_64_validation_gallery.png`
- `runs/m8_7_direct_structure_64.pt`
- `runs/m8_7_direct_structure_64.jsonl`
- `runs/m8_7_direct_structure_64.png`
- `runs/m8_7_direct_color_64.pt`
- `runs/m8_7_direct_color_64.jsonl`
- `runs/m8_7_direct_color_64.png`

## Decision

NO-WIN. M8.7 only improved boundary F1. It regressed transparency IoU,
component count error, component consistency, largest-component IoU, and CLIP
alignment, with no meaningful compute win. Do not make direct structure the
default architecture, and do not scale this path yet.

## Working Tree

M8.7 source files:

- `fenpix/direct_structure.py`
- `scripts/train_direct_structure.py`
- `tests/test_direct_structure.py`
- `fenpix/__init__.py`
- `scripts/evaluate_quality.py`
- `scripts/train_color.py`
- `context.md`

Generated M8.7 smoke artifacts:

- `runs/m8_7_direct_smoke.pt`
- `runs/m8_7_direct_smoke.jsonl`
- `runs/m8_7_direct_smoke.png`
- `runs/m8_7_direct_smoke_metrics.json`
- `runs/m8_7_direct_smoke_gallery.png`
- `runs/m8_7_direct_smoke_text_cache.pt`
- `runs/m8_7_direct_color_smoke.pt`
- `runs/m8_7_direct_color_smoke.jsonl`
- `runs/m8_7_direct_color_smoke.png`
- `runs/m8_7_direct_color_smoke_text_cache.pt`

---

# M8.6 Context: Explicit Silhouette/Occupancy Generation

Date: 2026-08-24

## Goal

Separate shape occupancy from structure detail:

text + lower-stage structure -> occupancy/alpha predictor -> binary silhouette mask -> structure MaskGIT constrained inside silhouette -> color stage

Dataset and scope stayed fixed:

- Used existing 5k filtered 64px dataset from `data/processed_m8_2`.
- Did not touch color, teacher, or dataset scale.
- Did not commit because the result did not clearly win.

## Implemented

- Optional dedicated 32->64 occupancy model in `fenpix/hierarchy.py`.
- BCE + Dice + differentiable boundary loss for occupancy.
- Occupancy mask sampled before 64-stage structure generation.
- Structure tokens outside the predicted foreground mask forced to background token `0`.
- Structure generation constrained to predicted foreground token positions.
- Variable aspect ratios preserved through existing `stage_native_shape` path.
- Train/eval flags:
  - `--occupancy-stage`
  - `--occupancy-loss-weight`
  - `--occupancy-boundary-weight`
  - `--occupancy-threshold`
- Tests added for occupancy gradient flow and outside-mask background forcing.

## Commands Run

Training:

```powershell
python scripts\train_hierarchy.py train data\processed_m8_2 --tokenizer runs\m8_3_64_m3_structure_tokenizer.pt --checkpoint runs\m8_6_occ_hierarchy.pt --viz runs\m8_6_occ_stages.png --device cuda --epochs 3 --batch-size 4 --limit 5000 --stages 32 64 --hidden-dim 64 --depth 2 --heads 4 --text-dim 64 --text-provider clip --embedding-cache runs\m8_5_mixed_only_text_cache.pt --log runs\m8_6_occ_hierarchy.jsonl --occupancy-stage 64 --occupancy-loss-weight 1.0 --occupancy-boundary-weight 1.0 --foreground-weight 2.0 --boundary-weight 2.0 --foreground-loss-weight 0.25 --sampled-lower-prob 0.25 --corrupt-lower-prob 0.25 --seed 0 --cache
```

Evaluation, threshold 0.5:

```powershell
python scripts\evaluate_quality.py data\processed_m8_2 --hierarchy runs\m8_6_occ_hierarchy.pt --color-checkpoint runs\m8_3_64_color.pt --tokenizer runs\m8_3_64_m3_structure_tokenizer.pt --metrics runs\m8_6_occ_64_validation_metrics.json --gallery runs\m8_6_occ_64_validation_gallery.png --device cuda --limit 5000 --max-size 64 --width 64 --height 64 --batch-size 4 --steps 4 --structure-steps 4 --split validation --text-provider clip --alignment-provider tiny --cache --seed 0
```

Evaluation, threshold 0.25:

```powershell
python scripts\evaluate_quality.py data\processed_m8_2 --hierarchy runs\m8_6_occ_hierarchy.pt --color-checkpoint runs\m8_3_64_color.pt --tokenizer runs\m8_3_64_m3_structure_tokenizer.pt --metrics runs\m8_6_occ_t025_64_validation_metrics.json --gallery runs\m8_6_occ_t025_64_validation_gallery.png --device cuda --limit 5000 --max-size 64 --width 64 --height 64 --batch-size 4 --steps 4 --structure-steps 4 --split validation --text-provider clip --alignment-provider tiny --cache --seed 0 --occupancy-threshold 0.25
```

Verification:

```powershell
python -m pytest -q
```

Result: `72 passed`.

## Training Result

Final epoch:

- `validation_loss`: 3.0292750148773195
- `validation_loss_32_64`: 2.009571493148804
- `validation_token_error_32_64`: 0.5102915561199188
- `images_per_second`: 8.671027493840693
- `peak_vram_mb`: 718.0

This was much worse than M8.5 token error around 0.154, so the quality eval was treated as the real gate.

## Metric Comparison

Validation metrics at 64px:

| run | transparency IoU | boundary F1 | component count error | component consistency | largest-component IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8.3 64 | 0.01206122466569861 | 0.036884984886538554 | n/a | 0.19104253612273334 | n/a |
| M8.5 all | 0.2369366124031327 | 0.14386012857551903 | 31.653 | 0.07825381409937614 | 0.186176534545399 |
| M8.6 occ t=0.5 | 0.15840413283124166 | 0.07850395402302514 | 3.938 | 0.6130586807666317 | 0.13671902847692452 |
| M8.6 occ t=0.25 | 0.15830654602624566 | 0.07939700269993606 | 4.001 | 0.6044865659268892 | 0.13665331472392206 |

## Decision

M8.6 failed the success criterion.

It improved fragmentation metrics strongly:

- Component count error improved from 31.653 to about 3.94.
- Component consistency improved from 0.078 to about 0.61.

But it regressed silhouette metrics versus M8.5:

- Transparency IoU fell from 0.2369 to about 0.158.
- Boundary F1 fell from 0.1439 to about 0.079.
- Largest-component IoU fell from 0.1862 to about 0.137.

Conclusion: do not commit this as a winning milestone. Stop and recommend a deeper structure-generation redesign.

## Working Tree

Modified files after M8.6 implementation:

- `fenpix/__init__.py`
- `fenpix/hierarchy.py`
- `scripts/evaluate_quality.py`
- `scripts/train_hierarchy.py`
- `tests/test_hierarchy.py`

Generated M8.6 artifacts:

- `runs/m8_6_occ_hierarchy.pt`
- `runs/m8_6_occ_hierarchy.jsonl`
- `runs/m8_6_occ_stages.png`
- `runs/m8_6_occ_64_validation_metrics.json`
- `runs/m8_6_occ_64_validation_gallery.png`
- `runs/m8_6_occ_t025_64_validation_metrics.json`
- `runs/m8_6_occ_t025_64_validation_gallery.png`

---

# M8.5 Freeze Corpus Acquisition Attempt

Date: 2026-08-24

## Goal

Freeze architecture at M8.5 and determine whether failures are data/training-limited before adding architecture variants.

Target corpus: 50k-100k usable lossless samples from licensed multi-source inputs, scalable toward 500k-1M.

## Implemented

- Reusable multi-source corpus ingestion in `fenpix/corpus.py`.
- Hugging Face dataset snapshot support via `hf_repo` sources.
- Zip extraction for archive-backed sources.
- Parquet image extraction, including Sc077y `grid` + `palette.npy` reconstruction.
- License allow/deny filtering and source provenance sidecars.
- Resumable archive/parquet extraction and procedural scene composition.
- Manifest usability fix: final manifests are written both at corpus root and under `assets/`.
- Faster manifest duplicate indexing and cheap manifest palette stats in `fenpix/dataset.py`.
- Real config at `data/corpus_sources_m8_5.json`.

## Sources Attempted

| source | license | status | raw accepted PNGs |
| --- | --- | --- | ---: |
| Kenney Existing M8.2 Corpus | Creative Commons CC0 | acquired from local processed corpus | 9,566 |
| OpenGameArt CC0 2D Art | CC0-1.0 | acquired from HF mirror, many oversized/non-pixel-art rejects | 2,860 |
| Sc077y Pixel Art Synthetic 10k | MIT | acquired and reconstructed from parquet grid + palette | 9,852 |
| DiffusionDB Pixelart | CC0-1.0 | blocked; HF snapshot stalled with only cache/control files and no image zips | 0 |
| Procedural compositions | mixed permissive from acquired parents | generated/reused | 20,000 |

Raw landed corpus under `data/fenpix_m8_5_scale/assets`: 42,278 PNGs.

## Commands Run

```powershell
python scripts\prepare_corpus.py data\corpus_sources_m8_5.json data\fenpix_m8_5_scale --target-count 100000 --min-count 50000 --compose-scenes 20000 --seed 0 --max-size 128 --max-colors 64 --max-pixel-art-colors 256
```

Finalization attempts were stopped because the raw acquired corpus was only 42,278 PNGs before curation/dedup, below the required 50k floor even in the impossible best case where every row survived.

Focused verification:

```powershell
python -m pytest tests\test_prepare_corpus.py tests\test_palette_pipeline.py -q
```

Result: `18 passed`.

## Decision

NO-GO for retraining from this corpus.

Do not claim the scale corpus is ready. Current blockers are source/count blockers, not architecture evidence:

- The raw corpus is below 50k before curation.
- DiffusionDB Pixelart did not download usable image zips in this run.
- OpenGameArt CC0 2D Art yielded only 2,860 usable <=128px PNGs after filters.
- Sc077y contributes 9,852 reconstructable MIT PNGs, but many are expected to be lossy under the current <=64 color training contract.
- No final manifest/report was produced for a training-ready corpus.
