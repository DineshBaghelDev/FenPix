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

