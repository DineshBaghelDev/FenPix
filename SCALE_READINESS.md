# FenPix Scale Readiness

## Current Chosen Config

- Tokenizer: discrete structure tokenizer, downsample 4, codebook 128 until the ablation plan beats it on held-out metrics.
- Structure generator: hierarchical MaskGIT at 32, 64, and 128 with noisy lower-stage conditions during training.
- Color: convolutional indexed-color denoiser with canonical palette ordering and explicit transparent index handling.
- Text conditioning: frozen CLIP by default for training and prompt-only evaluation; `tiny` remains test/offline-only.
- Refiner: disabled by default until held-out refiner ablations beat no-refiner on quality after latency.

## Required Ablations

Run `python scripts/ablation_plan.py --out runs/ablation_plan.json` and execute each candidate with held-out `metrics.json`, `gallery.png`, `config.json`, and `train_log.jsonl`.

Decision metric: prompt-only held-out quality first, then latency/VRAM.

## Scale Gates

- Build the 50k-100k M8.5 freeze corpus with `scripts/prepare_corpus.py` from
  multi-source licensed inputs before training. Required source metadata:
  `name`, `path` or `url`, `license`, `source_url`, `category`, and useful tags.
- Build `manifest.parquet` when `pyarrow` is installed, or `manifest.jsonl` as the dependency-free fallback.
- Keep original PNGs at native resolution; train with bucketed 32/64/128 batches.
- Exclude or segregate `lossy=True` rows unless explicitly testing lossy data.
- Treat `report.status=below_min_count` as a corpus blocker, not a training-ready dataset.
- Run `scripts/benchmark_scale.py` before long training to capture loader throughput, peak VRAM, and 500k/1M epoch estimates.
- Resume only from checkpoints saved by the training scripts; they include model, optimizer, epoch, metrics, and RNG state.

## Smoke Evidence

- `python -m unittest discover -s tests`
- GPU smoke tokenizer, hierarchy, color, and prompt-only eval ran on CUDA with frozen CLIP for hierarchy/color/eval.

## Verdict

NO-GO for the real 500k-1M run until the ablation matrix is actually executed on the target dataset and checkpoints.
