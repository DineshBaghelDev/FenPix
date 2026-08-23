from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit the required FenPix ablation matrix.")
    parser.add_argument("--out", type=Path, default=Path("runs/ablation_plan.json"))
    args = parser.parse_args()

    tokenizer = [
        {"tokenizer_downsample": downsample, "codebook_size": codebook}
        for downsample, codebook in itertools.product((2, 4), (128, 512, 1024))
    ]
    representation = [
        {"structure_representation": "current"},
        {"structure_representation": "connected_component_aware"},
    ]
    palette = [
        {"palette_prediction": "ordered_palette"},
        {"palette_prediction": "canonical_ordering"},
        {"palette_prediction": "permutation_invariant_matching"},
    ]
    refiner = [
        {"final_refinement": "none"},
        {"final_refinement": "flow_refiner"},
        {"final_refinement": "extra_masked_denoising_pass"},
    ]
    plan = {
        "decision_metric": "held_out_prompt_only_quality_then_latency",
        "tokenizer": tokenizer,
        "representation": representation,
        "palette": palette,
        "refiner": refiner,
        "required_artifacts": ["metrics.json", "gallery.png", "config.json", "train_log.jsonl"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
