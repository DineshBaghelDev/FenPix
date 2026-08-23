import tempfile
import unittest
from pathlib import Path

import torch

from fenpix.dataset import PixelArtDataset, dataset_quality_report
from fenpix.evaluation import PROMPT_EVAL_SET, compare_color_and_refiner, save_comparison_gallery, save_metrics
from fenpix.text import FrozenPretrainedTextEncoder, TextEncoderConfig


class QualityEvalTest(unittest.TestCase):
    def test_prompt_set_covers_m8_categories(self):
        joined = " ".join(PROMPT_EVAL_SET)
        for word in ("sprite", "icon", "tile", "object", "building", "scene", "isometric", "transparent"):
            self.assertIn(word, joined)

    def test_pretrained_encoder_has_semantic_signal(self):
        encoder = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=16))
        icon = encoder.encode(["red potion icon"])
        tile = encoder.encode(["grass tile"])
        self.assertFalse(torch.allclose(icon, tile))

    def test_metrics_gallery_and_dataset_hooks(self):
        dataset = PixelArtDataset(Path(__file__).parent / "sample_data" / "kenney_tiny_town")
        sample = dataset[0]
        indices = sample["indices"][None]
        valid = sample["valid_mask"][None]
        palette = sample["palette"][None]
        palette_mask = torch.ones((1, palette.shape[1]), dtype=torch.bool)
        logits = torch.nn.functional.one_hot(indices, 64).permute(0, 3, 1, 2).float()

        metrics = compare_color_and_refiner(
            logits,
            indices,
            valid,
            indices,
            None,
            palette,
            palette_mask,
            ["grass tile"],
            [{"caption": "grass tile"}],
            None,
        )

        self.assertGreaterEqual(metrics["baseline_color"]["index_accuracy"], 0.99)
        self.assertEqual(metrics["flow_refinement_default"], "disabled")
        self.assertIn("duplicate_count", dataset_quality_report(dataset))

        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            gallery_path = Path(tmp) / "gallery.png"
            save_metrics(metrics, metrics_path)
            save_comparison_gallery({"target": indices[0:1]}, palette, palette_mask, gallery_path, ["grass tile"])
            self.assertTrue(metrics_path.exists())
            self.assertTrue(gallery_path.exists())


if __name__ == "__main__":
    unittest.main()
