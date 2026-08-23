import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fenpix.evaluation import (
    PROMPT_EVAL_SET,
    boundary_f1,
    compute_quality_metrics,
    connected_component_consistency,
    grid_pixel_alignment,
    palette_fidelity,
    save_comparison_gallery,
    save_metrics,
    transparency_iou,
)
from fenpix.text import FrozenPretrainedTextEncoder, FrozenVisionLanguageEncoder, TextEncoderConfig


class QualityEvalTest(unittest.TestCase):
    def test_prompt_set_covers_m8_categories(self):
        joined = " ".join(PROMPT_EVAL_SET)
        for word in ("sprite", "icon", "tile", "object", "building", "scene", "isometric", "transparent"):
            self.assertIn(word, joined)

    def test_tiny_fallback_encoder_has_semantic_signal(self):
        encoder = FrozenPretrainedTextEncoder(TextEncoderConfig(dim=16, provider="tiny"))
        icon = encoder.encode(["red potion icon"])
        tile = encoder.encode(["grass tile"])
        self.assertFalse(torch.allclose(icon, tile))

    def test_pixel_art_metrics_reward_exact_match(self):
        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[1:3, 1:3] = [255, 0, 0, 255]

        self.assertEqual(palette_fidelity(pixels, pixels), 1.0)
        self.assertEqual(transparency_iou(pixels, pixels), 1.0)
        self.assertEqual(boundary_f1(pixels, pixels), 1.0)
        self.assertEqual(connected_component_consistency(pixels, pixels), 1.0)
        self.assertEqual(grid_pixel_alignment(pixels), 1.0)

    def test_pixel_art_metrics_notice_mismatch(self):
        target = np.zeros((4, 4, 4), dtype=np.uint8)
        target[1:3, 1:3] = [255, 0, 0, 255]
        pred = np.zeros((4, 4, 4), dtype=np.uint8)
        pred[0:2, 0:2] = [0, 0, 255, 255]
        pred[3, 3] = [0, 255, 0, 128]

        self.assertLess(palette_fidelity(pred, target), 1.0)
        self.assertLess(transparency_iou(pred, target), 1.0)
        self.assertLess(boundary_f1(pred, target), 1.0)
        self.assertLess(connected_component_consistency(pred, target), 1.0)
        self.assertLess(grid_pixel_alignment(pred), 1.0)

    def test_metrics_gallery_and_tiny_vlm(self):
        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[1:3, 1:3] = [255, 0, 0, 255]
        image = Image.fromarray(pixels, "RGBA")
        encoder = FrozenVisionLanguageEncoder(TextEncoderConfig(dim=16, provider="tiny"))

        metrics = compute_quality_metrics([image], [image], ["red icon"], encoder=encoder)

        self.assertGreaterEqual(metrics.palette_fidelity, 0.99)
        self.assertGreaterEqual(metrics.text_image_alignment, -1.0)

        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            gallery_path = Path(tmp) / "gallery.png"
            save_metrics(metrics.__dict__, metrics_path)
            save_comparison_gallery([image], [image], gallery_path, ["red icon"])
            self.assertTrue(metrics_path.exists())
            self.assertTrue(gallery_path.exists())


if __name__ == "__main__":
    unittest.main()
