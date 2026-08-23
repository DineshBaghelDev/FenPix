import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from fenpix.dataset import PixelArtDataset, pixel_art_collate
from fenpix.tokenizer import (
    StructureTokenizer,
    StructureTokenizerConfig,
    canonical_structure_indices,
    masked_cross_entropy,
    structure_one_hot,
    tokenizer_metrics,
)


def _sprite(path: Path, width: int, height: int, kind: int) -> None:
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    yy, xx = np.indices((height, width))
    mask = (xx - width // 2) ** 2 + (yy - height // 2) ** 2 < (min(width, height) // 3) ** 2
    pixels[mask] = [255, 0, 0, 255]
    if kind % 2:
        pixels[yy > height // 2] = [0, 255, 0, 255]
    if kind % 3 == 0:
        pixels[:, width // 3 : width // 3 + 2] = [0, 0, 255, 255]
    Image.fromarray(pixels, "RGBA").save(path)


class StructureTokenizerTest(unittest.TestCase):
    def test_tensor_shapes_and_codebook_lookup(self):
        model = StructureTokenizer(StructureTokenizerConfig(num_structure_classes=8, codebook_size=11, hidden_dim=16, latent_dim=12))
        x = torch.randn(2, 8, 32, 24)

        out = model(x)

        self.assertEqual(out["logits"].shape, (2, 8, 32, 24))
        self.assertEqual(out["codes"].shape, (2, 8, 6))
        self.assertTrue(((0 <= out["codes"]) & (out["codes"] < 11)).all())

    def test_masked_loss_ignores_padding(self):
        logits = torch.zeros(1, 3, 2, 2)
        valid = torch.tensor([[[True, False], [True, False]]])
        targets_a = torch.tensor([[[1, 1], [2, 1]]])
        targets_b = torch.tensor([[[1, 2], [2, 0]]])

        self.assertTrue(torch.allclose(masked_cross_entropy(logits, targets_a, valid), masked_cross_entropy(logits, targets_b, valid)))

    def test_canonical_structure_ignores_rgba_color_values(self):
        indices = torch.tensor([[[0, 1], [2, -1]], [[0, 1], [2, -1]]])
        palette_a = torch.tensor([[[0, 0, 0, 0], [255, 0, 0, 255], [0, 0, 255, 255]]], dtype=torch.uint8).repeat(2, 1, 1)
        palette_b = torch.tensor([[[0, 0, 0, 0], [5, 5, 5, 255], [9, 9, 9, 255]]], dtype=torch.uint8).repeat(2, 1, 1)
        valid = indices.ge(0)

        self.assertTrue(torch.equal(canonical_structure_indices(indices, palette_a, valid), canonical_structure_indices(indices, palette_b, valid)))

    def test_gradients_reach_encoder_and_codebook(self):
        config = StructureTokenizerConfig(num_structure_classes=8, codebook_size=16, hidden_dim=16, latent_dim=8, downsample=2)
        model = StructureTokenizer(config)
        targets = torch.randint(0, 8, (3, 16, 16))
        valid = torch.ones_like(targets, dtype=torch.bool)
        out = model(structure_one_hot(targets, valid, 8))
        loss = masked_cross_entropy(out["logits"], targets, valid) + out["vq_loss"]

        loss.backward()

        self.assertIsNotNone(model.encoder[0].weight.grad)
        self.assertIsNotNone(model.quantizer.embedding.weight.grad)

    def test_checkpoint_roundtrip(self):
        config = StructureTokenizerConfig(num_structure_classes=8, codebook_size=16, hidden_dim=16, latent_dim=8, downsample=2)
        model = StructureTokenizer(config)
        x = torch.randn(1, 8, 16, 16)
        before = model(x)["logits"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save_checkpoint(path)
            loaded = StructureTokenizer.load_checkpoint(path)
        after = loaded(x)["logits"]

        self.assertTrue(torch.allclose(before, after))

    def test_variable_aspect_ratio_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _sprite(root / "wide.png", 32, 16, 1)
            _sprite(root / "tall.png", 16, 32, 2)
            batch = next(iter(DataLoader(PixelArtDataset(root), batch_size=2, collate_fn=pixel_art_collate)))
        config = StructureTokenizerConfig(num_structure_classes=8, codebook_size=16, hidden_dim=16, latent_dim=8, downsample=2)
        targets = canonical_structure_indices(batch["indices"], batch["palette"], batch["valid_mask"], max_regions=7)
        out = StructureTokenizer(config)(structure_one_hot(targets, batch["valid_mask"], 8))

        self.assertEqual(out["logits"].shape[-2:], batch["indices"].shape[-2:])

    def test_odd_size_logits_match_input(self):
        model = StructureTokenizer(StructureTokenizerConfig(num_structure_classes=8, codebook_size=11, hidden_dim=16, latent_dim=12))
        x = torch.randn(2, 8, 21, 21)

        out = model(x)

        self.assertEqual(out["logits"].shape[-2:], (21, 21))

    def test_short_overfit_tiny_sample_set(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(8):
                _sprite(root / f"{i}.png", 16, 16, i)
            batch = next(iter(DataLoader(PixelArtDataset(root), batch_size=8, collate_fn=pixel_art_collate)))

        config = StructureTokenizerConfig(num_structure_classes=8, codebook_size=32, hidden_dim=32, latent_dim=16, downsample=2)
        model = StructureTokenizer(config)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        targets = canonical_structure_indices(batch["indices"], batch["palette"], batch["valid_mask"], max_regions=7)
        x = structure_one_hot(targets, batch["valid_mask"], 8)
        for _ in range(140):
            out = model(x)
            loss = masked_cross_entropy(out["logits"], targets, batch["valid_mask"]) + out["vq_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        metrics = tokenizer_metrics(out["logits"], targets, batch["valid_mask"], out["codes"], config.codebook_size)

        self.assertGreater(metrics["accuracy"], 0.9)
        self.assertGreater(metrics["silhouette_accuracy"], 0.95)
        self.assertIn("dead_codes", metrics)


if __name__ == "__main__":
    unittest.main()
