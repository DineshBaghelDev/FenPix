import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from fenpix.color import IndexedColorConfig, IndexedColorModel, palette_to_uint8, reconstruct_indexed_png
from fenpix.dataset import PixelArtDataset, pixel_art_collate


def _pad_palette(palette: torch.Tensor, max_colors: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.zeros((palette.shape[0], max_colors, 4), dtype=torch.uint8)
    mask = torch.zeros((palette.shape[0], max_colors), dtype=torch.bool)
    out[:, : palette.shape[1]] = palette
    mask[:, : palette.shape[1]] = True
    return out, mask, torch.full((palette.shape[0],), palette.shape[1], dtype=torch.int16)


class IndexedColorTest(unittest.TestCase):
    def test_palette_validity_and_dynamic_size(self):
        model = IndexedColorModel(IndexedColorConfig(max_colors=16, min_colors=8, structure_vocab_size=8, hidden_dim=16, depth=1, heads=4, text_dim=4, max_height=8, max_width=8))
        structure = torch.randint(0, 8, (3, 8, 8))
        valid = torch.ones_like(structure, dtype=torch.bool)
        out = model.predict_palette(structure, valid, torch.randn(3, 4))

        self.assertEqual(out["palette"].shape, (3, 16, 4))
        self.assertTrue(out["palette"].dtype == torch.uint8)
        self.assertTrue(out["palette_size"].ge(8).all())
        self.assertTrue(out["palette_size"].le(16).all())
        self.assertTrue(torch.equal(out["palette_mask"], torch.arange(16)[None, :] < out["palette_size"][:, None]))

    def test_sampled_indices_stay_inside_predicted_palette(self):
        torch.manual_seed(0)
        model = IndexedColorModel(IndexedColorConfig(max_colors=8, min_colors=8, structure_vocab_size=4, hidden_dim=16, depth=1, heads=4, text_dim=4, max_height=4, max_width=4))
        structure = torch.randint(0, 4, (2, 4, 4))
        valid = torch.ones_like(structure, dtype=torch.bool)
        out = model.sample(structure, valid, torch.randn(2, 4), steps=2)

        self.assertTrue(out["indices"].ge(0).all())
        self.assertTrue((out["indices"] < out["palette_size"][:, None, None]).all())

    def test_alpha_transparency_is_explicit(self):
        logits = torch.zeros((1, 8, 4))
        logits[:, 0, 3] = -20
        palette, mask = palette_to_uint8(logits, torch.tensor([8]))

        self.assertTrue(mask[0, 0])
        self.assertEqual(palette[0, 0].tolist(), [0, 0, 0, 0])

    def test_variable_native_sizes_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGBA", (7, 3), (255, 0, 0, 255)).save(root / "wide.png")
            Image.new("RGBA", (4, 9), (0, 0, 0, 0)).save(root / "tall.png")
            batch = next(iter(DataLoader(PixelArtDataset(root), batch_size=2, collate_fn=pixel_art_collate)))

        model = IndexedColorModel(IndexedColorConfig(max_colors=8, min_colors=8, structure_vocab_size=8, hidden_dim=16, depth=1, heads=4, text_dim=4, max_height=9, max_width=7))
        palette, palette_mask, palette_size = _pad_palette(batch["palette"], 8)
        loss = model.loss(batch["indices"], batch["valid_mask"], batch["structure_indices"].clamp_min(0), torch.randn(2, 4), palette, palette_mask, palette_size)["loss"]

        self.assertTrue(torch.isfinite(loss))

    def test_gradients(self):
        model = IndexedColorModel(IndexedColorConfig(max_colors=8, min_colors=8, structure_vocab_size=4, hidden_dim=16, depth=1, heads=4, text_dim=4, max_height=4, max_width=4))
        indices = torch.randint(0, 3, (2, 4, 4))
        valid = torch.ones_like(indices, dtype=torch.bool)
        structure = torch.randint(0, 4, indices.shape)
        palette = torch.tensor([[[0, 0, 0, 0], [255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [1, 1, 1, 255], [2, 2, 2, 255], [3, 3, 3, 255], [4, 4, 4, 255]]], dtype=torch.uint8).repeat(2, 1, 1)
        palette_mask = torch.ones((2, 8), dtype=torch.bool)

        loss = model.loss(indices, valid, structure, torch.randn(2, 4), palette, palette_mask, torch.full((2,), 8))["loss"]
        loss.backward()

        self.assertIsNotNone(model.palette_head[-1].weight.grad)
        self.assertIsNotNone(model.index_model.head.weight.grad)

    def test_checkpoint_roundtrip(self):
        model = IndexedColorModel(IndexedColorConfig(max_colors=8, min_colors=8, structure_vocab_size=4, hidden_dim=16, depth=1, heads=4, text_dim=4, max_height=4, max_width=4))
        structure = torch.randint(0, 4, (1, 4, 4))
        valid = torch.ones_like(structure, dtype=torch.bool)
        text = torch.randn(1, 4)
        before = model.predict_palette(structure, valid, text)["palette_logits"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "color.pt"
            model.save_checkpoint(path)
            loaded = IndexedColorModel.load_checkpoint(path)
        after = loaded.predict_palette(structure, valid, text)["palette_logits"]

        self.assertTrue(torch.allclose(before, after))

    def test_reconstruct_exact_rgba_from_palette_and_indices(self):
        palette = np.array([[0, 0, 0, 0], [255, 0, 0, 255], [0, 0, 255, 255]], dtype=np.uint8)
        indices = np.array([[0, 1], [2, 1]], dtype=np.int64)
        rgba = np.asarray(reconstruct_indexed_png(indices, palette))

        np.testing.assert_array_equal(rgba, palette[indices])

    def test_tiny_overfit(self):
        torch.manual_seed(0)
        model = IndexedColorModel(IndexedColorConfig(max_colors=8, min_colors=8, structure_vocab_size=4, hidden_dim=32, depth=1, heads=4, text_dim=4, max_height=4, max_width=4))
        opt = torch.optim.AdamW(model.parameters(), lr=8e-3)
        indices = torch.tensor([[[0, 1, 1, 0], [0, 2, 2, 0], [3, 2, 2, 3], [3, 3, 3, 3]]])
        valid = torch.ones_like(indices, dtype=torch.bool)
        structure = indices.clone()
        text = torch.ones((1, 4))
        palette = torch.tensor([[[0, 0, 0, 0], [255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [1, 1, 1, 255], [2, 2, 2, 255], [3, 3, 3, 255], [4, 4, 4, 255]]], dtype=torch.uint8)
        palette_mask = torch.ones((1, 8), dtype=torch.bool)

        for _ in range(80):
            losses = model.loss(indices, valid, structure, text, palette, palette_mask, torch.tensor([8]))
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()

        masked = torch.full_like(indices, model.index_model.config.mask_token_id)
        logits = model.index_model(masked, valid, model._index_text(text, palette, palette_mask), structure)
        pred = logits.argmax(dim=1)

        self.assertGreater(pred.eq(indices).float().mean().item(), 0.9)


if __name__ == "__main__":
    unittest.main()
