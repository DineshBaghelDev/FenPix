import tempfile
import unittest
from pathlib import Path

import torch

from fenpix.refiner import FlowRefinerConfig, PaletteLogitFlowRefiner, compare_refinement


class FlowRefinerTest(unittest.TestCase):
    def test_refine_keeps_shape_masks_and_argmax_indices(self):
        model = PaletteLogitFlowRefiner(FlowRefinerConfig(max_colors=8, structure_vocab_size=4, hidden_dim=16, depth=1, text_dim=4))
        logits = torch.randn(2, 8, 4, 5)
        valid = torch.ones((2, 4, 5), dtype=torch.bool)
        valid[1, :, 3:] = False
        structure = torch.randint(0, 4, (2, 4, 5))
        palette = torch.zeros((2, 8, 4), dtype=torch.uint8)
        palette_mask = torch.ones((2, 8), dtype=torch.bool)

        refined = model.refine(logits, 2, valid, structure, torch.randn(2, 4), palette, palette_mask)
        indices = refined.argmax(dim=1).masked_fill(~valid, -1)

        self.assertEqual(refined.shape, logits.shape)
        self.assertTrue((indices[~valid] == -1).all())
        self.assertTrue((indices[valid] >= 0).all())

    def test_loss_backprop_and_checkpoint_roundtrip(self):
        torch.manual_seed(0)
        model = PaletteLogitFlowRefiner(FlowRefinerConfig(max_colors=8, structure_vocab_size=4, hidden_dim=16, depth=1, text_dim=4))
        base = torch.randn(1, 8, 4, 4)
        indices = torch.randint(0, 8, (1, 4, 4))
        valid = torch.ones_like(indices, dtype=torch.bool)
        structure = torch.randint(0, 4, indices.shape)
        palette = torch.zeros((1, 8, 4), dtype=torch.uint8)
        palette_mask = torch.ones((1, 8), dtype=torch.bool)

        loss = model.loss(base, indices, valid, structure, torch.randn(1, 4), palette, palette_mask)
        loss.backward()

        self.assertIsNotNone(model.out_proj.weight.grad)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "refiner.pt"
            model.save_checkpoint(path, extra={"loss": float(loss.item())})
            loaded = PaletteLogitFlowRefiner.load_checkpoint(path)
        before = model.refine(base, 1, valid, structure, None, palette, palette_mask)
        after = loaded.refine(base, 1, valid, structure, None, palette, palette_mask)
        self.assertTrue(torch.allclose(before, after))

    def test_compare_reports_requested_ablation_metrics(self):
        model = PaletteLogitFlowRefiner(FlowRefinerConfig(max_colors=8, structure_vocab_size=4, hidden_dim=16, depth=1, text_dim=0))
        indices = torch.tensor([[[0, 1], [1, 0]]])
        base = torch.nn.functional.one_hot(indices, 8).permute(0, 3, 1, 2).float()
        valid = torch.ones_like(indices, dtype=torch.bool)
        structure = indices.clone()
        palette = torch.zeros((1, 8, 4), dtype=torch.uint8)
        palette_mask = torch.ones((1, 8), dtype=torch.bool)

        metrics = compare_refinement(model, base, indices, valid, structure, None, palette, palette_mask, steps=(0, 1, 2, 4))

        self.assertEqual(set(metrics), {0, 1, 2, 4})
        self.assertGreaterEqual(metrics[0]["index_accuracy"], 0.99)
        self.assertIn("latency_ms", metrics[4])


if __name__ == "__main__":
    unittest.main()
