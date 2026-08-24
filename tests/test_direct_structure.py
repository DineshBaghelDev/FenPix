import tempfile
import unittest
from pathlib import Path

import torch

from fenpix.direct_structure import DirectStructureConfig, DirectStructureGenerator


class DirectStructureTest(unittest.TestCase):
    def test_loss_adds_gradients_to_structure_and_aux_heads(self):
        model = DirectStructureGenerator(DirectStructureConfig(vocab_size=8, hidden_dim=16, depth=2, text_dim=4, max_height=8, max_width=8))
        targets = torch.zeros((2, 8, 8), dtype=torch.long)
        targets[:, 2:6, 2:6] = 1
        valid = torch.ones_like(targets, dtype=torch.bool)

        loss = model.loss(targets, valid, torch.randn(2, 4))["loss"]
        loss.backward()

        self.assertIsNotNone(model.structure_head.weight.grad)
        self.assertIsNotNone(model.occupancy_head.weight.grad)
        self.assertIsNotNone(model.boundary_head.weight.grad)
        self.assertIsNotNone(model.count_head.weight.grad)

    def test_sample_preserves_shape_and_padding(self):
        model = DirectStructureGenerator(DirectStructureConfig(vocab_size=8, hidden_dim=16, depth=1, text_dim=0, max_height=4, max_width=5))
        valid = torch.ones((1, 4, 5), dtype=torch.bool)
        valid[:, :, -1] = False

        sample = model.sample(valid.shape, valid, steps=1)

        self.assertEqual(sample.shape, valid.shape)
        self.assertTrue(sample[:, :, -1].eq(model.config.pad_token_id).all())
        self.assertTrue(sample[valid].lt(model.config.vocab_size).all())

    def test_checkpoint_roundtrip(self):
        model = DirectStructureGenerator(DirectStructureConfig(vocab_size=8, hidden_dim=16, depth=1, text_dim=4, max_height=4, max_width=4))
        tokens = torch.zeros((1, 4, 4), dtype=torch.long)
        valid = torch.ones_like(tokens, dtype=torch.bool)
        text = torch.randn(1, 4)
        before = model(tokens, valid, text)["logits"]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "direct.pt"
            model.save_checkpoint(path)
            loaded = DirectStructureGenerator.load_checkpoint(path)

        after = loaded(tokens, valid, text)["logits"]
        self.assertTrue(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
