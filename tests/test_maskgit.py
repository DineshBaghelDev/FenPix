import tempfile
import unittest
from pathlib import Path

import torch

from fenpix.maskgit import MaskGIT, MaskGITConfig, maskgit_loss, random_mask_tokens


class MaskGITTest(unittest.TestCase):
    def test_masking_keeps_padding_and_labels_only_masked_tokens(self):
        tokens = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
        valid = torch.tensor([[[True, False, True], [True, True, False]]])
        masked, labels = random_mask_tokens(tokens, valid, mask_token_id=9, min_ratio=1, max_ratio=1)

        self.assertTrue(torch.equal(masked[valid], torch.full((4,), 9)))
        self.assertTrue(torch.equal(masked[~valid], tokens[~valid]))
        self.assertTrue(torch.equal(labels[valid], tokens[valid]))
        self.assertTrue(labels[~valid].eq(-100).all())

    def test_transformer_shapes_variable_aspect(self):
        model = MaskGIT(MaskGITConfig(vocab_size=11, hidden_dim=16, depth=2, heads=4, max_height=8, max_width=9))
        tokens = torch.randint(0, 11, (2, 5, 7))
        valid = torch.ones_like(tokens, dtype=torch.bool)

        self.assertEqual(model(tokens, valid).shape, (2, 11, 5, 7))

    def test_loss_ignores_padding(self):
        logits = torch.zeros(1, 5, 2, 2)
        labels_a = torch.tensor([[[1, -100], [2, -100]]])
        labels_b = torch.tensor([[[1, -100], [2, -100]]])
        labels_b[0, 0, 1] = 4

        self.assertTrue(torch.allclose(maskgit_loss(logits, labels_a), maskgit_loss(logits, labels_b)))

    def test_gradients(self):
        model = MaskGIT(MaskGITConfig(vocab_size=8, hidden_dim=16, depth=1, heads=4, max_height=4, max_width=4))
        tokens = torch.randint(0, 8, (2, 4, 4))
        valid = torch.ones_like(tokens, dtype=torch.bool)
        masked, labels = random_mask_tokens(tokens, valid, model.config.mask_token_id, min_ratio=0.5, max_ratio=0.5)

        maskgit_loss(model(masked, valid), labels).backward()

        self.assertIsNotNone(model.token_embed.weight.grad)
        self.assertIsNotNone(model.head.weight.grad)

    def test_checkpoint_roundtrip(self):
        model = MaskGIT(MaskGITConfig(vocab_size=8, hidden_dim=16, depth=1, heads=4, max_height=4, max_width=4))
        tokens = torch.randint(0, 8, (1, 4, 4))
        valid = torch.ones_like(tokens, dtype=torch.bool)
        before = model(tokens, valid)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "maskgit.pt"
            model.save_checkpoint(path)
            loaded = MaskGIT.load_checkpoint(path)
        after = loaded(tokens, valid)

        self.assertTrue(torch.allclose(before, after))

    def test_iterative_decoding_returns_valid_tokens_and_padding(self):
        model = MaskGIT(MaskGITConfig(vocab_size=6, hidden_dim=16, depth=1, heads=4, max_height=4, max_width=4))
        valid = torch.tensor([[[True, True], [True, False]]])
        samples = model.sample((1, 2, 2), valid, steps=3)

        self.assertTrue(samples[valid].ge(0).all())
        self.assertTrue(samples[valid].lt(6).all())
        self.assertTrue(samples[~valid].eq(model.config.pad_token_id).all())

    def test_tiny_dataset_overfit(self):
        torch.manual_seed(0)
        tokens = torch.tensor([[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]])
        valid = torch.ones_like(tokens, dtype=torch.bool)
        model = MaskGIT(MaskGITConfig(vocab_size=5, hidden_dim=32, depth=2, heads=4, max_height=4, max_width=4))
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

        for _ in range(120):
            masked, labels = random_mask_tokens(tokens, valid, model.config.mask_token_id, min_ratio=1, max_ratio=1)
            loss = maskgit_loss(model(masked, valid), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        pred = model(torch.full_like(tokens, model.config.mask_token_id), valid).argmax(dim=1)
        self.assertGreater(pred.eq(tokens).float().mean().item(), 0.95)


if __name__ == "__main__":
    unittest.main()
