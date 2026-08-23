import tempfile
import unittest
from pathlib import Path

import torch

from fenpix.maskgit import MaskGIT, MaskGITConfig, maskgit_loss, random_mask_tokens
from fenpix.text import FrozenHashTextEncoder, TextEmbeddingCache, TextEncoderConfig


class TextConditioningTest(unittest.TestCase):
    def test_text_conditioning_changes_outputs(self):
        torch.manual_seed(0)
        model = MaskGIT(MaskGITConfig(vocab_size=7, hidden_dim=16, depth=1, heads=4, max_height=2, max_width=2, text_dim=8))
        encoder = FrozenHashTextEncoder(TextEncoderConfig(dim=8))
        tokens = torch.full((1, 2, 2), model.config.mask_token_id)
        valid = torch.ones_like(tokens, dtype=torch.bool)

        a = model(tokens, valid, encoder.encode(["red potion icon"]))
        b = model(tokens, valid, encoder.encode(["stone house"]))

        self.assertFalse(torch.allclose(a, b))

    def test_embedding_cache_correctness(self):
        encoder = FrozenHashTextEncoder(TextEncoderConfig(dim=8))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.pt"
            first = TextEmbeddingCache(path, encoder).encode(["grass tile", "small tree"])
            second = TextEmbeddingCache(path, encoder).encode(["small tree", "grass tile"])

        self.assertTrue(torch.allclose(first[0], second[1]))
        self.assertTrue(torch.allclose(first[1], second[0]))

    def test_unconditional_dropout_path(self):
        torch.manual_seed(0)
        model = MaskGIT(MaskGITConfig(vocab_size=7, hidden_dim=16, depth=1, heads=4, max_height=2, max_width=2, text_dim=8))
        encoder = FrozenHashTextEncoder(TextEncoderConfig(dim=8))
        tokens = torch.full((1, 2, 2), model.config.mask_token_id)
        valid = torch.ones_like(tokens, dtype=torch.bool)

        dropped = model(tokens, valid, encoder.encode(["red potion icon"]), cond_drop_prob=1)
        uncond = model(tokens, valid, None)

        self.assertTrue(torch.allclose(dropped, uncond))

    def test_gradients_only_through_generator(self):
        model = MaskGIT(MaskGITConfig(vocab_size=7, hidden_dim=16, depth=1, heads=4, max_height=2, max_width=2, text_dim=8))
        encoder = FrozenHashTextEncoder(TextEncoderConfig(dim=8))
        tokens = torch.tensor([[[1, 2], [3, 4]]])
        valid = torch.ones_like(tokens, dtype=torch.bool)
        masked, labels = random_mask_tokens(tokens, valid, model.config.mask_token_id, min_ratio=1, max_ratio=1)

        maskgit_loss(model(masked, valid, encoder.encode(["red potion icon"])), labels).backward()

        self.assertIsNone(encoder.proj.grad)
        self.assertIsNotNone(model.cond_proj.weight.grad)
        self.assertIsNotNone(model.head.weight.grad)

    def test_checkpoint_roundtrip_text_conditioned(self):
        model = MaskGIT(MaskGITConfig(vocab_size=7, hidden_dim=16, depth=1, heads=4, max_height=2, max_width=2, text_dim=8))
        encoder = FrozenHashTextEncoder(TextEncoderConfig(dim=8))
        tokens = torch.full((1, 2, 2), model.config.mask_token_id)
        valid = torch.ones_like(tokens, dtype=torch.bool)
        text = encoder.encode(["grass tile"])
        before = model(tokens, valid, text)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m5.pt"
            model.save_checkpoint(path)
            loaded = MaskGIT.load_checkpoint(path)
        after = loaded(tokens, valid, text)

        self.assertTrue(torch.allclose(before, after))

    def test_tiny_prompt_overfit(self):
        torch.manual_seed(0)
        encoder = FrozenHashTextEncoder(TextEncoderConfig(dim=16))
        prompts = ["red potion icon", "stone house"]
        text = encoder.encode(prompts)
        tokens = torch.tensor(
            [
                [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
                [[4, 4, 3, 3], [4, 4, 3, 3], [2, 2, 1, 1], [2, 2, 1, 1]],
            ]
        )
        valid = torch.ones_like(tokens, dtype=torch.bool)
        model = MaskGIT(MaskGITConfig(vocab_size=5, hidden_dim=32, depth=2, heads=4, max_height=4, max_width=4, text_dim=16))
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

        for _ in range(180):
            masked, labels = random_mask_tokens(tokens, valid, model.config.mask_token_id, min_ratio=1, max_ratio=1)
            loss = maskgit_loss(model(masked, valid, text), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        masked = torch.full_like(tokens, model.config.mask_token_id)
        pred = model(masked, valid, text).argmax(dim=1)

        self.assertGreater(pred.eq(tokens).float().mean().item(), 0.9)
        self.assertFalse(torch.equal(pred[0], pred[1]))


if __name__ == "__main__":
    unittest.main()
