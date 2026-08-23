import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from fenpix.dataset import PixelArtDataset, pixel_art_collate
from fenpix.hierarchy import (
    HierarchicalMaskGIT,
    HierarchicalMaskGITConfig,
    condition_to_shape,
    stage_native_shape,
    stage_tokens_from_batch,
)
from fenpix.maskgit import maskgit_loss
from fenpix.tokenizer import StructureTokenizer, StructureTokenizerConfig


def _sprite(path: Path, width: int, height: int) -> None:
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[:, : width // 2] = [255, 0, 0, 255]
    pixels[height // 2 :, :] = [0, 255, 0, 255]
    Image.fromarray(pixels, "RGBA").save(path)


class HierarchyTest(unittest.TestCase):
    def test_hierarchy_shapes(self):
        model = HierarchicalMaskGIT(HierarchicalMaskGITConfig(vocab_size=8, hidden_dim=16, depth=1, heads=4, text_dim=8))
        text = torch.randn(2, 8)

        samples = model.sample(128, 64, ["wide tile", "wide tile"], text_embeddings=text, steps=1)

        self.assertEqual(samples[32][0].shape, (2, 4, 8))
        self.assertEqual(samples[64][0].shape, (2, 8, 16))
        self.assertEqual(samples[128][0].shape, (2, 16, 32))

    def test_conditioning_between_stages_changes_logits(self):
        model = HierarchicalMaskGIT(HierarchicalMaskGITConfig(vocab_size=8, hidden_dim=16, depth=1, heads=4, text_dim=8))
        tokens = torch.zeros((1, 8, 8), dtype=torch.long)
        valid = torch.ones_like(tokens, dtype=torch.bool)
        text = torch.randn(1, 8)
        cond_a = torch.zeros((1, 4, 4), dtype=torch.long)
        cond_b = torch.ones((1, 4, 4), dtype=torch.long)
        stage = model.models["64"]

        logits_a = stage(tokens, valid, text, condition_to_shape(cond_a, torch.ones_like(cond_a, dtype=torch.bool), (8, 8), stage.config.pad_token_id))
        logits_b = stage(tokens, valid, text, condition_to_shape(cond_b, torch.ones_like(cond_b, dtype=torch.bool), (8, 8), stage.config.pad_token_id))

        self.assertFalse(torch.allclose(logits_a, logits_b))

    def test_variable_aspect_ratios(self):
        self.assertEqual(stage_native_shape(80, 40, 32), (16, 32))
        self.assertEqual(stage_native_shape(40, 80, 64), (64, 32))
        self.assertEqual(stage_native_shape(48, 48, 128), (48, 48))

    def test_masking_and_padding_in_stage_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _sprite(root / "wide.png", 32, 16)
            _sprite(root / "tall.png", 16, 32)
            batch = next(iter(DataLoader(PixelArtDataset(root), batch_size=2, collate_fn=pixel_art_collate)))

        tokenizer = StructureTokenizer(StructureTokenizerConfig(num_structure_classes=8, codebook_size=8, hidden_dim=8, latent_dim=4))
        tokens, valid = stage_tokens_from_batch(batch, tokenizer, 32)

        self.assertEqual(tokens.shape, valid.shape)
        self.assertFalse(valid[0, :, -1].any())
        self.assertFalse(valid[1, -1, :].any())

    def test_gradients(self):
        model = HierarchicalMaskGIT(HierarchicalMaskGITConfig(vocab_size=8, hidden_dim=16, depth=1, heads=4, text_dim=8, stages=(32, 64)))
        low = torch.randint(0, 8, (2, 8, 8))
        high = torch.randint(0, 8, (2, 16, 16))
        low_valid = torch.ones_like(low, dtype=torch.bool)
        high_valid = torch.ones_like(high, dtype=torch.bool)
        text = torch.randn(2, 8)

        loss = model.stage_loss(64, high, high_valid, text, low, low_valid)
        loss.backward()

        self.assertIsNotNone(model.models["64"].structure_cond_embed.weight.grad)
        self.assertIsNotNone(model.models["64"].head.weight.grad)

    def test_checkpoint_roundtrip(self):
        model = HierarchicalMaskGIT(HierarchicalMaskGITConfig(vocab_size=8, hidden_dim=16, depth=1, heads=4, text_dim=8, stages=(32, 64)))
        tokens = torch.randint(0, 8, (1, 8, 8))
        valid = torch.ones_like(tokens, dtype=torch.bool)
        text = torch.randn(1, 8)
        before = model.models["32"](tokens, valid, text)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hierarchy.pt"
            model.save_checkpoint(path)
            loaded = HierarchicalMaskGIT.load_checkpoint(path)
        after = loaded.models["32"](tokens, valid, text)

        self.assertTrue(torch.allclose(before, after))

    def test_tiny_hierarchical_overfit(self):
        torch.manual_seed(0)
        config = HierarchicalMaskGITConfig(vocab_size=4, hidden_dim=16, depth=1, heads=4, text_dim=4, stages=(32, 64, 128))
        model = HierarchicalMaskGIT(config)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        text = torch.ones((1, 4))
        targets = {
            32: torch.ones((1, 8, 8), dtype=torch.long),
            64: torch.full((1, 16, 16), 2, dtype=torch.long),
            128: torch.full((1, 32, 32), 3, dtype=torch.long),
        }
        valids = {stage: torch.ones_like(tokens, dtype=torch.bool) for stage, tokens in targets.items()}

        for _ in range(45):
            loss = torch.zeros(())
            previous = None
            for stage in config.stages:
                loss = loss + model.stage_loss(stage, targets[stage], valids[stage], text, *(previous or (None, None)))
                previous = (targets[stage], valids[stage])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        previous = None
        correct = []
        for stage in config.stages:
            stage_model = model.models[str(stage)]
            cond = condition_to_shape(previous[0], previous[1], targets[stage].shape[-2:], stage_model.config.pad_token_id) if previous else None
            masked = torch.full_like(targets[stage], stage_model.config.mask_token_id)
            pred = stage_model(masked, valids[stage], text, cond).argmax(dim=1)
            correct.append(pred.eq(targets[stage]).float().mean().item())
            previous = (targets[stage], valids[stage])

        self.assertGreater(min(correct), 0.95)


if __name__ == "__main__":
    unittest.main()
