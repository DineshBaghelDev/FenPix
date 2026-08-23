import tempfile
import unittest
from pathlib import Path

import torch

from fenpix.maskgit import MaskGIT, MaskGITConfig
from fenpix.training import load_training_checkpoint, save_training_checkpoint, set_deterministic


class TrainingStateTest(unittest.TestCase):
    def test_resume_restores_model_optimizer_and_rng(self):
        set_deterministic(123)
        model = MaskGIT(MaskGITConfig(vocab_size=4, hidden_dim=8, depth=1, heads=2, max_height=2, max_width=2))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _ = torch.rand(3)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume.pt"
            save_training_checkpoint(path, model, opt, 7, {"loss": 1.0})
            expected_next = torch.rand(3)
            torch.rand(9)
            loaded = MaskGIT(MaskGITConfig(vocab_size=4, hidden_dim=8, depth=1, heads=2, max_height=2, max_width=2))
            loaded_opt = torch.optim.AdamW(loaded.parameters(), lr=1e-3)
            epoch = load_training_checkpoint(path, loaded, loaded_opt)
            actual_next = torch.rand(3)

        self.assertEqual(epoch, 7)
        self.assertTrue(torch.allclose(expected_next, actual_next))
        for before, after in zip(model.parameters(), loaded.parameters()):
            self.assertTrue(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
