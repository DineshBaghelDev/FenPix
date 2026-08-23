import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from fenpix import PixelArtDataset, image_to_indices, pixel_art_collate, reconstruct_rgba


class PalettePipelineTest(unittest.TestCase):
    def test_roundtrip_preserves_native_rgba_when_within_palette_budget(self):
        pixels = np.array(
            [
                [[0, 0, 0, 0], [255, 0, 0, 255], [0, 255, 0, 255]],
                [[0, 0, 255, 255], [255, 255, 0, 255], [255, 255, 255, 255]],
            ],
            dtype=np.uint8,
        )
        image = Image.fromarray(pixels, "RGBA")

        encoding = image_to_indices(image, max_colors=8)
        rebuilt = np.asarray(reconstruct_rgba(encoding.indices, encoding.palette))

        self.assertEqual((encoding.width, encoding.height), (3, 2))
        np.testing.assert_array_equal(rebuilt, pixels)

    def test_dataset_keeps_variable_sizes_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGBA", (2, 3), (255, 0, 0, 255)).save(root / "a.png")
            (root / "a.json").write_text('{"caption":"red block"}', encoding="utf-8")
            Image.new("RGBA", (4, 1), (0, 0, 0, 0)).save(root / "b.png")

            dataset = PixelArtDataset(root)
            batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=pixel_art_collate)))

        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0]["indices"].shape, (3, 2))
        self.assertEqual(batch[1]["indices"].shape, (1, 4))
        self.assertEqual(batch[0]["metadata"]["caption"], "red block")


if __name__ == "__main__":
    unittest.main()
