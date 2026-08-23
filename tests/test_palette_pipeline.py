import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import DataLoader

from fenpix import PixelArtDataset, image_to_indices, pixel_art_collate, reconstruct_rgba


SAMPLES = Path(__file__).parent / "sample_data" / "kenney_tiny_town"


class PalettePipelineTest(unittest.TestCase):
    def test_roundtrip_preserves_native_rgba_when_within_palette_budget(self):
        pixels = np.array(
            [
                [[0, 0, 0, 0], [255, 0, 0, 128], [0, 255, 0, 255]],
                [[0, 0, 255, 255], [255, 255, 0, 255], [255, 255, 255, 255]],
            ],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sprite.png"
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("caption", "tiny test sprite")
            Image.fromarray(pixels, "RGBA").save(path, pnginfo=pnginfo)

            with Image.open(path) as image:
                encoding = image_to_indices(image, max_colors=8)
        rebuilt = np.asarray(reconstruct_rgba(encoding.indices, encoding.palette))

        self.assertEqual((encoding.width, encoding.height), (3, 2))
        self.assertFalse(encoding.lossy)
        self.assertEqual(encoding.metadata["caption"], "tiny test sprite")
        np.testing.assert_array_equal(rebuilt, pixels)

    def test_variable_aspect_ratios_small_sizes_and_128_max_are_native(self):
        for width, height in [(1, 1), (2, 7), (13, 3), (128, 128)]:
            pixels = np.zeros((height, width, 4), dtype=np.uint8)
            pixels[..., 0] = np.indices((height, width)).sum(axis=0) % 2 * 255
            pixels[..., 3] = 255

            encoding = image_to_indices(Image.fromarray(pixels, "RGBA"))

            self.assertEqual((encoding.width, encoding.height), (width, height))
            self.assertEqual(encoding.indices.shape, (height, width))
            np.testing.assert_array_equal(np.asarray(reconstruct_rgba(encoding.indices, encoding.palette)), pixels)

    def test_palette_budget_is_8_to_64_colors(self):
        image = Image.new("RGBA", (1, 1))

        with self.assertRaisesRegex(ValueError, "between 8 and 64"):
            image_to_indices(image, max_colors=7)
        with self.assertRaisesRegex(ValueError, "between 8 and 64"):
            image_to_indices(image, max_colors=65)

    def test_images_above_budget_are_marked_lossy(self):
        pixels = np.array([[[i * 20, 0, 255 - i * 20, 255] for i in range(9)]], dtype=np.uint8)
        image = Image.fromarray(pixels, "RGBA")

        encoding = image_to_indices(image, max_colors=8)
        rebuilt = np.asarray(reconstruct_rgba(encoding.indices, encoding.palette))

        self.assertTrue(encoding.lossy)
        self.assertEqual(encoding.unique_color_count, 9)
        self.assertLessEqual(len(encoding.palette), 8)
        self.assertFalse(np.array_equal(rebuilt, pixels))

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

    def test_malformed_png_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.png"
            path.write_bytes(b"not a png")

            dataset = PixelArtDataset(tmp)
            with self.assertRaises(UnidentifiedImageError):
                dataset[0]

    def test_real_kenney_samples_scan_and_roundtrip(self):
        paths = sorted(SAMPLES.glob("tile_*.png"))
        self.assertGreaterEqual(len(paths), 10)
        self.assertLessEqual(len(paths), 20)

        dataset = PixelArtDataset(SAMPLES)
        self.assertEqual(len(dataset), len(paths))

        for path in paths:
            with Image.open(path) as image:
                source = np.asarray(image.convert("RGBA"), dtype=np.uint8)
                encoding = image_to_indices(image, max_colors=64)

            self.assertLessEqual(encoding.width, 128)
            self.assertLessEqual(encoding.height, 128)
            self.assertLessEqual(len(encoding.palette), 64)
            self.assertEqual(encoding.indices.shape, (encoding.height, encoding.width))
            if encoding.unique_color_count <= 64:
                self.assertFalse(encoding.lossy)
                np.testing.assert_array_equal(np.asarray(reconstruct_rgba(encoding.indices, encoding.palette)), source)


if __name__ == "__main__":
    unittest.main()
