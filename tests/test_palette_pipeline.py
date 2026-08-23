import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import DataLoader

from fenpix import (
    BucketBatchSampler,
    PixelArtDataset,
    filtered_indices,
    image_to_indices,
    pixel_art_collate,
    reconstruct_rgba,
    train_val_test_split,
    train_validation_split,
)


SAMPLES = Path(__file__).parent / "sample_data" / "kenney_tiny_town"


class PalettePipelineTest(unittest.TestCase):
    def _write_rgba(self, path: Path, width: int, height: int, color=(255, 0, 0, 255)):
        pixels = np.zeros((height, width, 4), dtype=np.uint8)
        pixels[..., :] = color
        pixels[0, 0] = [0, 0, 0, 0]
        Image.fromarray(pixels, "RGBA").save(path)

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

        self.assertEqual(batch["indices"].shape, (2, 3, 4))
        self.assertEqual(batch["size"].tolist(), [[2, 3], [4, 1]])
        self.assertEqual(batch["metadata"][0]["caption"], "red block")
        self.assertTrue(batch["valid_mask"][0, :3, :2].all())
        self.assertFalse(batch["valid_mask"][0, :, 2:].any())
        self.assertTrue(batch["valid_mask"][1, :1, :4].all())
        self.assertFalse(batch["valid_mask"][1, 1:, :].any())

    def test_m2_mixed_dimensions_transparency_padding_and_batch_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_rgba(root / "icon_32.png", 32, 32)
            self._write_rgba(root / "wide_64.png", 64, 32, (0, 255, 0, 255))
            self._write_rgba(root / "tall_128.png", 40, 128, (0, 0, 255, 255))

            dataset = PixelArtDataset(root)
            batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=pixel_art_collate)))

        self.assertEqual(batch["indices"].shape, (3, 128, 64))
        self.assertEqual(batch["structure_indices"].shape, batch["indices"].shape)
        self.assertEqual(batch["palette"].shape[-1], 4)
        self.assertEqual(batch["size"].tolist(), [[32, 32], [40, 128], [64, 32]])
        self.assertEqual(batch["bucket_size"].tolist(), [32, 128, 64])
        self.assertEqual(batch["aspect_bucket"], ["square", "portrait", "landscape"])
        self.assertFalse(batch["valid_mask"][0, 32:, :].any())
        self.assertFalse(batch["valid_mask"][0, :, 32:].any())
        self.assertTrue((batch["palette"][:, :, 3] == 0).any())

    def test_train_validation_split_is_deterministic(self):
        dataset = PixelArtDataset(SAMPLES)

        train_a, val_a = train_validation_split(dataset, validation_fraction=0.25, seed=123)
        train_b, val_b = train_validation_split(dataset, validation_fraction=0.25, seed=123)
        train_c, val_c = train_validation_split(dataset, validation_fraction=0.25, seed=456)

        self.assertEqual(train_a.indices, train_b.indices)
        self.assertEqual(val_a.indices, val_b.indices)
        self.assertNotEqual(val_a.indices, val_c.indices)
        self.assertEqual(len(train_a) + len(val_a), len(dataset))

    def test_train_val_test_split_is_deterministic_and_disjoint(self):
        dataset = PixelArtDataset(SAMPLES)

        train, val, test = train_val_test_split(dataset, validation_fraction=0.25, test_fraction=0.25, seed=123)
        train_b, val_b, test_b = train_val_test_split(dataset, validation_fraction=0.25, test_fraction=0.25, seed=123)
        all_indices = train.indices + val.indices + test.indices

        self.assertEqual(train.indices, train_b.indices)
        self.assertEqual(val.indices, val_b.indices)
        self.assertEqual(test.indices, test_b.indices)
        self.assertEqual(len(all_indices), len(set(all_indices)))
        self.assertEqual(len(all_indices), len(dataset))

    def test_lossy_filter_excludes_by_default_and_can_include(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_rgba(root / "ok.png", 2, 2)
            noisy = np.array([[[i * 20, 0, 255 - i * 20, 255] for i in range(9)]], dtype=np.uint8)
            Image.fromarray(noisy, "RGBA").save(root / "lossy.png")
            dataset = PixelArtDataset(root, max_colors=8)

            lossless = filtered_indices(dataset)
            all_rows = filtered_indices(dataset, include_lossy=True)

        self.assertEqual(len(lossless), 1)
        self.assertEqual(len(all_rows), 2)

    def test_bucket_batch_sampler_groups_aspect_and_resolution_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_rgba(root / "a.png", 20, 20)
            self._write_rgba(root / "b.png", 30, 20)
            self._write_rgba(root / "c.png", 64, 20)
            self._write_rgba(root / "d.png", 32, 64)
            self._write_rgba(root / "e.png", 128, 80)

            dataset = PixelArtDataset(root)
            loader = DataLoader(
                dataset,
                batch_sampler=BucketBatchSampler(dataset, batch_size=2, seed=7),
                collate_fn=pixel_art_collate,
            )

            for batch in loader:
                self.assertEqual(len(set(batch["bucket"])), 1)
            buckets = {dataset[i]["bucket"] for i in range(len(dataset))}

        self.assertIn("32:square", buckets)
        self.assertIn("64:landscape", buckets)
        self.assertIn("64:portrait", buckets)
        self.assertIn("128:landscape", buckets)

    def test_cached_and_uncached_outputs_are_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_rgba(root / "sprite.png", 17, 9)
            (root / "sprite.json").write_text('{"caption":"cached"}', encoding="utf-8")

            uncached = PixelArtDataset(root)[0]
            cached_dataset = PixelArtDataset(root, cache=True)
            first_cached = cached_dataset[0]
            second_cached = PixelArtDataset(root, cache=True)[0]

        for cached in (first_cached, second_cached):
            self.assertTrue(torch.equal(uncached["indices"], cached["indices"]))
            self.assertTrue(torch.equal(uncached["palette"], cached["palette"]))
            self.assertEqual(uncached["dimensions"], cached["dimensions"])
            self.assertEqual(uncached["metadata"], cached["metadata"])

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
