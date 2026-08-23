import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from fenpix import PixelArtDataset
from scripts.prepare_kenney import prepare_kenney


SAMPLES = Path(__file__).parent / "sample_data" / "kenney_tiny_town"


class PrepareKenneyTest(unittest.TestCase):
    def test_prepares_sample_folder_in_fenpix_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "processed"

            report = prepare_kenney(SAMPLES, out)
            dataset = PixelArtDataset(out)
            metadata = json.loads((out / "tile_0000.json").read_text(encoding="utf-8"))
            copied_bytes = (out / "tile_0000.png").read_bytes()

        self.assertEqual(report["total_accepted"], 12)
        self.assertEqual(report["total_rejected"], 1)
        self.assertEqual(report["rejection_reasons"], {"unsupported_format": 1})
        self.assertEqual(report["size_distribution"], {"16x16": 12})
        self.assertEqual(len(dataset), 12)
        self.assertEqual(copied_bytes, (SAMPLES / "tile_0000.png").read_bytes())
        self.assertEqual(metadata["source"], "Kenney")
        self.assertEqual(metadata["source_url"], "https://kenney.nl/assets/tiny-town")
        self.assertEqual(metadata["license"], "Creative Commons CC0")
        self.assertEqual(metadata["width"], 16)
        self.assertEqual(metadata["height"], 16)
        self.assertFalse(metadata["lossy"])
        self.assertIn("tiny", metadata["tags"])

    def test_rejects_bad_and_oversized_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            root.mkdir()
            Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(root / "ok.png")
            Image.new("RGBA", (129, 8), (255, 0, 0, 255)).save(root / "big.png")
            colors = np.arange(300, dtype=np.uint16).reshape(10, 30)
            noisy = np.zeros((10, 30, 4), dtype=np.uint8)
            noisy[..., 0] = colors % 256
            noisy[..., 1] = colors // 256
            noisy[..., 3] = 255
            Image.fromarray(noisy, "RGBA").save(root / "photoish.png")
            (root / "bad.png").write_bytes(b"not a png")
            (root / "notes.txt").write_text("skip", encoding="utf-8")

            report = prepare_kenney(root, out)

        self.assertEqual(report["total_accepted"], 1)
        self.assertEqual(
            report["rejection_reasons"],
            {"corrupt": 1, "non_pixel_art": 1, "oversized": 1, "unsupported_format": 1},
        )

    def test_accepts_zip_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "kenney.zip"
            out = Path(tmp) / "processed"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.write(SAMPLES / "tile_0000.png", "kenney/tile_0000.png")

            report = prepare_kenney(archive, out)
            png_exists = (out / "kenney" / "tile_0000.png").exists()
            json_exists = (out / "kenney" / "tile_0000.json").exists()

        self.assertEqual(report["total_accepted"], 1)
        self.assertTrue(png_exists)
        self.assertTrue(json_exists)

    def test_rejects_unsafe_zip_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "kenney.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../bad.png", b"nope")

            with self.assertRaisesRegex(ValueError, "unsafe zip path"):
                prepare_kenney(archive, Path(tmp) / "processed")


if __name__ == "__main__":
    unittest.main()
