import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fenpix.corpus import curate_provisional_corpus
from fenpix.dataset import load_dataset_manifest


def _meta(path: Path) -> None:
    path.with_suffix(".json").write_text(
        json.dumps({"source": "test", "license": "CC0-1.0", "tags": ["pixel-art"]}),
        encoding="utf-8",
    )


class CurateCorpusTest(unittest.TestCase):
    def test_curates_splits_and_holds_out_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "candidates"
            out = Path(tmp) / "curated"
            root.mkdir()

            ok = root / "ok.png"
            Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(ok)
            _meta(ok)
            before = ok.read_bytes()

            dupe = root / "dupe.png"
            dupe.write_bytes(before)
            _meta(dupe)

            sheet = Image.new("RGBA", (260, 16), (0, 0, 0, 0))
            sprite_a = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
            sprite_b = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
            for i in range(8):
                sprite_a.putpixel((i, i), (0, 255, 0, 255))
                sprite_b.putpixel((7 - i, i), (0, 0, 255, 255))
            sheet.alpha_composite(sprite_a, (1, 1))
            sheet.alpha_composite(sprite_b, (240, 1))
            sheet_path = root / "sheet.png"
            sheet.save(sheet_path)
            _meta(sheet_path)

            colors = np.zeros((17, 17, 4), dtype=np.uint8)
            for i in range(289):
                y, x = divmod(i, 17)
                colors[y, x] = (i % 256, i // 256, 0, 255)
            smooth = root / "smooth.png"
            Image.fromarray(colors, "RGBA").save(smooth)
            _meta(smooth)

            lossy = np.zeros((9, 8, 4), dtype=np.uint8)
            for i in range(72):
                y, x = divmod(i, 8)
                lossy[y, x] = (i, 0, 0, 255)
            lossy_path = root / "lossy.png"
            Image.fromarray(lossy, "RGBA").save(lossy_path)
            _meta(lossy_path)

            Image.new("RGBA", (129, 8), (1, 2, 3, 255)).save(root / "big.png")
            (root / "bad.png").write_bytes(b"not a png")
            (root / "notes.txt").write_text("skip", encoding="utf-8")

            report = curate_provisional_corpus(root, out)
            rows = load_dataset_manifest(out / "manifest.provisional.jsonl", root=out)
            holdout = load_dataset_manifest(out / "holdout_manifest.jsonl", root=out)
            after = ok.read_bytes()
            smooth_held = (out / "holdout" / "non_pixel_art_smooth" / "smooth.png").exists()

        self.assertEqual(after, before)
        self.assertEqual(report["sprite_sheets_split"], 1)
        self.assertEqual(report["duplicates"], 1)
        self.assertEqual(report["non_pixel_art_smooth"], 1)
        self.assertEqual(report["lossy_gt_64_colors"], 1)
        self.assertEqual(report["oversized_high_res"], 1)
        self.assertEqual(report["corrupt_unsupported"], 2)
        self.assertEqual(len(rows), 3)
        self.assertGreaterEqual(len(holdout), 5)
        self.assertTrue(smooth_held)
        self.assertTrue(all(row["metadata"]["license"] == "CC0-1.0" for row in rows))


if __name__ == "__main__":
    unittest.main()
