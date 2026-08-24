import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from fenpix.corpus import prepare_corpus
from fenpix.dataset import load_dataset_manifest


SAMPLES = Path(__file__).parent / "sample_data" / "kenney_tiny_town"


class PrepareCorpusTest(unittest.TestCase):
    def test_prepares_balanced_licensed_corpus_from_local_and_zip_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / "sample.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.write(SAMPLES / "tile_0000.png", "pack/tile_0000.png")
                zf.write(SAMPLES / "tile_0001.png", "pack/tile_0001.png")
            config = tmp / "corpus.json"
            out = tmp / "corpus"
            config.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "name": "Kenney Tiny Town",
                                "path": str(SAMPLES),
                                "license": "Creative Commons CC0",
                                "source_url": "https://kenney.nl/assets/tiny-town",
                                "category": "tile",
                                "tags": ["kenney", "tile"],
                            },
                            {
                                "name": "Zip Pack",
                                "path": str(archive),
                                "license": "CC0-1.0",
                                "source_url": "local",
                                "category": "object",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = prepare_corpus(config, out, target_count=20, min_count=1, compose_scenes=1)
            rows = load_dataset_manifest(out / "manifest.jsonl", root=out / "assets")

        self.assertEqual(report["status"], "ready")
        self.assertGreaterEqual(report["total_accepted"], 14)
        self.assertGreaterEqual(report["composition"]["created"], 1)
        self.assertTrue(rows)
        self.assertTrue(all(row["license"] in {"Creative Commons CC0", "CC0-1.0"} for row in rows))
        self.assertTrue(all(not row["lossy"] for row in rows))
        self.assertTrue(all(not row["duplicate_of"] for row in rows))
        self.assertTrue(all("source" in row["metadata"] for row in rows))

    def test_rejects_disallowed_license_before_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = tmp / "corpus.json"
            out = tmp / "corpus"
            config.write_text(
                json.dumps({"sources": [{"name": "Bad", "path": str(SAMPLES), "license": "CC-BY-NC"}]}),
                encoding="utf-8",
            )

            report = prepare_corpus(config, out, min_count=1)

        self.assertEqual(report["status"], "below_min_count")
        self.assertEqual(report["sources"][0]["rejection_reasons"], {"license_not_allowed": 1})
        self.assertEqual(report["curation"]["selected"], 0)


if __name__ == "__main__":
    unittest.main()
