"""No-network tests for the parallel OCR runner."""
from __future__ import annotations
import json
import types
from contextlib import nullcontext
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
import parallel_ocr as c


class ConcurrentOCRTests(unittest.TestCase):
    def test_pacer_unlimited(self):
        c.RequestPacer(0).wait()

    def test_cache_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            job = c.OCRJob(1882, 245, "right", "col1", base / "panel.png", "year.pdf",
                           "derived/panel.png", "abc", base / "cache.json")
            self.assertFalse(c.cache_matches(job, "mistral-ocr-4-1"))
            job.cache_path.write_text(json.dumps({
                "image_sha256": "abc", "model": "mistral-ocr-4-1",
                "segmentation_source": "segmented", "panel": "col1",
                "image": "derived/panel.png", "source_pdf": "year.pdf",
            }), encoding="utf-8")
            self.assertTrue(c.cache_matches(job, "mistral-ocr-4-1"))
            self.assertFalse(c.cache_matches(job, "different-model"))

    def test_parallel_saves_same_cache_schema_without_api(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            for i in range(4):
                (base / f"img{i}.png").write_bytes(b"test")
            jobs = [c.OCRJob(1882, i, "single", "col1", base / f"img{i}.png", "year.pdf",
                             f"test/img{i}.png", f"sha{i}", base / f"cache{i}.json")
                    for i in range(4)]
            state = {"active": 0, "peak": 0}
            lock = threading.Lock()

            def fake_api(image, model):
                with lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                time.sleep(0.035)
                with lock:
                    state["active"] -= 1
                return {"pages": [{"markdown": "mock"}]}

            with patch.object(c.sf, "api_ocr", side_effect=fake_api):
                errors = c.run_jobs(jobs, workers=3, model="mistral-ocr-4-1")
            self.assertEqual(errors, [])
            self.assertGreater(state["peak"], 1)
            self.assertLessEqual(state["peak"], 3)
            for job in jobs:
                result = json.loads(job.cache_path.read_text(encoding="utf-8"))
                self.assertEqual(result["image_sha256"], job.checksum)
                self.assertEqual(result["model"], "mistral-ocr-4-1")
                self.assertEqual(result["response"]["pages"][0]["markdown"], "mock")

    def test_gather_jobs_validates_cache_before_queuing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image = root / "derived" / "situation_financiere" / "table_panels" / "1882" / "accepted" / "p0001_single_col1_200dpi.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"fixture-image")
            pdf = root / "raw_data" / "year_1882.pdf"
            args = types.SimpleNamespace(pages=None, max_pages=0, dpi=200,
                                         force=False, model="mistral-ocr-4-1")
            with patch.object(c.sf, "ROOT", root), \
                 patch.object(c.sf, "DERIVED", root / "derived" / "situation_financiere"), \
                 patch.object(c.sf, "find_pdfs", return_value={1882: pdf}, create=True), \
                 patch.object(c.sf, "select_years", return_value=[1882], create=True), \
                 patch.object(c.sf, "valid_range", return_value=(1, 1), create=True), \
                 patch.object(c.sf, "segmented_images", return_value=([(1, "single", "col1", image)], 0), create=True), \
                 patch.object(c.fitz, "open", return_value=nullcontext(object())):
                jobs, total, hits, pending = c.gather_jobs(args)
                self.assertEqual((len(jobs), total, hits, pending), (1, 1, 0, 0))
                response = {"year": 1882, "pdf_page": 1, "side": "single", "panel": "col1",
                            "segmentation_source": "segmented", "source_pdf": pdf.name,
                            "image": jobs[0].relative_image, "image_sha256": jobs[0].checksum,
                            "model": args.model, "response": {"pages": []}}
                c.sf.atomic_json(jobs[0].cache_path, response)
                jobs, total, hits, pending = c.gather_jobs(args)
                self.assertEqual((len(jobs), total, hits, pending), (0, 1, 1, 0))

    def test_failures_preserve_successes_for_retries(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            jobs = [c.OCRJob(1882, i, "single", "col1", base / f"img{i}.png", "year.pdf",
                             f"test/img{i}.png", f"sha{i}", base / f"cache{i}.json")
                    for i in range(3)]
            def fake_api(image, model):
                if "img1" in image.name:
                    raise RuntimeError("simulated 429 exhaustion")
                return {"pages": []}
            with patch.object(c.sf, "api_ocr", side_effect=fake_api):
                errors = c.run_jobs(jobs, workers=2, model="mistral-ocr-4-1")
            self.assertEqual(len(errors), 1)
            self.assertTrue(jobs[0].cache_path.exists())
            self.assertFalse(jobs[1].cache_path.exists())
            self.assertTrue(jobs[2].cache_path.exists())
