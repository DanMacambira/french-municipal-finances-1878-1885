"""Offline tests for experimental tax-only Mistral crop, no API calls."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "code" / "tax_ocr_pilot.py"
spec = importlib.util.spec_from_file_location("tax_ocr_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_two_strips_keep_the_original_y_and_pixel_values():
    src = Image.new("RGB", (100, 40), "white")
    for y in range(40):
        for x in range(100):
            src.putpixel((x, y), (x, y * 5, 0))
    out = pilot.crop_strips(src, .3, .7, .9, gap=12)
    assert out.size == (30 + 12 + 20, 40)
    for y in (0, 6, 39):
        assert out.getpixel((5, y)) == src.getpixel((5, y))
        assert out.getpixel((42, y)) == src.getpixel((70, y))
        assert out.getpixel((31, y)) == (255, 255, 255)


def test_reject_overlapping_regions():
    with pytest.raises(ValueError):
        pilot.crop_strips(Image.new("RGB", (100, 100)), .9, .8, 1.0)


def test_preview_does_not_use_api_key_or_write_ocr(tmp_path, monkeypatch):
    src, _, target_cache = pilot.pilot_paths(tmp_path, 1882, 245, "left_col2", 210)
    src.parent.mkdir(parents=True)
    Image.new("RGB", (679, 150), "white").save(src)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    args = SimpleNamespace(command="preview", year=1882, page=245,
                           panels=["left_col2"], dpi=210, yes=False, force=False)
    pilot.run(args, root=tmp_path)
    assert not target_cache.exists()
    assert pilot.pilot_paths(tmp_path, 1882, 245, "left_col2", 210)[1].exists()


def test_ocr_cached_request_is_not_billed_twice(tmp_path, monkeypatch):
    src, _, cache = pilot.pilot_paths(tmp_path, 1882, 245, "right_col2", 210)
    src.parent.mkdir(parents=True)
    Image.new("RGB", (729, 100), "white").save(src)
    monkeypatch.setenv("MISTRAL_API_KEY", "not-a-real-key")
    args = SimpleNamespace(command="ocr", year=1882, page=245,
                           panels=["right_col2"], dpi=210, yes=True, force=False)
    image, _, digest = pilot.prepare(tmp_path, 1882, 245, "right_col2", 210)
    cache.parent.mkdir(parents=True)
    cache.write_text(__import__("json").dumps({"image_sha256": digest, "model": pilot.MODEL}))
    monkeypatch.setattr(pilot, "call_ocr", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected paid OCR")))
    pilot.run(args, root=tmp_path)
