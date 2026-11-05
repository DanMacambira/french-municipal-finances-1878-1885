"""No-API regression tests for center-biased dark-gutter splitter."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

SOURCE = Path(__file__).resolve().parents[1] / "code" / "split_spreads.py"
spec = importlib.util.spec_from_file_location("split_spreads", SOURCE)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def spread(width=1200, height=800, crease=.49, dark=True):
    im = Image.new("L", (width, height), 246)
    draw = ImageDraw.Draw(im)
    for x in [50, 100, 160, 220, 280, 340, 405, 455, 520, 670, 740, 800, 900, 960, 1040, 1100]:
        draw.line([(x, 30), (x, 760)], fill=35, width=2)
    for y in range(50, 740, 70):
        draw.line([(30, y), (570, y)], fill=85, width=2)
        draw.line([(630, y), (1160, y)], fill=85, width=2)
    if dark:
        x = int(width * crease)
        draw.rectangle((x - 8, 0, x + 8, height), fill=0)
        draw.rectangle((x - 15, 60, x + 15, height - 50), fill=85)
    return im


def test_picks_uneven_central_crease_not_skinny_lines():
    result = mod.detect_gutter(spread(crease=.485))
    assert result.accepted, result
    assert abs(result.x_ratio - .485) < .015
    assert result.reason == "dark_central_gutter"


def test_does_not_force_a_split_without_gutter():
    result = mod.detect_gutter(spread(dark=False))
    assert not result.accepted


def test_refuses_offcenter_dark_strip():
    result = mod.detect_gutter(spread(crease=.59))
    assert not result.accepted


def test_split_adds_safety_overlap():
    im = spread()
    left, right, x = mod.split_image(im, .49, .012)
    assert 0.45 * im.width < x < 0.55 * im.width
    assert left.width + right.width > im.width
    assert left.height == right.height == im.height


def test_splits_only_confirmed_pdf_pages_and_respects_boundary_sides(tmp_path):
    import pymupdf
    root = tmp_path
    inp = root / "raw"
    inp.mkdir()
    pdf = inp / "volume_1882.pdf"
    with pymupdf.open() as d:
        for _ in range(2):
            p = d.new_page(width=1200, height=800)
            import io
            buf = io.BytesIO()
            spread().save(buf, format="PNG")
            p.insert_image(p.rect, stream=buf.getvalue())
        d.save(pdf)
    config = root / "config.json"
    config.write_text(json.dumps({"1882": {"confirmed": True, "pdf": "volume_1882.pdf",
                                           "start": 1, "end": 2,
                                           "page_sides": {"1": ["right"]}}}))
    rows = mod.process_year(year=1882, requested_pages=[1, 2], dpi=100,
                            config=config, input_dir=inp, output_dir=root / "out",
                            center_min=.45, center_max=.55,
                            overlap_fraction=.012, make_pdfs=False, force_review=False,
                            inner_trim_fraction=0.)
    assert all(r["status"] == "accepted" for r in rows), rows
    out = root / "out" / "1882"
    assert (out / "p0001_right_100dpi.png").exists()
    assert not (out / "p0001_left_100dpi.png").exists()
    assert (out / "p0002_left_100dpi.png").exists()
    assert (out / "p0002_right_100dpi.png").exists()
    assert (out / "review_contact_sheet.jpg").exists()


def test_inner_trim_removes_foreign_page_content_without_overlap():
    im = spread(width=1200, height=800, crease=.50)
    left, right, center = mod.split_image(im, .50, overlap_fraction=0.,
                                         inner_trim_fraction=.006)
    assert center == 600
    assert left.width == 593
    assert right.width == 593
    assert left.width + right.width == im.width - 14
    assert left.height == right.height == im.height
    # Cropped outputs are exact pixel subsets, without resampling/rotation.
    assert np.array_equal(np.asarray(left), np.asarray(im)[:, :593])
    assert np.array_equal(np.asarray(right), np.asarray(im)[:, 607:])


def test_inner_trim_excludes_legacy_overlap_and_unsafe_trims():
    with pytest.raises(ValueError):
        mod.split_image(spread(), .5, .012, .006)
    with pytest.raises(ValueError):
        mod.split_image(spread(), .5, 0., .03)


def test_selected_checkpoint_trim_default_and_exact_crop_boundaries(tmp_path):
    """The pilot-approved 0.001 trim is used both by the CLI and process_year."""
    import inspect
    assert inspect.signature(mod.process_year).parameters["inner_trim_fraction"].default == .001
    assert 'default=.001' in inspect.getsource(mod.main)
    image = spread(width=1200, height=800, crease=.50)
    left, right, x = mod.split_image(image, .50, inner_trim_fraction=.001)
    assert x == 600
    assert left.width == 599 and right.width == 599
    assert left.width + right.width == 1198
