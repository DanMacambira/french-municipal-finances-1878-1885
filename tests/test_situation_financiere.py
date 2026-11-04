"""Offline tests; never call Mistral or require the national PDFs."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import fitz
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "code" / "situation_financiere.py"
spec = importlib.util.spec_from_file_location("situation_financiere", SRC)
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(sf, "ROOT", tmp_path)
    monkeypatch.setattr(sf, "INPUT", tmp_path / "raw_data" / "situation_financiere_communes")
    monkeypatch.setattr(sf, "DERIVED", tmp_path / "derived" / "situation_financiere")
    monkeypatch.setattr(sf, "CONFIG", tmp_path / "schema" / "situation_financiere")
    monkeypatch.setattr(sf, "RANGES", sf.CONFIG / "page_ranges.json")
    monkeypatch.setattr(sf, "MAPPINGS", sf.CONFIG / "column_mappings.json")
    sf.INPUT.mkdir(parents=True)
    sf.CONFIG.mkdir(parents=True)
    return tmp_path


def test_html_grid_header_spans_and_numeric_classification():
    html = '''<table><tr><th rowspan="2">COMMUNES</th><th colspan="2">CENTIMES</th></tr>
    <tr><th>ordinary</th><th>extraordinary</th></tr>
    <tr><td>Arras</td><td>29</td><td>101</td></tr></table>'''
    grid = sf.html_grid(html)
    assert grid[0] == ["COMMUNES", "CENTIMES", "CENTIMES"]
    assert grid[1] == ["COMMUNES", "ordinary", "extraordinary"]
    assert grid[2] == ["Arras", "29", "101"]
    assert sf.row_type(grid[2]) == "candidate_commune"
    assert sf.row_type(["CANTON D'ARRAS", "", ""]) == "header"


def test_french_numbers_are_strict():
    assert sf.french_number("1,506", "integer") == (1506, "parsed")
    assert sf.french_number("17,35", "decimal_fr") == (17.35, "parsed")
    assert sf.french_number("1S78", "integer") == (None, "ocr_uncertain")
    assert sf.french_number("1,506", "decimal_fr") == (None, "numeric_ambiguous")
    assert sf.french_number("»", "integer") == (None, "missing_or_symbol")


def test_discover_and_confirm_range(project):
    f = sf.INPUT / "situation_financiere_1878.pdf"
    pdf = fitz.open()
    for t in ["DEPARTEMENT DE L'AIN", "DEPARTEMENT DU PAS-DE-CALAIS",
              "COMMUNES d Arras", "DEPARTEMENT DE LA SOMME"]:
        page = pdf.new_page(width=595, height=842)
        page.insert_text((60, 70), t, fontsize=15)
    pdf.save(str(f))
    sf.discover(SimpleNamespace(years=[1878], header_ocr=False))
    data = sf.read_json(sf.RANGES)
    assert data["1878"]["start"] == 2
    assert data["1878"]["end"] == 3
    assert data["1878"]["confirmed"] is False
    data["1878"]["confirmed"] = True
    sf.atomic_json(sf.RANGES, data)
    with fitz.open(f) as src:
        assert sf.valid_range(1878, src) == (2, 3)


def test_ocr_cache_and_build_with_year_specific_schema(project, monkeypatch):
    pdf = fitz.open()
    page = pdf.new_page(width=595, height=842)
    page.insert_text((60, 70), "DEPARTEMENT DU PAS-DE-CALAIS", fontsize=12)
    pdf.save(str(sf.INPUT / "volume_1879.pdf"))
    sf.atomic_json(sf.RANGES, {"1879": {"pdf": "volume_1879.pdf", "start": 1,
                                      "end": 1, "confirmed": True}})
    sf.atomic_json(sf.MAPPINGS, {"1879": {"layouts": [{"name": "main", "column_count": 4,
        "columns": {"2": {"name": "population_sf", "type": "integer"},
                    "3": {"name": "centime_value_sf", "type": "decimal_fr"},
                    "4": {"name": "ordinary_centimes_sf", "type": "integer"}}}]}})
    html = """<table><tr><th>COMMUNES</th><th>pop</th><th>value</th><th>centimes</th></tr>
    <tr><td>Arras</td><td>1,506</td><td>17,35</td><td>42</td></tr>
    <tr><td>Lens</td><td>1S78</td><td>32,99</td><td>56</td></tr>
    <tr><td>TOTAL</td><td>9999</td><td>999</td><td>98</td></tr></table>"""
    calls = []
    def mock_api(image, model):
        calls.append((image, model))
        # A single printed page must not be split into two OCR calls.
        tables = [{"id": "0", "content": html}] if "full" in image.name else []
        return {"model": model, "pages": [{"index": 0, "markdown": "CANTON D'ARRAS",
                 "tables": tables, "confidence_scores": {"average_page_confidence_score": .95}}]}
    monkeypatch.setattr(sf, "api_ocr", mock_api)
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy-for-mocked-tests")
    sf.ocr(SimpleNamespace(years=[1879], max_pages=1, yes_all=False, engine="mistral",
                           model=sf.MODEL, dpi=120, force=False))
    assert len(calls) == 1
    sf.ocr(SimpleNamespace(years=[1879], max_pages=1, yes_all=False, engine="mistral",
                           model=sf.MODEL, dpi=120, force=False))
    assert len(calls) == 1  # cache hit
    sf.build(SimpleNamespace(engine="mistral", years=[1879], no_match=True,
                             min_confidence=.75))
    out = pd.read_csv(sf.DERIVED / "output" / "situation_financiere_commune_year_candidates.csv")
    assert len(out) == 2
    assert out.loc[0, "population_sf"] == 1506
    assert out.loc[0, "centime_value_sf"] == 17.35
    assert pd.isna(out.loc[1, "population_sf"])
    assert "ocr_uncertain" in out.loc[1, "review_flags"]
    assert (sf.DERIVED / "output" / "situation_financiere_all_rows.csv").exists()


def test_ambiguous_year_layout_not_guessed():
    m = {"1880": {"layouts": [
        {"name": "A", "column_count": 4}, {"name": "B", "column_count": 4}]}}
    assert sf.layout_for(1880, 4, m, 100, "left") is None
    m["1880"]["page_layout_overrides"] = {"p100_left": "B"}
    assert sf.layout_for(1880, 4, m, 100, "left")["name"] == "B"


def test_spread_boundary_excludes_previous_department(project, monkeypatch, capsys):
    """Only the Pas-de-Calais right half of a mixed spread is sent to OCR."""
    pdf = fitz.open()
    page = pdf.new_page(width=1000, height=700)
    page.insert_text((50, 50), "DEPARTEMENT DE L'ORNE", fontsize=13)
    page.insert_text((550, 50), "DEPARTEMENT DU PAS-DE-CALAIS", fontsize=13)
    pdf.save(str(sf.INPUT / "volume_1882.pdf"))
    sf.atomic_json(sf.RANGES, {"1882": {"pdf": "volume_1882.pdf", "start": 1,
        "end": 1, "confirmed": True, "page_sides": {"1": ["right"]}}})
    sf.plan(SimpleNamespace(years=[1882]))
    assert "1 source pages, 1 image OCR requests" in capsys.readouterr().out
    calls = []
    def mock_api(image, model):
        calls.append(image.name)
        return {"pages": [{"index": 0, "markdown": "", "tables": []}]}
    monkeypatch.setattr(sf, "api_ocr", mock_api)
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy-for-mocked-tests")
    sf.ocr(SimpleNamespace(years=[1882], max_pages=1, yes_all=False, engine="mistral",
                           model=sf.MODEL, dpi=120, force=False))
    assert len(calls) == 1
    assert "right" in calls[0]
    assert not list((sf.DERIVED / "ocr" / "mistral" / "1882").glob("*left*"))


def test_invalid_spread_side_rejected(project):
    sf.atomic_json(sf.RANGES, {"1880": {"page_sides": {"17": ["middle"]}}})
    with pytest.raises(ValueError, match="invalid page_sides"):
        sf.sides_for_page(1880, 17)


def test_auto_layout_counts_and_override(project, capsys):
    pdf = fitz.open()
    pdf.new_page(width=595, height=842)   # single portrait scan
    pdf.new_page(width=1200, height=800)  # landscape two-page spread
    pdf.save(str(sf.INPUT / "volume_1880.pdf"))
    sf.atomic_json(sf.RANGES, {"1880": {"pdf": "volume_1880.pdf", "start": 1,
                                      "end": 2, "confirmed": True}})
    with fitz.open(sf.INPUT / "volume_1880.pdf") as doc:
        assert sf.regions_for_page(1880, 1, doc[0]) == ("full",)
        assert sf.regions_for_page(1880, 2, doc[1]) == ("left", "right")
    sf.plan(SimpleNamespace(years=[1880]))
    printed = capsys.readouterr().out
    assert "2 source pages, 3 image OCR requests" in printed
    assert "Single scans" in printed and "Spreads" in printed
    cfg = sf.read_json(sf.RANGES)
    cfg["1880"]["scan_layout_overrides"] = {"2": "single"}
    sf.atomic_json(sf.RANGES, cfg)
    with fitz.open(sf.INPUT / "volume_1880.pdf") as doc:
        assert sf.regions_for_page(1880, 2, doc[1]) == ("full",)


def test_boundary_and_rendered_regions(project):
    pdf = fitz.open()
    pdf.new_page(width=1200, height=800)
    pdf.save(str(sf.INPUT / "volume_1882.pdf"))
    sf.atomic_json(sf.RANGES, {"1882": {"pdf": "volume_1882.pdf", "start": 1,
        "end": 1, "confirmed": True, "page_sides": {"1": ["right"]}}})
    with fitz.open(sf.INPUT / "volume_1882.pdf") as doc:
        regions = sf.regions_for_page(1882, 1, doc[0])
        assert regions == ("right",)
        rendered = sf.render_regions(doc[0], 1882, 1, 72, regions)
        assert len(rendered) == 1
        assert rendered[0][0] == "right"
        from PIL import Image
        with Image.open(rendered[0][1]) as im:
            assert 600 <= im.width <= 620
            assert im.height == 800


def test_conflicting_page_sides_and_single_scan(project):
    sf.atomic_json(sf.RANGES, {"1883": {"page_sides": {"12": ["right"]}}})
    pdf = fitz.open()
    pdf.new_page(width=595, height=842)
    with pytest.raises(ValueError, match="page_sides requires a spread"):
        sf.regions_for_page(1883, 12, pdf[0])


def test_preview_actual_ocr_regions(project):
    pdf = fitz.open()
    pdf.new_page(width=595, height=842)
    pdf.new_page(width=1200, height=800)
    pdf.save(str(sf.INPUT / "volume_1885.pdf"))
    sf.atomic_json(sf.RANGES, {"1885": {"pdf": "volume_1885.pdf", "start": 1,
        "end": 2, "confirmed": True, "page_sides": {"2": ["right"]}}})
    sf.preview(SimpleNamespace(years=[1885], center=None, radius=5,
                               every=1, regions=True))
    assert (sf.DERIVED / "previews" / "1885_p1-2_regions.jpg").exists()
    images = list((sf.DERIVED / "images" / "1885").glob("*.png"))
    assert len(images) == 2
    assert any("full" in im.name for im in images)
    assert any("right" in im.name for im in images)


def test_parallel_table_panels_plan_preview_and_build(project, monkeypatch, capsys):
    """One printed page yields 2 tables; a selected half-spread yields 2."""
    pdf = fitz.open()
    pdf.new_page(width=1000, height=1400)  # single-page scan
    pdf.new_page(width=1200, height=800)   # two-page spread
    pdf.save(str(sf.INPUT / "volume_1882.pdf"))
    sf.atomic_json(sf.RANGES, {"1882": {"pdf": "volume_1882.pdf", "start": 1,
        "end": 2, "confirmed": True, "page_sides": {"2": ["right"]},
        "table_panels": 2, "table_panel_split_overrides": {
            "p1_full": 0.54, "p2_right": 0.485}}})
    with fitz.open(sf.INPUT / "volume_1882.pdf") as doc:
        assert sf.regions_for_page(1882, 1, doc[0]) == ("full_col1", "full_col2")
        assert sf.regions_for_page(1882, 2, doc[1]) == ("right_col1", "right_col2")
        images = sf.render_regions(doc[0], 1882, 1, 72, ("full_col1", "full_col2"))
        from PIL import Image
        with Image.open(images[0][1]) as left:
            assert 540 < left.width < 550
        with Image.open(images[1][1]) as right:
            assert 460 < right.width < 480
    sf.plan(SimpleNamespace(years=[1882]))
    assert "2 source pages, 4 image OCR requests" in capsys.readouterr().out
    sf.preview(SimpleNamespace(years=[1882], center=1, radius=1,
                               every=1, regions=True))
    assert (sf.DERIVED / "previews" / "1882_p1-2_regions.jpg").exists()
    calls = []

    def mock_api(image, model):
        calls.append(image.name)
        name = image.name.split("_")[1:3]
        html = f"<table><tr><th>COMMUNES</th><th>Pop</th><th>Sup</th></tr>" \
               f"<tr><td>{'-'.join(name)}</td><td>123</td><td>456</td></tr></table>"
        return {"pages": [{"index": 0, "markdown": "", "tables": [{"content": html}]}]}

    monkeypatch.setattr(sf, "api_ocr", mock_api)
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy-for-mocked-tests")
    args = SimpleNamespace(years=[1882], max_pages=0, yes_all=True, pages=None,
                           engine="mistral", model=sf.MODEL, dpi=72, force=False)
    sf.ocr(args)
    assert len(calls) == 4
    sf.ocr(args)
    assert len(calls) == 4  # calls are cached

    # An obsolete OCR of the unsplit page must be excluded from build.
    old = {"year": 1882, "pdf_page": 1, "side": "full", "source_pdf": "volume_1882.pdf",
           "image": "derived/situation_financiere/images/1882/p0001_full_72dpi.png",
           "image_sha256": "obsolete", "model": sf.MODEL, "response": {"pages": []}}
    sf.atomic_json(sf.DERIVED / "ocr" / "mistral" / "1882" / "p0001_full.json", old)
    raw, candidates, inventory, audit = sf.parse_cache("mistral", [1882])
    assert len(candidates) == 4
    assert sorted(candidates["side"].tolist()) == sorted([
        "full_col1", "full_col2", "right_col1", "right_col2"])
    assert len(audit) == 4

    # A new split position invalidates prior panel OCR until it is rerun.
    specs = sf.read_json(sf.RANGES)
    specs["1882"]["table_panel_split_overrides"]["p1_full"] = 0.52
    sf.atomic_json(sf.RANGES, specs)
    assert sf.region_signature(1882, 1, "full_col1") is not None
    raw, candidates, inventory, audit = sf.parse_cache("mistral", [1882])
    assert len(candidates) == 2  # the right-half spread remains active
    assert set(candidates["side"]) == {"right_col1", "right_col2"}


def test_invalid_parallel_table_config(project):
    sf.atomic_json(sf.RANGES, {"1881": {"table_panels": 3}})
    with pytest.raises(ValueError, match="table_panels must be 1 or 2"):
        sf.table_panels_for_page(1881, 1)
    sf.atomic_json(sf.RANGES, {"1881": {"table_panel_split": 0.8}})
    with pytest.raises(ValueError, match="invalid table panel split"):
        sf.table_panel_split_for_page(1881, 1, "full")
