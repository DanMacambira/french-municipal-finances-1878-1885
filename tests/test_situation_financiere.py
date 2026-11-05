"""Offline parser and handoff regression tests; no paid OCR calls."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pymupdf as fitz
import pytest
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "code" / "situation_financiere.py"
spec = importlib.util.spec_from_file_location("sf_finance_tests", SRC)
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)


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


def test_ambiguous_year_layout_not_guessed():
    mappings = {"1880": {"layouts": [
        {"name": "A", "column_count": 4}, {"name": "B", "column_count": 4}]}}
    assert sf.layout_for(1880, 4, mappings, 100, "left") is None
    mappings["1880"]["page_layout_overrides"] = {"p100_left": "B"}
    assert sf.layout_for(1880, 4, mappings, 100, "left")["name"] == "B"


def test_only_approved_panels_feed_financial_builder(tmp_path, monkeypatch):
    root = tmp_path
    source_dir = root / "raw_data" / "situation_financiere_communes"
    source_dir.mkdir(parents=True)
    with fitz.open() as pdf:
        pdf.new_page(width=595, height=842)
        pdf.save(source_dir / "volume_1879.pdf")
    cfg = root / "schema" / "situation_financiere"
    cfg.mkdir(parents=True)
    ranges = cfg / "page_ranges.json"
    ranges.write_text(json.dumps({"1879": {"pdf": "volume_1879.pdf", "start": 1,
                                             "end": 1, "confirmed": True, "scan_layout": "single"}}))
    mappings = cfg / "column_mappings.json"
    mappings.write_text(json.dumps({"1879": {"layouts": [{"name": "main", "column_count": 4,
        "columns": {"2": {"name": "population_sf", "type": "integer"},
                    "3": {"name": "centime_value_sf", "type": "decimal_fr"},
                    "4": {"name": "ordinary_centimes_sf", "type": "integer"}}}]}}))
    derived = root / "derived" / "situation_financiere"
    year_folder = derived / "table_panels" / "1879"
    approved = year_folder / "accepted"
    approved.mkdir(parents=True)
    import csv
    rows = []
    for col in (1, 2):
        file = approved / f"p0001_single_col{col}_200dpi.png"
        Image.new("L", (45, 55), 235).save(file)
    rows.append({"pdf_page": "1", "printed_side": "single", "status": "accepted",
                 "table_count": "2", "table_1": str(approved / "p0001_single_col1_200dpi.png"),
                 "table_2": str(approved / "p0001_single_col2_200dpi.png")})
    with (year_folder / "table_split_review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(sf, "ROOT", root)
    monkeypatch.setattr(sf, "INPUT", source_dir)
    monkeypatch.setattr(sf, "DERIVED", derived)
    monkeypatch.setattr(sf, "RANGES", ranges)
    monkeypatch.setattr(sf, "MAPPINGS", mappings)
    monkeypatch.setattr(sf, "SEGMENTED_IMAGES", derived / "table_panels")
    monkeypatch.setenv("MISTRAL_API_KEY", "DUMMY_FOR_MOCKED_TEST")
    html = """<table><tr><th>COMMUNES</th><th>pop</th><th>value</th><th>centimes</th></tr>
    <tr><td>Arras</td><td>1,506</td><td>17,35</td><td>42</td></tr>
    <tr><td>Lens</td><td>1S78</td><td>32,99</td><td>56</td></tr>
    <tr><td>TOTAL</td><td>9999</td><td>999</td><td>98</td></tr></table>"""
    calls = []
    def mock_ocr(image, model):
        calls.append(image.name)
        tables = [{"id": "0", "content": html}] if "col1" in image.name else []
        return {"pages": [{"markdown": "CANTON D'ARRAS", "tables": tables,
                            "confidence_scores": {"average_page_confidence_score": .95}}]}
    monkeypatch.setattr(sf, "api_ocr", mock_ocr)
    args = SimpleNamespace(years=[1879], max_pages=1, yes_all=False,
                           model=sf.MODEL, dpi=200, force=False, dry_run=False,
                           pages=None)
    sf.ocr(args)
    sf.ocr(args)
    assert len(calls) == 2  # second run entirely cached
    # Dependencies are independently tested; stub only their external effects here.
    validator = ModuleType("territorial_validation")
    validator.markdown_table_prefaces = lambda _text: {}
    validator.annotate_territories = lambda raw, communes: (raw, communes)
    validator.canton_subtotal_checks = lambda *a, **k: pd.DataFrame()
    parser = ModuleType("extraordinary_tax_parser")
    parser.enrich_with_extraordinary_taxes = lambda communes, raw: (
        communes, pd.DataFrame(), pd.DataFrame())
    monkeypatch.setitem(sys.modules, "territorial_validation", validator)
    monkeypatch.setitem(sys.modules, "extraordinary_tax_parser", parser)
    sf.build(SimpleNamespace(years=[1879], dpi=200, no_match=True, min_confidence=.75))
    out = pd.read_csv(derived / "output" / "situation_financiere_commune_year_candidates.csv")
    assert len(out) == 2
    assert out.loc[0, "population_sf"] == 1506
    assert out.loc[0, "centime_value_sf"] == 17.35
    assert pd.isna(out.loc[1, "population_sf"])
    assert "ocr_uncertain" in out.loc[1, "review_flags"]
    assert set(out["panel"]) == {"col1"}


def test_obsolete_pdf_rendering_and_ocr_source_switch_removed():
    assert not hasattr(sf, "render_regions")
    assert not hasattr(sf, "regions_for_page")
    assert not hasattr(sf, "discover")
    assert not hasattr(sf, "preview")
