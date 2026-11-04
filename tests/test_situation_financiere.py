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
        # Only left half has data; right half transcription is empty.
        tables = [{"id": "0", "content": html}] if "left" in image.name else []
        return {"model": model, "pages": [{"index": 0, "markdown": "CANTON D'ARRAS",
                 "tables": tables, "confidence_scores": {"average_page_confidence_score": .95}}]}
    monkeypatch.setattr(sf, "api_ocr", mock_api)
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy-for-mocked-tests")
    sf.ocr(SimpleNamespace(years=[1879], max_pages=1, yes_all=False, engine="mistral",
                           model=sf.MODEL, dpi=120, force=False))
    assert len(calls) == 2
    sf.ocr(SimpleNamespace(years=[1879], max_pages=1, yes_all=False, engine="mistral",
                           model=sf.MODEL, dpi=120, force=False))
    assert len(calls) == 2  # cache hit
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
