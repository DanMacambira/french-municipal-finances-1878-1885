"""Local, offline tests for the accepted-panel OCR handoff."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from types import ModuleType
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

MODULE = Path(__file__).resolve().parents[1] / 'code' / 'situation_financiere.py'
spec = importlib.util.spec_from_file_location('sf_segmentation', MODULE)
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)


def fixture(tmp_path, monkeypatch):
    root = tmp_path
    raw = root / 'raw_data' / 'situation_financiere_communes'
    raw.mkdir(parents=True)
    volume = raw / 'La_Situation_financiere_des_communes_1882.pdf'
    with pymupdf.open() as pdf:
        for _ in range(3):
            pdf.new_page(width=800, height=550)
        pdf.save(volume)
    year_dir = root / 'derived' / 'situation_financiere' / 'table_panels' / '1882'
    accepted = year_dir / 'accepted'
    review = year_dir / 'needs_review'
    accepted.mkdir(parents=True)
    review.mkdir()
    rows = []
    for page, side, status in [(1, 'right', 'accepted'), (2, 'left', 'accepted'),
                               (2, 'right', 'accepted_extreme_pending_review'),
                               (3, 'left', 'accepted_short_table_pending_review')]:
        row = {'pdf_page': str(page), 'printed_side': side,
               'status': status, 'table_count': '2'}
        for i in (1, 2):
            basename = f'p{page:04d}_{side}_col{i}_200dpi.png'
            file = (accepted if status == 'accepted' else review) / basename
            Image.new('L', (65, 90), color=245-i).save(file)
            row[f'table_{i}'] = str(file)
        rows.append(row)
    with (year_dir / 'table_split_review.csv').open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cfg = root / 'schema' / 'situation_financiere'
    cfg.mkdir(parents=True)
    (cfg / 'page_ranges.json').write_text(json.dumps({'1882': {
        'start': 1, 'end': 3, 'confirmed': True, 'pdf': volume.name,
        'scan_layout': 'spread', 'page_sides': {'1': ['right'], '3': ['left']}}}))
    monkeypatch.setattr(sf, 'ROOT', root)
    monkeypatch.setattr(sf, 'INPUT', raw)
    monkeypatch.setattr(sf, 'DERIVED', root / 'derived' / 'situation_financiere')
    monkeypatch.setattr(sf, 'SEGMENTED_IMAGES', sf.DERIVED / 'table_panels')
    monkeypatch.setattr(sf, 'RANGES', cfg / 'page_ranges.json')
    monkeypatch.setattr(sf, 'MAPPINGS', cfg / 'column_mappings.json')
    return root


def args(**kwargs):
    base = dict(years=[1882], pages=[1, 2, 3], max_pages=0, yes_all=False,
                dpi=200,
                force=False, model='test-model', dry_run=True)
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_list_approved_only_and_keep_two_panels_separate(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    panels, pending = sf.segmented_images(1882, [1, 2, 3], dpi=200)
    assert pending == 2
    assert [(x[0], x[1], x[2]) for x in panels] == [
        (1, 'right', 'col1'), (1, 'right', 'col2'),
        (2, 'left', 'col1'), (2, 'left', 'col2')]
    assert all('/accepted/' in str(x[3]) for x in panels)


def test_manifest_missing_or_bad_dpi_fails_closed(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='dpi'):
        sf.segmented_images(1882, [1], dpi=210)
    with pytest.raises(ValueError, match='missing Stage 2 manifest'):
        sf.segmented_images(1882, [4])


def test_mistral_dry_run_does_not_need_key_or_call_api(tmp_path, monkeypatch, capsys):
    fixture(tmp_path, monkeypatch)
    monkeypatch.delenv('MISTRAL_API_KEY', raising=False)
    monkeypatch.setattr(sf, 'api_ocr', lambda *_: pytest.fail('API must not be invoked'))
    sf.ocr(args())
    out = capsys.readouterr().out
    assert '4 OCR-ready images' in out
    assert 'skipping 2 review-only' in out
    assert 'no OCR requests' in out


def test_cache_isolated_parser_provenance_and_cache_hits(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    # The full project has this module; keep integration test independent of
    # other modules that are not part of this small installation package.
    validator = ModuleType('territorial_validation')
    validator.markdown_table_prefaces = lambda _text: {}
    validator.annotate_territories = lambda raw, candidates: (raw, candidates)
    monkeypatch.setitem(sys.modules, 'territorial_validation', validator)
    monkeypatch.setenv('MISTRAL_API_KEY', 'TEST_NO_NETWORK')
    calls = []
    table = ('<table><tr><th>Commune</th><th>Population</th><th>Revenues</th></tr>'
             '<tr><td>Arras</td><td>100</td><td>200</td></tr></table>')
    def fake_ocr(image, model):
        calls.append(image.name)
        return {'pages': [{'markdown': '', 'tables': [{'id': 't1', 'content': table}]}]}
    monkeypatch.setattr(sf, 'api_ocr', fake_ocr)
    runargs = args(pages=[2], dry_run=False)
    sf.ocr(runargs)
    sf.ocr(runargs)
    assert len(calls) == 2
    location = sf.DERIVED / 'ocr' / 'mistral' / 'segmented' / '1882'
    assert (location / 'p0002_left_col1.json').exists()
    assert (location / 'p0002_left_col2.json').exists()
    # Old half-page output must never leak into a segmented build.
    old = sf.DERIVED / 'ocr' / 'mistral' / '1882'
    old.mkdir(parents=True)
    (old / 'p0002_left.json').write_text(json.dumps({'year':1882,'pdf_page':2,
        'side':'left','image':'legacy.png','source_pdf':'legacy.pdf',
        'model':'legacy','response':{'pages':[{'tables':[{'content':table}]}]}}))
    raw, candidates, inventory, audit = sf.parse_cache([1882])
    assert len(candidates) == 2
    assert candidates['row_id'].is_unique
    assert set(candidates['panel']) == {'col1', 'col2'}
    assert len(audit) == 2
    assert all('table_panels/' in x for x in raw['source_image'])
    # The old cache directory is ignored by design; only segmented paths are read.
    assert len(sf.parse_cache([1882])[1]) == 2


def test_manifest_not_forged_by_accepted_directory_contents(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    orphan = sf.SEGMENTED_IMAGES / '1882' / 'accepted' / 'p0003_left_col1_200dpi.png'
    Image.new('L', (80, 80), 255).save(orphan)
    files, pending = sf.segmented_images(1882, [3])
    assert files == [] and pending == 1


def test_modified_image_invalidates_segmented_cache(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    monkeypatch.setenv('MISTRAL_API_KEY', 'TEST_NO_NETWORK')
    calls = []
    monkeypatch.setattr(sf, 'api_ocr', lambda image, model: (calls.append(image.name) or
                        {'pages': [{'markdown': '', 'tables': []}]}))
    runargs = args(pages=[2], dry_run=False)
    sf.ocr(runargs)
    assert len(calls) == 2
    sf.ocr(runargs)
    assert len(calls) == 2
    accepted = sf.SEGMENTED_IMAGES / '1882' / 'accepted' / 'p0002_left_col1_200dpi.png'
    Image.new('L', (65, 90), 66).save(accepted)
    sf.ocr(runargs)
    assert len(calls) == 3


def test_dry_run_never_renders_pdf_regions(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    assert not hasattr(sf, 'render_regions')
    monkeypatch.setattr(sf, 'api_ocr', lambda *_args: pytest.fail('no network calls'))
    sf.ocr(args(pages=[2]))
