"""Integration tests for multiprocessing, review folders, and quiet progress output."""
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_page(blank=False):
    w, h = 900, 1250
    pixels = np.full((h, w), 250, dtype=np.uint8)
    if not blank:
        x = 440
        cv2.line(pixels, (12, 90), (885, 90), 5, 3)
        cv2.line(pixels, (12, 210), (885, 210), 5, 2)
        for xx in [12, x-2, x+2, 885, x-150, x-110, x-70, x-35,
                   x+150, x+205, x+250, x+300]:
            cv2.line(pixels, (xx, 90), (xx, 1160), 5, 2)
    return Image.fromarray(pixels)


def _write_test_volume(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    filename = raw / '1878.pdf'
    with pymupdf.open() as doc:
        for blank in (False, False, True):
            image = _synthetic_page(blank)
            stream = io.BytesIO()
            image.save(stream, format='PNG')
            page = doc.new_page(width=image.width, height=image.height)
            page.insert_image(page.rect, stream=stream.getvalue())
        doc.save(filename)
    config = tmp_path / 'pages.json'
    config.write_text(json.dumps({'1878': {'pdf': '1878.pdf', 'start': 1,
                          'end': 3, 'confirmed': True, 'scan_layout': 'single'}}))
    return raw, config


def _run(script, *params):
    p = subprocess.run([sys.executable, str(ROOT / 'code' / script), *map(str, params)],
                       cwd=ROOT, text=True, capture_output=True, timeout=90)
    assert p.returncode == 0, p.stdout + '\n' + p.stderr
    assert 'elapsed' in p.stderr
    return p


def _rows(file):
    with file.open(newline='', encoding='utf-8') as src:
        return list(csv.DictReader(src))


def test_serial_and_parallel_have_same_cuts_and_review_outputs(tmp_path):
    raw, config = _write_test_volume(tmp_path)
    summaries = []
    for workers in (1, 2):
        root = tmp_path / f'work{workers}'
        pages = root / 'split_pages'
        panels = root / 'table_panels'
        run1 = _run('split_spreads.py', '--year', 1878, '--workers', workers,
                    '--dpi', 100, '--input-dir', raw, '--config', config,
                    '--output-dir', pages)
        assert 'Single-page passthrough' in run1.stdout
        assert (pages / '1878' / 'accepted' / 'p0001_single_100dpi.png').exists()
        assert not (pages / '1878' / 'needs_review' / 'p0001_source_100dpi.png').exists()
        run2 = _run('split_tables.py', '--year', 1878, '--workers', workers,
                    '--dpi', 100, '--input-root', pages, '--output-root', panels)
        assert 'Pass 1: strict' in run2.stdout
        assert (panels / '1878' / 'accepted' / 'p0001_single_col1_100dpi.png').exists()
        assert (panels / '1878' / 'needs_review' / 'p0003_single_source_100dpi.png').exists()
        assert (panels / '1878' / 'needs_review' / 'review_contact_sheet.jpg').exists()
        assert (panels / 'all_years_table_split_review.csv').exists()
        statuses = [(r['pdf_page'], r['status'], r['cut_method'], r['detection_pass'])
                    for r in _rows(panels / '1878' / 'table_split_review.csv')]
        # Check actual crop pixels, not only statuses.
        im = Image.open(panels / '1878' / 'accepted' / 'p0001_single_col2_100dpi.png')
        summaries.append((statuses, np.asarray(im).copy()))
    assert summaries[0][0] == summaries[1][0]
    assert np.array_equal(summaries[0][1], summaries[1][1])


def test_contact_sheet_pagination_uses_bounded_sheets(tmp_path):
    sys.path.insert(0, str(ROOT / 'code'))
    from segmentation_runtime import contact_sheets
    files = []
    for i in range(51):
        p = tmp_path / f'page_{i:03d}.jpg'
        Image.new('RGB', (80, 100), 'white').save(p)
        files.append(p)
    outputs = contact_sheets(files, tmp_path / 'sheets', max_per_sheet=20)
    assert len(outputs) == 3
    assert all(p.is_file() for p in outputs)
    with Image.open(outputs[0]) as im:
        assert im.height <= 7 * 440


def test_parallel_spread_review_is_isolated_from_accepted(tmp_path):
    from PIL import ImageDraw
    raw = tmp_path / 'raw'
    raw.mkdir()
    pdf = raw / '1882.pdf'
    good = Image.new('L', (1200, 800), 248)
    draw = ImageDraw.Draw(good)
    draw.rectangle((593, 0, 607, 799), fill=0)
    draw.rectangle((586, 40, 614, 769), fill=40)
    bad = Image.new('L', (1200, 800), 249)
    with pymupdf.open() as doc:
        for im in [good, bad]:
            data = io.BytesIO()
            im.save(data, format='PNG')
            pg = doc.new_page(width=1200, height=800)
            pg.insert_image(pg.rect, stream=data.getvalue())
        doc.save(pdf)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'1882': {'confirmed': True, 'pdf': '1882.pdf',
                       'scan_layout': 'spread', 'start': 1, 'end': 2}}))
    output = tmp_path / 'split_pages'
    _run('split_spreads.py', '--year', 1882, '--workers', 2,
         '--dpi', 100, '--input-dir', raw, '--config', config,
         '--output-dir', output)
    folder = output / '1882'
    entries = _rows(folder / 'split_review.csv')
    assert [r['status'] for r in entries] == ['accepted', 'needs_review']
    assert (folder / 'accepted' / 'p0001_left_100dpi.png').exists()
    assert (folder / 'accepted' / 'p0001_right_100dpi.png').exists()
    assert (folder / 'needs_review' / 'p0002_source_100dpi.png').exists()
    assert (folder / 'needs_review' / 'p0002_cut.jpg').exists()
    assert (folder / 'needs_review' / 'review_contact_sheet.jpg').exists()
    assert not (folder / 'accepted' / 'p0002_left_100dpi.png').exists()
