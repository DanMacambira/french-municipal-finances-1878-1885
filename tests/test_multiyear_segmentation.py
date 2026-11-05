"""Configuration and single-page pipeline regression tests, without paid OCR."""
import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / 'code' / (name + '.py')
    spec = importlib.util.spec_from_file_location('test_' + name + '_multi', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SPREAD = load('split_spreads')
TABLES = load('split_tables')


def make_page_image():
    """Two table headers sharing a long printed border."""
    w, h = 900, 1250
    ar = np.full((h, w), 250, dtype=np.uint8)
    x = 435
    for xx in [12, x-2, x+2, 885]:
        cv2.line(ar, (xx, 90), (xx, 1160), 5, 2)
    cv2.line(ar, (12, 90), (885, 90), 5, 3)
    cv2.line(ar, (12, 210), (885, 210), 5, 1)
    for xx in [x-150, x-110, x-70, x-35, x+150, x+205, x+250, x+300]:
        cv2.line(ar, (xx, 92), (xx, 1160), 5, 2)
    return Image.fromarray(ar)


def pdf_with_image(path, image, width=None, height=None):
    data = io.BytesIO()
    image.save(data, format='PNG')
    with pymupdf.open() as doc:
        pg = doc.new_page(width=width or image.width, height=height or image.height)
        pg.insert_image(pg.rect, stream=data.getvalue())
        doc.save(path)


def run_first(tmp_path, year, image, scan_layout, *, sides=None):
    inp = tmp_path / 'raw'
    inp.mkdir()
    pdf_with_image(inp / f'{year}.pdf', image)
    config = tmp_path / 'ranges.json'
    entry = {'confirmed': True, 'pdf': f'{year}.pdf', 'start': 1, 'end': 1,
             'scan_layout': scan_layout}
    if sides is not None:
        entry['page_sides'] = {'1': sides}
    config.write_text(json.dumps({str(year): entry}))
    out = tmp_path / 'split_pages'
    rows = SPREAD.process_year(year=year, requested_pages=None, dpi=100,
                               config=config, input_dir=inp, output_dir=out,
                               center_min=.45, center_max=.55,
                               overlap_fraction=0., make_pdfs=False,
                               force_review=False, inner_trim_fraction=.001)
    return rows, out, config


def test_year_layouts_are_explicit_and_exceptions_preserved():
    mapping = json.loads((ROOT / 'schema/situation_financiere/page_ranges.json').read_text())
    assert {y for y, d in mapping.items() if d['scan_layout'] == 'spread'} == {
        '1880', '1882', '1885'}
    assert {y for y, d in mapping.items() if d['scan_layout'] == 'single'} == {
        '1878', '1879', '1881', '1883', '1884'}
    assert mapping['1882']['page_sides']['244'] == ['right']
    assert mapping['1885']['page_sides'] == {'262': ['right'], '272': ['left']}
    assert mapping['1882']['table_panel_split_overrides']['p245_left'] == .521


def test_landscape_pdf_explicit_single_is_not_bisected(tmp_path):
    # Deliberately use a landscape PDF: this would be misclassified under auto.
    im = Image.new('L', (1200, 780), 240)
    rows, out, _ = run_first(tmp_path, 1878, im, 'single')
    assert rows[0]['status'] == 'accepted_single'
    folder = out / '1878'
    assert (folder / 'accepted' / 'p0001_single_100dpi.png').exists()
    assert not (folder / 'accepted' / 'p0001_left_100dpi.png').exists()
    assert not (folder / 'accepted' / 'p0001_right_100dpi.png').exists()
    assert (folder / 'review_contact_sheet.jpg').exists()
    with (folder / 'split_review.csv').open(newline='') as f:
        items = list(csv.DictReader(f))
    assert items[0]['scan_layout'] == 'single'
    assert items[0]['status'] == 'accepted_single'


def test_single_page_flows_into_second_table_split(tmp_path):
    rows, out, _ = run_first(tmp_path, 1878, make_page_image(), 'single')
    assert rows[0]['status'] == 'accepted_single'
    output = tmp_path / 'table_panels'
    result = TABLES.split_year(year=1878, pages=None, dpi=100,
                               input_root=out, output_root=output)
    assert len(result) == 1
    assert result[0]['printed_side'] == 'single'
    assert result[0]['status'] == 'accepted', result[0]
    folder = output / '1878'
    assert (folder / 'accepted' / 'p0001_single_col1_100dpi.png').exists()
    assert (folder / 'accepted' / 'p0001_single_col2_100dpi.png').exists()
    assert (folder / 'review_contact_sheet.jpg').exists()


def test_spread_sides_not_overridden_and_no_single_pass_through(tmp_path):
    width, height = 1200, 800
    image = Image.new('L', (width, height), 250)
    ar = np.array(image)
    ar[:, 593:608] = 0
    ar[40:height-30, 585:616] = 45
    rows, out, _ = run_first(tmp_path, 1882, Image.fromarray(ar), 'spread', sides=['right'])
    assert rows[0]['scan_layout'] == 'spread'
    assert (out / '1882' / 'accepted' / 'p0001_right_100dpi.png').exists()
    assert not (out / '1882' / 'accepted' / 'p0001_left_100dpi.png').exists()
    assert not (out / '1882' / 'accepted' / 'p0001_single_100dpi.png').exists()


def test_tolerant_stage2_and_one_table_output_and_aggregate(tmp_path):
    """Accepted peers supply a location prior; image evidence decides outcomes."""
    folder = tmp_path / 'split_pages' / '1884'
    folder.mkdir(parents=True)
    base = np.asarray(make_page_image()).copy()
    for page in [491, 493]:
        Image.fromarray(base).save(folder / f'p{page:04d}_single_100dpi.png')
    broken = base.copy()
    for start in range(90, 1160, 43):
        broken[start+29:min(1161,start+43),:] = 250
    Image.fromarray(broken).save(folder/'p0495_single_100dpi.png')
    one = base.copy()
    one[:,450:] = 255
    Image.fromarray(one).save(folder/'p0497_single_100dpi.png')
    output = tmp_path / 'table_panels'
    rows = TABLES.split_year(year=1884, pages=None, dpi=100,
                             input_root=tmp_path/'split_pages', output_root=output)
    by_page = {int(row['pdf_page']):row for row in rows}
    assert by_page[495]['status'] == 'accepted', by_page[495]
    assert by_page[495]['detection_pass'] == 'tolerant'
    assert by_page[497]['status'] == 'accepted_single_table', by_page[497]
    assert by_page[497]['table_1']
    assert not by_page[497]['table_2']
    assert not (output/'1884'/'p0497_single_col2_100dpi.png').exists()
    joined = output/'all_years_table_split_review.csv'
    assert joined.exists()
    with joined.open(newline='') as f:
        r = list(csv.DictReader(f))
    assert len(r)==4
    assert r[-1]['status']=='accepted_single_table'


def test_partial_rerun_keeps_other_report_rows_and_uses_full_year_priors(tmp_path):
    folder = tmp_path/'split_pages'/'1884'
    folder.mkdir(parents=True)
    base = np.asarray(make_page_image()).copy()
    for page in [491,493]:
        Image.fromarray(base).save(folder/f'p{page:04d}_single_100dpi.png')
    fragmented=base.copy()
    for y in range(90,1160,43):
        fragmented[y+29:min(1161,y+43),:] = 250
    Image.fromarray(fragmented).save(folder/'p0495_single_100dpi.png')
    out = tmp_path/'table_panels'
    full=TABLES.split_year(year=1884,pages=None,dpi=100,
                           input_root=tmp_path/'split_pages',output_root=out)
    assert len(full)==3
    one=TABLES.split_year(year=1884,pages=[495],dpi=100,
                          input_root=tmp_path/'split_pages',output_root=out)
    assert len(one)==3  # prior review rows retained, not overwritten
    assert next(r for r in one if int(r['pdf_page'])==495)['detection_pass']=='tolerant'
    with (out/'all_years_table_split_review.csv').open(newline='') as f:
        assert len(list(csv.DictReader(f)))==3
