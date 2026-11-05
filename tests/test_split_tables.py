"""Tests for deterministic, independent two-table separator detection."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MOD = Path(__file__).resolve().parents[1] / "code" / "split_tables.py"
spec = importlib.util.spec_from_file_location("split_tables", MOD)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def synthetic_page(width=900, height=1250, separator=.48, shift=0):
    """Two 19th-century style tables, with fiscal columns then wide NOMS."""
    img = np.full((height, width), 252, dtype=np.uint8)
    seam = round(separator * width) + shift
    x_left, x_right = 12, width-15
    cv2.line(img,(x_left,90),(x_left,1160),0,2)
    cv2.line(img,(x_right,90),(x_right,1160),0,2)
    cv2.line(img,(x_left,90),(x_right,90),0,3)
    cv2.line(img,(x_left,210),(x_right,210),0,1)
    # Long central double-line (between narrow fiscal and wide name columns).
    cv2.line(img,(seam-2,90),(seam-2,1160),0,2)
    cv2.line(img,(seam+2,90),(seam+2,1160),0,2)
    # First table has tight fiscal columns immediately before the boundary.
    for x in [int(seam-0.17*width), int(seam-0.12*width),int(seam-0.08*width), int(seam-0.04*width)]:
        cv2.line(img,(x,93),(x,1160),0,2)
    # Second table: wide NOMS column immediately after the boundary, then
    # many narrow financial columns.
    for x in [int(seam+0.16*width), int(seam+0.22*width), int(seam+0.27*width),int(seam+0.31*width)]:
        if x < x_right-10:
            cv2.line(img,(x,93),(x,1160),0,2)
    for y in range(300,1150,35):
        cv2.line(img,(20,y),(130,y),140,1)
        cv2.line(img,(seam+15,y),(seam+85,y),140,1)
    return Image.fromarray(img)


def test_detect_distinct_separator():
    for fraction in (.455,.48,.521):
        result = module.detect_separator(synthetic_page(separator=fraction))
        assert result.accepted, result
        assert abs(result.fraction - fraction) < .015, result


def test_detect_rescaled_and_warped():
    image = np.asarray(synthetic_page(separator=.50))
    height, width = image.shape
    yy, xx = np.mgrid[0:height,0:width].astype(np.float32)
    # Mild non-rigid page displacement from book binding.
    map_x = xx + (7*np.sin(yy/height*2*np.pi)).astype(np.float32)
    warped = cv2.remap(image,map_x,yy,cv2.INTER_LINEAR,borderValue=255)
    result = module.detect_separator(Image.fromarray(warped))
    assert result.accepted, result
    assert abs(result.fraction - .50) < .015


def test_reject_blank_image():
    result = module.detect_separator(Image.new("L",(900,1250),"white"))
    assert not result.accepted


def test_crop_preserves_and_limits_overlap():
    image = Image.new("L",(900,1000),"white")
    left,right,x = module.crop_tables(image,.50,.004)
    assert x == 450
    assert left.width == 454
    assert right.width == 454
    assert left.width + right.width == 908


def test_does_not_accept_invalid_cut():
    image = Image.new("L",(900,1000),"white")
    try:
        module.crop_tables(image, .8)
    except ValueError:
        pass
    else:
        assert False, "Unsafe cut must raise"


def _warped_table(kind: str):
    base = np.asarray(synthetic_page(width=900, height=1250, separator=.50))
    h, w = base.shape
    yy, xx = np.indices(base.shape, dtype=np.float32)
    t = yy / (h - 1) * 2 - 1
    if kind == "linear":
        shift = 13 * t
    elif kind == "quadratic":
        shift = 20 * (t*t - .35)
    else:
        raise ValueError(kind)
    altered = cv2.remap(base, xx - shift.astype(np.float32), yy,
                        cv2.INTER_LINEAR, borderValue=255)
    return Image.fromarray(altered), shift


def test_follows_a_clear_slanted_separator_and_recovers_drift_review():
    image, shift = _warped_table("linear")
    separator = module.detect_separator(image)
    assert separator.reason == "curved_or_inconsistent_border_needs_review"
    path = module.detect_cut_path(image, separator)
    assert path.kind == "linear", path
    assert path.sample_count >= 10
    fractions = path.fractions(image.height)
    for y in [int(image.height * z) for z in [.12, .4, .7, .88]]:
        expected = 450 + float(shift[y, 0])
        assert abs(float(fractions[y]) * image.width - expected) < 7
    a, b, _ = module.crop_tables_along_path(image, path)
    assert a.height == b.height == image.height
    assert a.width + b.width >= image.width


def test_quadratic_is_used_only_when_evidence_supports_real_curvature():
    image, shift = _warped_table("quadratic")
    sep = module.detect_separator(image)
    path = module.detect_cut_path(image, sep)
    assert path.kind == "quadratic", path
    for y in [int(image.height*z) for z in [.12, .4, .7, .88]]:
        expected = 450 + float(shift[y, 0])
        assert abs(path.fractions(image.height)[y] * image.width - expected) < 7


def test_clean_straight_separator_falls_back_to_old_exact_crop():
    image = synthetic_page()
    result = module.detect_separator(image)
    path = module.detect_cut_path(image, result)
    assert path.kind == "vertical", path
    a, b, x = module.crop_tables_along_path(image, path)
    expected_a, expected_b, expected_x = module.crop_tables(image, result.fraction)
    assert x == expected_x
    assert np.array_equal(np.asarray(a), np.asarray(expected_a))
    assert np.array_equal(np.asarray(b), np.asarray(expected_b))


def test_explicit_vertical_mode_disables_all_fitting():
    image, _ = _warped_table("linear")
    result = module.detect_separator(image)
    path = module.detect_cut_path(image, result, mode="vertical")
    assert path.kind == "vertical"
    assert path.coefficients == (result.fraction,)


def test_ambiguous_header_cannot_be_rescued_by_polynomial():
    image = Image.new("L", (900, 1250), "white")
    result = module.detect_separator(image)
    path = module.detect_cut_path(image, result)
    assert not result.accepted
    assert path.kind == "vertical"


def test_tolerant_detector_recovers_fragmented_column_lines():
    base = np.asarray(synthetic_page(separator=.50)).copy()
    # Break all long vertical strokes into short 30-pixel segments. The strict
    # 3.5%-height opening cannot retain them; the tolerant 1.7% opening can.
    for y in range(90, 1160, 43):
        base[y+29:min(1161,y+43), :] = 252
    image = Image.fromarray(base)
    strict = module.detect_separator(image)
    assert not strict.accepted, strict
    tolerant = module.detect_separator_tolerant(image, expected=.50)
    assert tolerant is not None, (strict, tolerant)
    assert tolerant.accepted
    assert abs(tolerant.fraction - .50) < .012


def test_tolerant_detector_never_uses_prior_without_visual_rule():
    empty = Image.new('L', (900, 1250), 'white')
    assert module.detect_separator_tolerant(empty, expected=.54) is None
    assert module.detect_single_table_side(empty) is None


def test_single_table_side_requires_an_absent_parallel_half():
    original = np.asarray(synthetic_page(separator=.50)).copy()
    only_left = original.copy()
    only_left[:, 450:] = 255
    only_right = original.copy()
    only_right[:, :450] = 255
    assert module.detect_single_table_side(Image.fromarray(only_left)) == 'left'
    assert module.detect_single_table_side(Image.fromarray(only_right)) == 'right'
    assert module.detect_single_table_side(Image.fromarray(original)) is None


def test_layout_prior_is_by_year_page_orientation_and_requires_support(tmp_path):
    from dataclasses import replace
    s = module.Separator(.5,.2,1,0,0,.8,0,True,'ok')
    entries = [(100,'single',tmp_path/'a'),(102,'single',tmp_path/'b'),
               (101,'single',tmp_path/'c'),(103,'single',tmp_path/'d')]
    init = {entries[0][2]:replace(s,fraction=.535),
            entries[1][2]:replace(s,fraction=.542),
            entries[2][2]:replace(s,fraction=.467),
            entries[3][2]:replace(s,fraction=.463)}
    priors = module.layout_priors(entries,init)
    assert .53 < priors['single_0'] < .55
    assert .46 < priors['single_1'] < .48
    assert module.layout_priors(entries[:2],init).get('single_1') is None


def test_concatenated_review_collects_all_years(tmp_path):
    import csv
    for year, page in [(1880,272),(1878,450),(1880,271)]:
        out = tmp_path / str(year)
        out.mkdir(exist_ok=True)
        file = out/'table_split_review.csv'
        rows = []
        if file.exists():
            with file.open(newline='') as h:
                rows = list(csv.DictReader(h))
        rows.append(dict(year=str(year),pdf_page=str(page),printed_side='single',status='accepted'))
        with file.open('w',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=['year','pdf_page','printed_side','status'])
            writer.writeheader();writer.writerows(rows)
    merged_path=module.aggregate_review_csvs(tmp_path)
    assert merged_path.name == 'all_years_table_split_review.csv'
    with merged_path.open(newline='') as h:
        merged=list(csv.DictReader(h))
    assert [(int(r['year']),int(r['pdf_page'])) for r in merged]==[(1878,450),(1880,271),(1880,272)]


def _extremely_distorted_page(kind, *, fragmented=False):
    """Strong, fully observed separator, moved locally by page geometry."""
    base = np.asarray(synthetic_page(separator=.475)).copy()
    h, w = base.shape
    yy, xx = np.indices(base.shape, dtype=np.float32)
    y = 2 * yy / (h-1) - 1
    if kind == 'vertical':
        shift = np.zeros_like(y)
    elif kind == 'linear':
        shift = 43*y
    elif kind == 'quadratic':
        shift = 58*(y*y-.30)
    else:
        raise ValueError(kind)
    warped = cv2.remap(base, xx-shift.astype(np.float32), yy,
                       cv2.INTER_LINEAR, borderValue=255)
    if fragmented:
        for top in range(100, 1150, 52):
            warped[top+39:min(top+52, h), :] = 255
    return Image.fromarray(warped), shift


def test_extreme_detector_fits_vertical_linear_and_quadratic():
    for kind in ('vertical','linear','quadratic'):
        image, shift = _extremely_distorted_page(kind)
        result = module.detect_separator_extreme(image, expected=.475)
        assert result is not None, kind
        separator, path = result
        assert separator.reason == 'recovered_extreme_rule_trace'
        assert path.kind == kind, (kind, path)
        assert path.sample_count >= 10
        positions = path.fractions(image.height)
        for y in (int(image.height*o) for o in (.15, .40, .65, .85)):
            expected_x = .475*image.width + float(shift[y, 0])
            assert abs(positions[y]*image.width-expected_x) < 9, (kind,y,path)


def test_extreme_detector_handles_fragmented_slanted_and_curved_rules():
    for kind in ('linear','quadratic'):
        image, _ = _extremely_distorted_page(kind, fragmented=True)
        assert not module.detect_separator(image).accepted
        result = module.detect_separator_extreme(image, expected=.475)
        assert result is not None, kind
        assert result[1].kind == kind


def test_extreme_prior_alone_cannot_generate_separator():
    for image in (Image.new('L',(900,1250),255),
                  Image.new('L',(900,1250),170)):
        assert module.detect_separator_extreme(image, expected=.475) is None
    # A lone line without the expected fiscal-column-to-NOMS signature
    # is insufficient, however close it is to the learned position.
    isolated = np.full((1250,900),255,dtype=np.uint8)
    cv2.line(isolated,(428,100),(428,1160),0,2)
    assert module.detect_separator_extreme(Image.fromarray(isolated),.475) is None


def test_extreme_is_review_only_and_does_not_change_accepted_peers(tmp_path):
    import csv
    original = synthetic_page(separator=.475)
    warped, _ = _extremely_distorted_page('linear',fragmented=True)
    folder = tmp_path/'split_pages'/'1883'
    folder.mkdir(parents=True)
    for page in (491,493):
        original.save(folder / f'p{page:04d}_single_100dpi.png')
    warped.save(folder/'p0495_single_100dpi.png')
    output = tmp_path/'table_panels'
    rows = module.split_year(year=1883, pages=None, dpi=100,
                             input_root=tmp_path/'split_pages', output_root=output)
    by_page = {int(r['pdf_page']):r for r in rows}
    assert by_page[495]['status'] == 'accepted_extreme_pending_review', by_page[495]
    assert by_page[495]['detection_pass'] == 'extreme'
    assert by_page[495]['cut_method'] == 'linear'
    assert 'needs_review' in by_page[495]['table_1']
    assert Path(by_page[495]['table_1']).exists()
    assert Path(by_page[495]['table_2']).exists()
    assert not (output/'1883'/'p0495_single_col1_100dpi.png').exists()
    for page in (491,493):
        assert by_page[page]['status']=='accepted', by_page[page]
        assert by_page[page]['detection_pass']=='strict'
        assert 'accepted' in by_page[page]['table_1']
    with (output/'all_years_table_split_review.csv').open(newline='') as f:
        combined=list(csv.DictReader(f))
    assert len(combined)==3
    assert combined[-1]['status']=='accepted_extreme_pending_review'
    # Disabling extreme restores v1.3 decision and removes any ordinary
    # stale crops; the unverified candidates are not promoted automatically.
    old = module.split_year(year=1883,pages=[495],dpi=100,
                            input_root=tmp_path/'split_pages',output_root=output,
                            extreme=False)
    assert next(r for r in old if int(r['pdf_page'])==495)['status']=='needs_review'
    assert not Path(by_page[495]['table_1']).exists()
    assert not Path(by_page[495]['table_2']).exists()


def _short_printed_page(kind: str = 'linear', fraction: float = .535):
    """Two normal columns ending around halfway down an otherwise blank page."""
    base = np.asarray(synthetic_page(separator=fraction)).copy()
    h, w = base.shape
    yy, xx = np.indices(base.shape, dtype=np.float32)
    t = 2*yy/(h-1)-1
    if kind == 'linear':
        shift = 13*t
    elif kind == 'quadratic':
        shift = 20*(t*t-.35)
    elif kind == 'vertical':
        shift = np.zeros_like(t)
    else:
        raise ValueError(kind)
    warped = cv2.remap(base, xx-shift.astype(np.float32), yy,
                       cv2.INTER_LINEAR, borderValue=255)
    warped[round(.48*h):, :] = 255
    return Image.fromarray(warped), shift


def test_short_table_fourth_pass_traces_only_populated_area():
    for kind in ('vertical', 'linear', 'quadratic'):
        image, shift = _short_printed_page(kind)
        original = module.detect_separator(image)
        old_path = module.detect_cut_path(image, original)
        assert not original.accepted
        assert old_path.reason == 'too_few_consistent_rule_samples'
        assert module.detect_separator_extreme(image, .535) is None
        recovered = module.detect_separator_short_table(image, .535, original)
        assert recovered is not None, kind
        separator, path = recovered
        assert separator.reason == 'recovered_short_table_active_region'
        assert path.kind in {'vertical', 'linear', 'quadratic'}
        assert path.sample_count >= 10
        assert path.active_y_range is not None
        assert .45 <= (path.active_y_range[1]+1)/2 <= .55
        positions = path.fractions(image.height)
        for y_frac in (.12, .30, .43):
            y = int(image.height*y_frac)
            expected = .535*image.width + float(shift[y,0])
            assert abs(positions[y]*image.width-expected) < 7, (kind,y,path)
        # The path must NOT extrapolate into unprinted white paper.
        assert abs(positions[int(.7*image.height)] - positions[-1]) < 1e-12
        assert abs(positions[-1] - positions[int(.99*image.height)]) < 1e-12


def test_short_table_fallback_rejects_blank_and_full_height():
    blank = Image.new('L', (900, 1250), 255)
    assert module.detect_separator_short_table(blank, .535,
                                               module.detect_separator(blank)) is None
    full = synthetic_page(separator=.535)
    assert module._short_table_active_region(full, .535) is None
    short, _ = _short_printed_page('linear')
    only_one_side = np.asarray(short).copy()
    only_one_side[:, 485:] = 255
    assert module._short_table_active_region(Image.fromarray(only_one_side), .535) is None


def test_short_table_candidate_is_review_only_and_keeps_good_peer_crops(tmp_path):
    import csv
    import hashlib
    folder = tmp_path/'split_pages'/'1881'
    folder.mkdir(parents=True)
    normal = synthetic_page(separator=.535)
    short, _ = _short_printed_page('linear')
    for page in (462, 464):
        normal.save(folder/f'p{page:04d}_single_100dpi.png')
    short.save(folder/'p0466_single_100dpi.png')
    output = tmp_path/'table_panels'
    rows = module.split_year(year=1881, pages=None, dpi=100,
                             input_root=tmp_path/'split_pages', output_root=output)
    by_page = {int(r['pdf_page']):r for r in rows}
    for page in (462, 464):
        assert by_page[page]['status'] == 'accepted'
        assert by_page[page]['detection_pass'] == 'strict'
    short_row = by_page[466]
    assert short_row['status'] == 'accepted_short_table_pending_review', short_row
    assert short_row['detection_pass'] == 'short_table'
    assert float(short_row['active_table_bottom_fraction']) < .55
    assert short_row['table_count'] == 2
    assert 'needs_review' in short_row['table_1']
    assert Path(short_row['table_1']).exists() and Path(short_row['table_2']).exists()
    assert not (output/'1881'/'p0466_single_col1_100dpi.png').exists()
    earlier = {p:hashlib.sha256(Path(by_page[p]['table_1']).read_bytes()).hexdigest()
               for p in (462,464)}
    merged = output/'all_years_table_split_review.csv'
    with merged.open(newline='') as f:
        combined = list(csv.DictReader(f))
    assert len(combined) == 3 and combined[-1]['detection_pass'] == 'short_table'
    again = module.split_year(year=1881, pages=[466], dpi=100,
                              input_root=tmp_path/'split_pages', output_root=output,
                              short_table=False)
    assert next(r for r in again if int(r['pdf_page']) == 466)['status'] == 'needs_review'
    assert not Path(short_row['table_1']).exists()
    for page in (462,464):
        assert hashlib.sha256(Path(by_page[page]['table_1']).read_bytes()).hexdigest() == earlier[page]
    with merged.open(newline='') as f:
        combined = list(csv.DictReader(f))
    assert len(combined) == 3
    assert next(r for r in combined if int(r['pdf_page'])==466)['status'] == 'needs_review'
