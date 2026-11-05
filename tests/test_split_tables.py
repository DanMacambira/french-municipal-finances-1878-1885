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
