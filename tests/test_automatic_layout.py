from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from automatic_layout import brace_proposals, printed_page_regions, table_split, vertical_rules, diagnose


def test_single_page_classification():
    im = Image.new("L", (680, 1000), 255)
    assert printed_page_regions(im, [])["type"] == "single_printed_page"


def test_spread_diagnostic_is_not_claimed_as_verified():
    im = Image.new("L", (1600, 1000), 255)
    result = printed_page_regions(im, [])
    assert result["type"] == "possible_spread"
    assert result["status"] in {"candidate", "needs_review"}


def test_detect_table_rule_without_fixed_midpoint():
    im = Image.new("L", (700, 1000), 255)
    d = ImageDraw.Draw(im)
    for x in (12, 105, 190, 290, 345, 435, 510, 620, 690):
        d.line([(x, 90), (x, 930)], fill=0, width=3)
    rules = vertical_rules(im)
    result = table_split(im, rules)
    assert result["status"] == "candidate"
    assert abs(result["x"] - 345) < 5


def test_no_invented_split_when_no_rule():
    im = Image.new("L", (700, 1000), 255)
    assert table_split(im, vertical_rules(im))["status"] == "unresolved"


def test_detect_brace_like_components():
    im = Image.new("L", (340, 1000), 255)
    d = ImageDraw.Draw(im)
    for y in (300, 420, 540, 660):
        # Slight curvature, 5x45px; repeated x-location.
        d.arc((242, y, 251, y + 44), 95, 265, fill=0, width=2)
    result = brace_proposals(im)
    assert any(238 <= c["x_min"] <= 255 for c in result)
    assert all("y_min" in c and "y_max" in c for c in result)


def test_layout_contains_no_commune_assignments():
    im = Image.new("L", (700, 1000), 255)
    result = diagnose(im)
    assert result["status"] == "diagnostic_only_not_validated"
    assert "commune" not in str(result)


def test_continuous_straight_bar_never_marked_curly_brace():
    im = Image.new('L', (340, 1000), 255)
    dr = ImageDraw.Draw(im)
    dr.line([(245, 200), (245, 870)], fill=0, width=2)
    proposed = brace_proposals(im)
    assert not any(p['kind'] == 'curly_brace_candidate' for p in proposed)
