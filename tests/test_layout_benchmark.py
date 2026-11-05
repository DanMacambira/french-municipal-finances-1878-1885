"""Offline sampler, manifest, spread splitting, review and score smoke tests."""
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from layout_benchmark import (
    _image_regions, diagnose_sample, frozen_plan, main, score_reviews, select_pages,
)


def config(years=(1878, 1880, 1881, 1882)):
    return {str(y): {"pdf": f"La_Situation_financiere_des_communes_{y}.pdf",
                     "start": 10, "end": 25, "confirmed": True}
            for y in years}


def test_balanced_deterministic_sample():
    c = config(years=range(1878, 1886))
    a = select_pages(c, list(range(1878, 1886)), per_year=5, seed=1882)
    b = select_pages(c, list(range(1878, 1886)), per_year=5, seed=1882)
    assert a == b
    assert len(a) == 40
    assert {(r["year"], r["pdf_page"]) for r in a} == {
        (r["year"], r["pdf_page"]) for r in b
    }
    for year in range(1878, 1886):
        sampled = [r for r in a if r["year"] == year]
        assert len(sampled) == 5
        assert {10, 17, 25}.issubset({r["pdf_page"] for r in sampled})
        assert sum(r["role"] == "holdout" for r in sampled) == 2


def test_page_side_exclusion_kept_in_manifest():
    c = config(years=(1882,))
    c["1882"]["page_sides"] = {"10": ["right"]}
    a = select_pages(c, [1882], per_year=3, seed=2)
    assert next(x for x in a if x["pdf_page"] == 10)["allowed_sides"] == ["right"]


def test_unconfirmed_ranges_refused():
    c = config(years=(1881,))
    c["1881"]["confirmed"] = False
    with pytest.raises(ValueError, match="confirmed"):
        select_pages(c, [1881], per_year=3, seed=5)


def test_frozen_plan_reproducible_and_guarded(tmp_path):
    cfg = tmp_path / "page_ranges.json"
    cfg.write_text(json.dumps(config(years=(1878,))), encoding="utf-8")
    out = tmp_path / "output"
    p1 = frozen_plan(cfg, out, years=[1878], per_year=5, seed=25)
    p2 = frozen_plan(cfg, out, years=[1878], per_year=5, seed=25)
    assert p1 == p2
    with pytest.raises(ValueError, match="Frozen"):
        frozen_plan(cfg, out, years=[1878], per_year=5, seed=26)
    assert frozen_plan(cfg, out, years=[1878], per_year=5, seed=26, reset=True)["seed"] == 26


def _draw_page(size=(700, 1000)):
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    for x in [15, 122, 242, 343, 444, 562, 687]:
        d.line([(x, 100), (x, 930)], fill="black", width=3)
    for y in [300, 400, 500, 600, 700]:
        d.line([(120, y), (320, y)], fill="black", width=1)
    return im


def test_offline_diagnose_creates_manifest_images(tmp_path):
    im = _draw_page()
    sample = {"year": 1881, "pdf_page": 10, "source_pdf": "scan.pdf",
              "role": "holdout", "allowed_sides": ["left", "right"]}
    rows, review = diagnose_sample(im, sample, tmp_path)
    assert review["year"] == 1881
    assert len(rows) == 1
    assert (tmp_path / rows[0]["overlay"]).is_file()
    assert (tmp_path / rows[0]["diagnostic_json"]).is_file()
    assert (tmp_path / "overlays" / "1881" / "1881_p0010_page.jpg").is_file()
    assert rows[0]["table_split_status"] in ("candidate", "unresolved", "not_applicable")


def test_wide_page_is_split_to_two_printed_pages(tmp_path):
    left = _draw_page()
    both = Image.new("RGB", (1400, 1000), "white")
    both.paste(left, (0, 0))
    both.paste(left, (700, 0))
    structure, regions = _image_regions(both)
    assert structure["type"] == "possible_spread"
    assert {name for name, _ in regions} == {"left", "right"}
    assert all(r.width > 600 for _, r in regions)


def test_review_scores_count_only_completed_items(tmp_path):
    path = tmp_path / "human_review.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "pdf_page", "role", "page_structure_ok",
                                                  "gutter_ok", "table_splits_ok", "all_columns_visible",
                                                  "brace_proposals_acceptable"])
        writer.writeheader()
        writer.writerow({"year": 1882, "pdf_page": 245, "role": "holdout", "page_structure_ok": "yes",
                         "gutter_ok": "no", "table_splits_ok": "", "all_columns_visible": "na"})
    result = score_reviews(tmp_path)
    assert result["evaluated_checks"] == 2
    with (tmp_path / "human_validation_summary.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert next(x for x in rows if x["sample"] == "holdout" and x["metric"] == "gutter_ok")["accuracy"] == "0.0"
    assert next(x for x in rows if x["sample"] == "holdout" and x["metric"] == "table_splits_ok")["accuracy"] == ""


def test_cli_offline_pdf_end_to_end_and_preserves_review(tmp_path):
    fitz = pytest.importorskip("fitz")
    repo = tmp_path / "repo"
    cfg_dir = repo / "schema" / "situation_financiere"
    cfg_dir.mkdir(parents=True)
    pdf_dir = repo / "raw_data" / "situation_financiere_communes"
    pdf_dir.mkdir(parents=True)
    filename = "La_Situation_financiere_des_communes_1881.pdf"
    cfg_dir.joinpath("page_ranges.json").write_text(json.dumps({
        "1881": {"pdf": filename, "start": 1, "end": 2, "confirmed": True}
    }), encoding="utf-8")
    with fitz.open() as pdf:
        for _ in range(2):
            p = pdf.new_page(width=700, height=1000)
            p.draw_line((340, 90), (340, 900), color=(0, 0, 0), width=3)
            for x in [50, 145, 245, 445, 540, 640]:
                p.draw_line((x, 90), (x, 900), color=(0, 0, 0), width=2)
        pdf.save(pdf_dir / filename)
    opts = ["--repo-root", str(repo), "--years", "1881", "--per-year", "2", "--dpi", "90"]
    assert main(opts + ["--plan-only"]) == 0
    assert main(opts) == 0
    report_dir = repo / "derived" / "situation_financiere" / "layout_benchmark"
    assert (report_dir / "diagnostic_manifest.csv").is_file()
    assert (report_dir / "human_review.csv").is_file()
    assert (report_dir / "diagnostic_summary.json").is_file()
    review_path = report_dir / "human_review.csv"
    text = review_path.read_text(encoding="utf-8").replace(",,,,,,", ",,,,,,")
    review_path.write_text(text + "\n", encoding="utf-8")
    assert main(opts) == 0
    assert review_path.read_text(encoding="utf-8").endswith("\n\n")
    assert main(opts + ["--score-only"]) == 0
