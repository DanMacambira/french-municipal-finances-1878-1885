#!/usr/bin/env python3
"""Reproducible, offline multi-year benchmark for automatic table layout.

Usage (from the repository root):
  python code/layout_benchmark.py --plan-only
  python code/layout_benchmark.py
  python code/layout_benchmark.py --score-only

No OCR/API/network calls, page-specific coordinates, or changes to OCR caches.
Image overlays are proposals, NOT evidence of correct layout without review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from automatic_layout import diagnose, overlay, printed_page_regions, vertical_rules

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = Path("derived/situation_financiere/layout_benchmark")
REVIEW_METRICS = (
    "page_structure_ok",
    "gutter_ok",
    "table_splits_ok",
    "all_columns_visible",
    "brace_proposals_acceptable",
)
REVIEW_FIELDS = (
    "year", "pdf_page", "role", "page_structure", "reviewer",
    *REVIEW_METRICS, "notes",
)
MANIFEST_FIELDS = (
    "year", "pdf_page", "role", "source_pdf", "printed_side", "allowed_side",
    "page_structure", "gutter_status", "gutter_relative_x", "table_split_status",
    "table_split_relative_x", "vertical_rule_count", "brace_candidate_count",
    "curly_brace_candidate_count", "overlay", "diagnostic_json", "warning",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def _digest(seed: int, year: int, page: int, tag: str) -> str:
    return hashlib.sha256(f"{seed}:{year}:{page}:{tag}".encode()).hexdigest()


def select_pages(config: dict, years: list[int], *, per_year: int, seed: int) -> list[dict]:
    """Stratified across years, endpoints/midpoints plus deterministic random pages.

    Freeze the resulting list in sample_plan.json before detector evaluation.
    Holdout labels are assigned independently of layout detection.
    """
    if per_year < 2:
        raise ValueError("--per-year must be >= 2 for a development/holdout split")
    samples: list[dict] = []
    for year in years:
        spec = config.get(str(year))
        if not isinstance(spec, dict):
            raise ValueError(f"No page-range configuration for {year}")
        if spec.get("confirmed") is not True:
            raise ValueError(f"Page range for {year} must be confirmed=true")
        start, end = spec.get("start"), spec.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
            raise ValueError(f"Invalid 1-indexed PDF page range for {year}: {start}..{end}")
        pdf_name = spec.get("pdf")
        if not isinstance(pdf_name, str) or not pdf_name.lower().endswith(".pdf") or Path(pdf_name).name != pdf_name:
            raise ValueError(f"Invalid PDF filename for {year}: {pdf_name!r}")
        all_pages = list(range(start, end + 1))
        if len(all_pages) < 2:
            raise ValueError(f"Need at least two pages for {year} to form a holdout")
        count = min(per_year, len(all_pages))
        anchors = list(dict.fromkeys((start, (start + end) // 2, end)))[:count]
        extra = sorted((p for p in all_pages if p not in anchors),
                       key=lambda p: _digest(seed, year, p, "sample"))
        selected = sorted((anchors + extra[:count - len(anchors)]))
        holdout_n = max(1, min(count - 1, round(count * .4)))
        holdout = set(sorted(selected, key=lambda p: _digest(seed, year, p, "holdout"))[:holdout_n])
        for page in selected:
            sides = spec.get("page_sides", {}).get(str(page))
            if sides is None:
                sides = ["left", "right"]
            if not isinstance(sides, list) or not sides or any(s not in ("left", "right") for s in sides):
                raise ValueError(f"Invalid page_sides for {year} PDF page {page}: {sides!r}")
            samples.append({
                "year": year, "pdf_page": page, "source_pdf": pdf_name,
                "role": "holdout" if page in holdout else "development",
                "allowed_sides": list(dict.fromkeys(sides)),
            })
    return samples


def frozen_plan(config_file: Path, output: Path, *, years: list[int],
                per_year: int, seed: int, reset: bool = False) -> dict:
    config_bytes = config_file.read_bytes()
    sha = hashlib.sha256(config_bytes).hexdigest()
    path = output / "sample_plan.json"
    if path.exists() and not reset:
        plan = _json(path)
        if (plan.get("source_config_sha256") != sha or plan.get("years") != years
                or plan.get("per_year") != per_year or plan.get("seed") != seed):
            raise ValueError("Frozen sample_plan.json does not match requested settings or config. "
                             "Keep the existing plan for a valid benchmark, or use --reset-plan deliberately.")
        return plan
    plan = {
        "version": 1,
        "source_config": str(config_file),
        "source_config_sha256": sha,
        "years": years,
        "per_year": per_year,
        "seed": seed,
        "samples": select_pages(json.loads(config_bytes), years, per_year=per_year, seed=seed),
        "note": "Selection is frozen before detector evaluation. Holdout pages must not be used to tune thresholds.",
    }
    _write_json(path, plan)
    return plan


def _render(pdf, page: int, dpi: int) -> Image.Image:
    import fitz
    if not (1 <= page <= len(pdf)):
        raise ValueError(f"PDF page {page} outside 1..{len(pdf)}")
    pix = pdf[page - 1].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _image_regions(image: Image.Image) -> tuple[dict, list[tuple[str, Image.Image]]]:
    structure = printed_page_regions(image, vertical_rules(image))
    if structure["type"] == "single_printed_page":
        return structure, [("full", image)]
    x = structure.get("split_x")
    if not isinstance(x, int) or x <= 0 or x >= image.width:
        return structure, [("full_unresolved", image)]
    # Even a low-confidence gutter is visualized, but NOT declared correct.
    # Overlap protects characters touching the scan's binding edge.
    padding = max(3, round(image.width * .004))
    return structure, [
        ("left", image.crop((0, 0, min(image.width, x + padding), image.height))),
        ("right", image.crop((max(0, x - padding), 0, image.width, image.height))),
    ]


def _page_overlay(image: Image.Image, structure: dict) -> Image.Image:
    out = image.copy().convert("RGB")
    x = structure.get("split_x")
    if x is not None:
        d = ImageDraw.Draw(out)
        d.line([(x, 0), (x, out.height)], fill=(10, 190, 30), width=max(3, out.width // 350))
    return out


def diagnose_sample(image: Image.Image, sample: dict, output: Path) -> tuple[list[dict], dict]:
    """Diagnose the spread and then EACH printed page independently."""
    year, page = sample["year"], sample["pdf_page"]
    stem = f"{year}_p{page:04d}"
    structure, regions = _image_regions(image)
    images_dir = output / "overlays" / str(year)
    data_dir = output / "diagnostics" / str(year)
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    _page_overlay(image, structure).save(images_dir / f"{stem}_page.jpg", quality=89)
    _write_json(data_dir / f"{stem}_structure.json", structure)
    records = []
    for side, tile in regions:
        result = diagnose(tile)
        overlay_path = images_dir / f"{stem}_{side}_overlay.jpg"
        json_path = data_dir / f"{stem}_{side}.json"
        overlay(tile, result).save(overlay_path, quality=90)
        _write_json(json_path, result)
        splits = result["table_splits"]
        split = splits[0] if splits else {"status": "not_evaluated", "relative_x": None}
        expected = side == "full" or side in sample["allowed_sides"]
        warnings = []
        if structure.get("status") == "needs_review":
            warnings.append("gutter_uncertain")
        if split.get("status") != "candidate":
            warnings.append("table_split_unresolved")
        if not expected:
            warnings.append("excluded_side_in_source_config")
        records.append({
            "year": year, "pdf_page": page, "role": sample["role"],
            "source_pdf": sample["source_pdf"], "printed_side": side,
            "allowed_side": expected, "page_structure": structure["type"],
            "gutter_status": structure.get("status", "not_applicable"),
            "gutter_relative_x": structure.get("relative_x", ""),
            "table_split_status": split.get("status", "not_evaluated"),
            "table_split_relative_x": split.get("relative_x", ""),
            "vertical_rule_count": len(result["vertical_rules"]),
            "brace_candidate_count": len(result["brace_proposals"]),
            "curly_brace_candidate_count": sum(b["kind"] == "curly_brace_candidate"
                                               for b in result["brace_proposals"]),
            "overlay": str(overlay_path.relative_to(output)),
            "diagnostic_json": str(json_path.relative_to(output)),
            "warning": ";".join(warnings),
        })
    review = {"year": year, "pdf_page": page, "role": sample["role"],
              "page_structure": structure["type"]}
    return records, review


def _review_csv(output: Path, review_rows: list[dict]) -> None:
    path = output / "human_review.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            old_rows = list(csv.DictReader(f))
        old_ids = {(int(r["year"]), int(r["pdf_page"]), r["role"]) for r in old_rows}
        new_ids = {(r["year"], r["pdf_page"], r["role"]) for r in review_rows}
        if old_ids != new_ids:
            raise ValueError("Existing human_review.csv refers to a different sample. "
                             "Move it aside before changing the frozen plan.")
        return  # Never destroy human corrections on reruns.
    _write_csv(path, REVIEW_FIELDS, review_rows)


def run_benchmark(repo: Path, output: Path, plan: dict, dpi: int) -> dict:
    import fitz
    if not 72 <= dpi <= 400:
        raise ValueError("dpi must be between 72 and 400")
    pdf_dir = repo / "raw_data" / "situation_financiere_communes"
    samples = plan["samples"]
    by_year: dict[int, list[dict]] = defaultdict(list)
    for s in samples:
        by_year[int(s["year"])].append(s)
    # Preflight missing inputs before creating a misleading partial benchmark.
    missing = [str(pdf_dir / s["source_pdf"]) for s in samples
               if not (pdf_dir / s["source_pdf"]).is_file()]
    if missing:
        raise FileNotFoundError("Missing source PDF(s):\n" + "\n".join(sorted(set(missing))))
    records, reviews = [], []
    for year in sorted(by_year):
        src = pdf_dir / by_year[year][0]["source_pdf"]
        with fitz.open(src) as pdf:
            for s in by_year[year]:
                print(f"{year} PDF page {s['pdf_page']} ({s['role']})", flush=True)
                im = _render(pdf, s["pdf_page"], dpi)
                rows, review = diagnose_sample(im, s, output)
                records.extend(rows)
                reviews.append(review)
    _write_csv(output / "diagnostic_manifest.csv", MANIFEST_FIELDS, records)
    _review_csv(output, reviews)
    statuses = Counter(r["table_split_status"] for r in records if r["allowed_side"])
    summary = {
        "pdf_pages": len(samples), "years": len(by_year),
        "development_pages": sum(s["role"] == "development" for s in samples),
        "holdout_pages": sum(s["role"] == "holdout" for s in samples),
        "printed_page_regions": len(records),
        "table_split_proposal_statuses": dict(statuses),
        "page_gutter_needs_review": len({(r["year"], r["pdf_page"]) for r in records
                                         if r["gutter_status"] == "needs_review"}),
        "curved_brace_shape_candidates": sum(r["curly_brace_candidate_count"] for r in records),
        "verified_accuracy": None,
        "warning": "Counts are diagnostic proposals only; human_review.csv is still required.",
    }
    _write_json(output / "diagnostic_summary.json", summary)
    return summary


def score_reviews(output: Path) -> dict:
    review_path = output / "human_review.csv"
    if not review_path.is_file():
        raise FileNotFoundError(f"Review CSV not found: {review_path}")
    with review_path.open(encoding="utf-8", newline="") as f:
        reviews = list(csv.DictReader(f))
    summaries = []
    years = sorted({int(r["year"]) for r in reviews})
    groups = [
        ("all", reviews),
        ("development", [r for r in reviews if r["role"] == "development"]),
        ("holdout", [r for r in reviews if r["role"] == "holdout"]),
    ]
    for year in years:
        groups.append((f"year_{year}", [r for r in reviews if int(r["year"]) == year]))
        groups.append((f"holdout_{year}", [r for r in reviews if int(r["year"]) == year
                                            and r["role"] == "holdout"]))
    for label, rows in groups:
        for metric in REVIEW_METRICS:
            vals = [str(r.get(metric, "")).strip().lower() for r in rows]
            if any(v not in ("", "yes", "no", "na") for v in vals):
                raise ValueError(f"Invalid value for {metric}: use yes, no, na or blank")
            yes, no, na = vals.count("yes"), vals.count("no"), vals.count("na")
            evaluated = yes + no
            summaries.append({"sample": label, "metric": metric,
                              "correct": yes, "incorrect": no, "not_applicable": na,
                              "unreviewed": len(vals) - evaluated - na,
                              "accuracy": round(yes / evaluated, 4) if evaluated else ""})
    _write_csv(output / "human_validation_summary.csv",
               ("sample", "metric", "correct", "incorrect", "not_applicable", "unreviewed", "accuracy"),
               summaries)
    return {"review_rows": len(reviews),
            "evaluated_checks": sum(row["correct"] + row["incorrect"] for row in summaries
                                    if row["sample"] == "all"),
            "report": "human_validation_summary.csv"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--years", type=int, nargs="+", default=list(range(1878, 1886)))
    p.add_argument("--per-year", type=int, default=5,
                   help="Five samples per year by default; endpoints, middle and deterministic random pages")
    p.add_argument("--seed", type=int, default=1882)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--plan-only", action="store_true", help="Freeze sample selection without opening PDFs")
    p.add_argument("--score-only", action="store_true", help="Evaluate yes/no human reviews, no image processing")
    p.add_argument("--reset-plan", action="store_true", help="Explicitly discard the old frozen sample selection")
    args = p.parse_args(argv)
    repo = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    if args.score_only:
        print(json.dumps(score_reviews(output), indent=2))
        return 0
    config = repo / "schema" / "situation_financiere" / "page_ranges.json"
    if not config.is_file():
        p.error(f"Missing {config}")
    years = sorted(set(args.years))
    plan = frozen_plan(config, output, years=years, per_year=args.per_year,
                       seed=args.seed, reset=args.reset_plan)
    print(f"Frozen sample: {len(plan['samples'])} PDF pages from {len(years)} years, "
          f"seed={args.seed}; {output / 'sample_plan.json'}")
    if args.plan_only:
        for s in plan["samples"]:
            print(f"  {s['year']} p{s['pdf_page']:04d}: {s['role']}")
        return 0
    result = run_benchmark(repo, output, plan, args.dpi)
    print(json.dumps(result, indent=2))
    print(f"Open overlays under {output / 'overlays'} and fill {output / 'human_review.csv'}")
    print("No OCR/API requests made. Geometric proposals are NOT verified observations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
