#!/usr/bin/env python3
"""Extract municipal-finance tables from approved Stage 2 image panels.

Commands:
  plan  Count OCR-ready table images, excluding printed pages under review.
  ocr   Transcribe the approved images with Mistral OCR (or --dry-run).
  build Parse verified-image OCR caches into provisional commune-year records.

This program never renders, crops, or re-segments PDFs. Source PDFs are
opened locally only to validate the confirmed page range and provenance.
Extraordinary-tax attribution remains provisional pending its separate review.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf as fitz
import pandas as pd
from bs4 import BeautifulSoup

YEARS = tuple(range(1878, 1886))
ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "raw_data" / "situation_financiere_communes"
DERIVED = ROOT / "derived" / "situation_financiere"
CONFIG = ROOT / "schema" / "situation_financiere"
RANGES = CONFIG / "page_ranges.json"
MAPPINGS = CONFIG / "column_mappings.json"
MODEL = "mistral-ocr-4-1"
OCR_URL = "https://api.mistral.ai/v1/ocr"
SEGMENTED_IMAGES = DERIVED / "table_panels"
SEGMENTED_NAME = re.compile(
    r"^p(?P<page>\d{4})_(?P<side>single|left|right)_col(?P<panel>[12])_(?P<dpi>\d+)dpi\.png$"
)
ACCEPTED_SEGMENTATION_STATUSES = frozenset({"accepted", "accepted_single_table"})

def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def unaccent(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(x for x in s if not unicodedata.combining(x)).upper()


def normalized(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", unaccent(s)).strip()


def find_pdfs() -> dict[int, Path]:
    """Exactly one PDF with an unambiguous year in its file name."""
    if not INPUT.exists():
        raise FileNotFoundError(f"Create {INPUT} and put the annual PDFs there")
    found: dict[int, Path] = {}
    for path in sorted(INPUT.glob("*.pdf")):
        matches = [y for y in YEARS if re.search(rf"(?<!\d){y}(?!\d)", path.stem)]
        if len(matches) != 1:
            print(f"Skipping PDF without one 1878-1885 year: {path.name}")
            continue
        y = matches[0]
        if y in found:
            raise ValueError(f"Two PDFs for {y}: {found[y]} and {path}")
        found[y] = path
    if not found:
        raise FileNotFoundError(f"No year-labelled PDFs in {INPUT}")
    return found


def select_years(args, pdfs: dict[int, Path]) -> list[int]:
    chosen = args.years or sorted(pdfs)
    absent = sorted(set(chosen) - set(pdfs))
    if absent:
        raise FileNotFoundError(f"Missing PDF(s) for years {absent} in {INPUT}")
    return chosen


def valid_range(year: int, pdf: fitz.Document) -> tuple[int, int]:
    spec = read_json(RANGES, {}).get(str(year), {})
    if not spec.get("confirmed", False):
        raise ValueError(f"{year}: page range not confirmed in {RANGES}")
    a, b = spec.get("start"), spec.get("end")
    if not isinstance(a, int) or not isinstance(b, int) or not 1 <= a <= b <= len(pdf):
        raise ValueError(f"{year}: bad range {a}-{b}; PDF has {len(pdf)} pages")
    if spec.get("pdf") and spec["pdf"] != Path(pdf.name).name:
        raise ValueError(f"{year}: range was prepared for {spec['pdf']}, not {Path(pdf.name).name}")
    return a, b


def selected_panel_sources(years: list[int] | None, dpi: int = 200) -> tuple[dict, list]:
    """Return only accepted Stage 2 images and their original PDF provenance."""
    pdfs = find_pdfs()
    selected = years if years is not None else sorted(pdfs)
    absent = sorted(set(selected) - set(pdfs))
    if absent:
        raise FileNotFoundError(f"Missing PDF(s) for years {absent} in {INPUT}")
    active = {}
    counts = []
    for year in selected:
        with fitz.open(pdfs[year]) as pdf:
            first, last = valid_range(year, pdf)
        panels, pending = segmented_images(year, range(first, last + 1), dpi=dpi)
        for page, side, panel, image in panels:
            active[(year, page, side, panel)] = (pdfs[year].name, image)
        counts.append((year, last - first + 1, len(panels), pending))
    return active, counts


def plan(args) -> None:
    dpi = getattr(args, "dpi", 200)
    _, totals = selected_panel_sources(args.years, dpi=dpi)
    print("Year | PDF scans | Accepted table images | Printed pages pending review")
    for year, scans, images, pending in totals:
        print(f"{year:4d} | {scans:9d} | {images:21d} | {pending:28d}")
    print(f"Total: {sum(x[2] for x in totals)} OCR-ready table images; "
          f"{sum(x[3] for x in totals)} review-only printed pages excluded")
    print("No OCR was performed. Stage 2 pending-review images are not OCR-ready.")


def segmented_images(year: int, pages: list[int] | range | None = None,
                     dpi: int = 200) -> tuple[list[tuple[int, str, str, Path]], int]:
    """Return OCR-ready panels from the Stage 2 manifest, never review candidates.

    The manifest is authoritative: not every PNG found on disk is necessarily
    approved. Resolve image *names* within this repo rather than using the
    absolute source paths stored by a prior run on another machine.
    """
    folder = SEGMENTED_IMAGES / str(year)
    manifest = folder / "table_split_review.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Missing Stage 2 manifest: {manifest}. Run split_tables.py --year {year} first."
        )
    wanted = set(pages) if pages is not None else None
    result: list[tuple[int, str, str, Path]] = []
    pending = 0
    seen: set[tuple[int, str, str]] = set()
    encountered: set[int] = set()
    found_sides: dict[int, set[str]] = {}
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                page = int(row["pdf_page"])
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid pdf_page in {manifest}: {row!r}") from exc
            if wanted is not None and page not in wanted:
                continue
            encountered.add(page)
            side = row.get("printed_side", "")
            if side not in {"single", "left", "right"}:
                raise ValueError(f"Bad printed_side for {year} p.{page}: {side!r}")
            found_sides.setdefault(page, set())
            if side in found_sides[page]:
                raise ValueError(f"Duplicate Stage 2 row for {year} p.{page} {side}")
            found_sides[page].add(side)
            status = row.get("status", "")
            if status not in ACCEPTED_SEGMENTATION_STATUSES:
                pending += 1
                continue
            n_tables = 0
            for col, panel in (("table_1", "col1"), ("table_2", "col2")):
                value = (row.get(col) or "").strip()
                if not value:
                    continue
                name = Path(value).name
                match = SEGMENTED_NAME.fullmatch(name)
                if (match is None or int(match["page"]) != page or
                    match["side"] != side or match["panel"] != panel[-1]):
                    raise ValueError(f"Unexpected accepted panel file {value!r} in {manifest}")
                if int(match["dpi"]) != dpi:
                    raise ValueError(
                        f"{year} p.{page}: segmentation image {name} has dpi={match['dpi']}; "
                        f"requested --dpi {dpi}. Reuse --dpi {match['dpi']} or rerun segmentation."
                    )
                image = folder / "accepted" / name
                if not image.is_file():
                    raise FileNotFoundError(f"Accepted panel missing: {image}")
                ident = (page, side, panel)
                if ident in seen:
                    raise ValueError(f"Duplicate accepted panel in {manifest}: {ident}")
                seen.add(ident)
                n_tables += 1
                result.append((page, side, panel, image))
            if n_tables < 1 or n_tables != int(row.get("table_count") or n_tables):
                raise ValueError(f"{year} p.{page} {side}: manifest table_count mismatch")
    if wanted is not None and not wanted.issubset(encountered):
        raise ValueError(f"{year}: missing Stage 2 manifest rows for pages "
                         f"{sorted(wanted - encountered)}")
    # With explicit scan configuration, do not silently miss a whole printed
    # page (e.g. the right half of a spread) when upstream processing failed.
    spec = read_json(RANGES, {}).get(str(year), {})
    for page in (wanted if wanted is not None else encountered):
        layout = spec.get("scan_layout_overrides", {}).get(
            str(page), spec.get("scan_layout", "auto"))
        if layout == "spread":
            expected = set(spec.get("page_sides", {}).get(str(page), ["left", "right"]))
        elif layout == "single":
            expected = {"single"}
        else:
            continue
        if found_sides.get(page, set()) != expected:
            raise ValueError(
                f"{year} p.{page}: Stage 2 manifest has printed sides "
                f"{sorted(found_sides.get(page, set()))}, expected {sorted(expected)}"
            )
    return sorted(result, key=lambda item: (item[0],
                  {"single": 0, "left": 0, "right": 1}[item[1]], item[2])), pending


def api_ocr(image: Path, model: str) -> dict:
    import requests
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("Set MISTRAL_API_KEY in your shell; never store it in a source file")
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    req = {"model": model, "document": {"type": "image_url",
                                        "image_url": f"data:image/png;base64,{encoded}"},
           "table_format": "html", "include_blocks": True,
           "confidence_scores_granularity": "word"}
    for attempt in range(5):
        try:
            res = requests.post(OCR_URL, json=req,
                                headers={"Authorization": f"Bearer {key}",
                                         "Content-Type": "application/json"},
                                timeout=(20, 240))
            if res.status_code in {429, 500, 502, 503, 504} and attempt < 4:
                delay = min(int(res.headers.get("Retry-After", 2 ** (attempt + 1))), 30)
                print(f"OCR HTTP {res.status_code}, retry in {delay}s")
                time.sleep(delay)
                continue
            if not res.ok:
                raise RuntimeError(
                    f"Mistral OCR HTTP {res.status_code}: {res.text[:1500]}"
                )
            return res.json()
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("OCR retries exhausted")


def ocr(args) -> None:
    """OCR approved Stage 2 panels; never regenerate images from PDFs."""
    pdfs = find_pdfs()
    selected = select_years(args, pdfs)
    dry_run = getattr(args, "dry_run", False)
    dpi = getattr(args, "dpi", 200)
    if dpi <= 0:
        raise ValueError("--dpi must be positive")
    if (args.max_pages == 0 and not getattr(args, "pages", None)
            and not args.yes_all and not dry_run):
        raise ValueError("Full paid OCR run requires --yes-all. For a pilot use "
                         "--max-pages 3 or --pages 245")
    if not dry_run and not os.environ.get("MISTRAL_API_KEY"):
        raise ValueError("Set MISTRAL_API_KEY before running paid OCR")
    count_sent = count_skipped = pending_total = 0
    for year in selected:
        with fitz.open(pdfs[year]) as pdf:
            start, end = valid_range(year, pdf)
        if getattr(args, "pages", None):
            invalid = [p for p in args.pages if p < start or p > end]
            if invalid:
                raise ValueError(f"{year}: requested page(s) {invalid} lie outside the "
                                 f"confirmed {start}-{end} Pas-de-Calais range")
            requested = sorted(set(args.pages))
        else:
            end = min(end, start + args.max_pages - 1) if args.max_pages else end
            requested = list(range(start, end + 1))
        tasks, pending = segmented_images(year, requested, dpi=dpi)
        pending_total += pending
        if pending:
            print(f"{year}: skipping {pending} review-only printed pages; "
                  "these are NOT sent to OCR.")
        if not tasks:
            print(f"{year}: no OCR-ready images selected")
            continue
        if dry_run:
            description = ', '.join(image.name for _, _, _, image in tasks[:6])
            print(f"{year}: {len(tasks)} OCR-ready images; first: {description}")
            continue
        cache_base = DERIVED / "ocr" / "mistral" / "segmented"
        for page_no, side, panel, image in tasks:
            out = cache_base / str(year) / f"p{page_no:04d}_{side}_{panel}.json"
            checksum = hashlib.sha256(image.read_bytes()).hexdigest()
            old = read_json(out, {})
            if (not args.force and old.get("image_sha256") == checksum
                    and old.get("model") == args.model
                    and old.get("segmentation_source") == "segmented"
                    and old.get("panel") == panel
                    and old.get("image") == str(image.relative_to(ROOT))
                    and old.get("source_pdf") == pdfs[year].name):
                count_skipped += 1
                continue
            print(f"OCR {year}, PDF page {page_no}, {side}_{panel}, "
                  f"model={args.model}", flush=True)
            response = api_ocr(image, args.model)
            atomic_json(out, {"year": year, "pdf_page": page_no, "side": side,
                              "panel": panel, "segmentation_source": "segmented",
                              "source_pdf": pdfs[year].name,
                              "image": str(image.relative_to(ROOT)),
                              "image_sha256": checksum, "model": args.model,
                              "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
                              "response": response})
            count_sent += 1
    if dry_run:
        print(f"Dry run: no OCR requests. Skipped {pending_total} pending-review printed pages.")
    else:
        print(f"Done: {count_sent} OCR requests, {count_skipped} cache hits, "
              f"{pending_total} review-only printed pages skipped.")


def html_grid(html: str) -> list[list[str]]:
    """Rectangularize HTML tables, including nested header colspan/rowspan."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []
    rows = []
    pending: dict[int, tuple[str, int]] = {}
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue
        cells = tr.find_all(["td", "th"], recursive=False)
        row: list[str] = []
        col = 0
        for cell in cells:
            while col in pending:
                val, count = pending[col]
                row.append(val)
                if count <= 1:
                    del pending[col]
                else:
                    pending[col] = (val, count - 1)
                col += 1
            value = cell.get_text(" ", strip=True)
            colspan = max(1, int(cell.get("colspan", 1)))
            rowspan = max(1, int(cell.get("rowspan", 1)))
            for _ in range(min(colspan, 80)):
                row.append(value)
                if rowspan > 1:
                    pending[col] = (value, rowspan - 1)
                col += 1
        while col in pending:
            val, count = pending[col]
            row.append(val)
            if count <= 1:
                del pending[col]
            else:
                pending[col] = (val, count - 1)
            col += 1
        rows.append(row)
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows]


def table_sources(page: dict) -> list[tuple[str, str]]:
    """Mistral table objects hold HTML separately from page markdown."""
    found = []
    for i, table in enumerate(page.get("tables") or []):
        if isinstance(table, dict):
            content = next((table.get(k) for k in ("content", "html", "data")
                            if isinstance(table.get(k), str)), "")
            name = str(table.get("id", table.get("table_id", i)))
        else:
            continue
        if "<table" in content.lower():
            found.append((name, content))
    # Some OCR responses put tables directly into Markdown rather than tables[].
    if not found:
        markdown = str(page.get("markdown") or "")
        for i, hit in enumerate(re.findall(r"<table\b.*?</table>", markdown, flags=re.I | re.S)):
            found.append((f"markdown_html_{i}", hit))
    # If the provider ignored HTML table extraction, retain basic Markdown
    # tables instead of silently producing an empty structured dataset.
    if not found:
        from html import escape
        markdown = str(page.get("markdown") or "")
        groups = []
        running = []
        for line in markdown.splitlines() + [""]:
            if line.strip().startswith("|") and line.strip().endswith("|"):
                running.append(line.strip())
            elif running:
                if len(running) >= 2:
                    groups.append(running)
                running = []
        for i, group in enumerate(groups):
            trs = []
            for line in group:
                cells = [x.strip() for x in line.strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", x) for x in cells):
                    continue
                trs.append("<tr>" + "".join(f"<td>{escape(x)}</td>" for x in cells) + "</tr>")
            if trs:
                found.append((f"markdown_pipe_{i}", "<table>" + "".join(trs) + "</table>"))
    return found


def row_type(cells: list[str]) -> str:
    content = [str(c).strip() for c in cells if str(c).strip()]
    if not content:
        return "blank"
    joined = normalized(" ".join(content))
    first = normalized(content[0])
    if re.search(r"\b(TOTAL|TOTAUX|A REPORTER|REPORT)\b", first):
        return "subtotal"
    if re.search(r"\b(CANTON|ARRONDISSEMENT|DEPARTEMENT|COMMUNES|CENTIMES|NOMS|POPULATION)\b", first):
        return "header"
    if re.search(r"\b(CANTON|ARRONDISSEMENT)\b", joined) and len(content) < 4:
        return "header"
    # Avoid treating a number as a commune name; retain in raw_rows for review.
    if not re.search(r"[A-Za-zÀ-ÿ]", content[0]) or len(first) <= 1:
        return "uncertain"
    trailing = content[1:]
    if sum(bool(re.search(r"\d", val)) for val in trailing) >= 2:
        return "candidate_commune"
    return "uncertain"


def french_number(raw: str, kind: str) -> tuple[float | int | None, str]:
    """Only parse fields whose YEAR-SPECIFIC meaning has been reviewed."""
    s = str(raw or "").strip().replace("\u00a0", " ")
    if not s or s in {"-", "—", "»", "«", ".", ".."}:
        return None, "missing_or_symbol"
    s = s.replace(" ", "").replace("'", "")
    if not re.fullmatch(r"[0-9.,]+", s):
        return None, "ocr_uncertain"
    if kind == "integer":
        # A comma followed by three digits is commonly a thousands separator
        # in the printed 19th-century population/revenue integer columns.
        if re.fullmatch(r"\d{1,3}(?:[,.]\d{3})+", s):
            return int(s.replace(",", "").replace(".", "")), "parsed"
        if re.fullmatch(r"\d+", s):
            return int(s), "parsed"
        return None, "numeric_ambiguous"
    if kind in {"decimal", "decimal_fr", "money"}:
        # Ambiguous commas with three decimals are not silently normalized.
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", s):
            return None, "numeric_ambiguous"
        if s.count(",") <= 1 and s.count(".") == 0:
            s = s.replace(",", ".")
        elif s.count(",") == 1 and s.count(".") >= 1:
            s = s.replace(".", "").replace(",", ".")
        elif s.count(".") > 1:
            return None, "numeric_ambiguous"
        try:
            return float(s), "parsed"
        except ValueError:
            return None, "ocr_uncertain"
    raise ValueError(f"Unknown column type {kind!r}")


def layout_for(year: int, count: int, mappings: dict, page_no: int, side: str) -> dict | None:
    year_cfg = mappings.get(str(year), {})
    layouts = year_cfg.get("layouts", [])
    candidates = [x for x in layouts if x.get("column_count") == count]
    override = year_cfg.get("page_layout_overrides", {}).get(f"p{page_no}_{side}")
    if override:
        matches = [x for x in candidates if x.get("name") == override]
        if len(matches) != 1:
            raise ValueError(f"{year} page {page_no} {side}: bad layout override {override!r}")
        return matches[0]
    return candidates[0] if len(candidates) == 1 else None


def parse_cache(years: list[int] | None = None, dpi: int = 200) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse only OCR responses associated with currently accepted panels."""
    active, _ = selected_panel_sources(years, dpi=dpi)
    mappings = read_json(MAPPINGS, {})
    raw_rows = []
    candidates = []
    inventory = []
    page_audit = []
    directory = DERIVED / "ocr" / "mistral" / "segmented"
    for path in sorted(directory.glob("*/p*_*.json")):
        record = read_json(path, {})
        year, page_no, side = record["year"], record["pdf_page"], record["side"]
        panel = record.get("panel")
        if years is not None and year not in years:
            continue
        expected = active.get((year, page_no, side, panel))
        if (expected is None or record.get("segmentation_source") != "segmented"
                or expected[0] != record.get("source_pdf")
                or str(expected[1].relative_to(ROOT)) != record.get("image")):
            continue
        image_path = expected[1]
        if (not image_path.is_file() or
                hashlib.sha256(image_path.read_bytes()).hexdigest() != record.get("image_sha256")):
            print(f"Skipping changed/missing OCR image {path}; rerun OCR", file=sys.stderr)
            continue
        response = record.get("response", {})
        ocr_page = (response.get("pages") or [{}])[0]
        sources = table_sources(ocr_page)
        page_context = str(ocr_page.get("markdown", ""))
        from territorial_validation import markdown_table_prefaces
        prefaces = markdown_table_prefaces(page_context)
        page_audit.append({"year": year, "pdf_page": page_no, "side": side, "panel": panel,
                           "source_pdf": record.get("source_pdf"),
                           "source_image": record.get("image"),
                           "model": record.get("model"),
                           "structured_tables_found": len(sources),
                           "markdown_characters": len(page_context),
                           "status": "tables_found" if sources else "no_structured_table_review"})
        for table_id, html in sources:
            rows = html_grid(html)
            if not rows:
                continue
            width = len(rows[0])
            layout = layout_for(year, width, mappings, page_no, "full" if side == "single" else side)
            cols = layout.get("columns", {}) if layout else {}
            inventoried = {"year": year, "pdf_page": page_no, "side": side, "panel": panel,
                           "table_id": table_id, "column_count": width,
                           "layout": layout.get("name") if layout else None,
                           "n_rows": len(rows), "html_header_excerpt": " | ".join(rows[0])[:350],
                           "image": record["image"]}
            inventory.append(inventoried)
            for i, cells in enumerate(rows, start=1):
                typ = row_type(cells)
                label = f"-{panel}" if panel else ""
                ident = f"{year}-p{page_no:04d}-{side}{label}-{table_id}-r{i:03d}"
                base = {"row_id": ident, "year": year, "pdf_page": page_no,
                        "side": side, "panel": panel, "table_id": table_id, "table_row": i,
                        "row_type": typ, "column_count": width,
                        "layout": layout.get("name") if layout else None,
                        "source_pdf": record["source_pdf"], "source_image": record["image"],
                        "ocr_model": record["model"],
                        "ocr_page_confidence": (ocr_page.get("confidence_scores") or {}).get(
                            "average_page_confidence_score"),
                        "raw_cells_json": json.dumps(cells, ensure_ascii=False),
                        "ocr_markdown_excerpt": page_context[:500],
                        "ocr_table_preface": prefaces.get(table_id, "")}
                for k, cell in enumerate(cells, start=1):
                    base[f"raw_col_{k:02d}"] = cell
                raw_rows.append(base)
                if typ != "candidate_commune":
                    continue
                entry = dict(base)
                entry["commune_raw"] = cells[0]
                flags = []
                if not layout:
                    flags.append("unmapped_columns")
                for k, spec in cols.items():
                    idx = int(k) - 1
                    if not 0 <= idx < len(cells):
                        flags.append(f"mapping_out_of_bounds:{k}")
                        continue
                    name = spec["name"]
                    if name in {"row_id", "year", "pdf_page", "cassini_id", "commune_raw"}:
                        raise ValueError(f"Reserved column name in mapping: {name}")
                    kind = spec.get("type", "text")
                    if kind == "text":
                        entry[name] = cells[idx]
                    else:
                        value, status = french_number(cells[idx], kind)
                        entry[name] = value
                        if status in {"ocr_uncertain", "numeric_ambiguous"}:
                            flags.append(f"{name}:{status}")
                entry["extraction_flags"] = ";".join(flags)
                candidates.append(entry)
    from territorial_validation import annotate_territories
    raw_frame, candidate_frame = annotate_territories(
        pd.DataFrame(raw_rows), pd.DataFrame(candidates))
    return (raw_frame, candidate_frame,
            pd.DataFrame(inventory), pd.DataFrame(page_audit))


def apply_commune_match(df: pd.DataFrame) -> pd.DataFrame:
    """Uses your existing code/commune_matcher.py and communematch package."""
    try:
        import commune_matcher as cm
    except ImportError as exc:
        raise RuntimeError("Install your project's 'communematch' dependency, or use --no-match") from exc
    matcher = cm.get_matcher()
    matched = matcher.translate_frame(df.copy(), column="commune_raw")
    matched = matched.rename(columns={"cm_id": "cassini_id", "cm_name": "commune_cleaned"})
    if "cassini_id" not in matched:
        raise RuntimeError("commune_matcher did not return cm_id; check package version")
    matched["cassini_id"] = pd.to_numeric(matched["cassini_id"], errors="coerce").astype("Int64")
    return matched


def export_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8")
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    except ImportError:
        print("Warning: pyarrow not installed, wrote CSV only. Install pyarrow for Parquet.")


def build(args) -> None:
    dpi = getattr(args, "dpi", 200)
    raw, communes, inventory, page_audit = parse_cache(args.years, dpi=dpi)
    output = DERIVED / "output"
    if not page_audit.empty:
        export_frame(page_audit, output / "situation_financiere_page_audit.parquet")
    if raw.empty:
        raise RuntimeError(f"No structured HTML tables found under {DERIVED / 'ocr' / 'mistral' / 'segmented'}. "
                           "Check OCR JSON responses; plain-text Tesseract is not enough.")
    export_frame(raw, output / "situation_financiere_all_rows.parquet")
    export_frame(inventory, output / "situation_financiere_column_inventory.parquet")
    from territorial_validation import canton_subtotal_checks
    active, _ = selected_panel_sources(args.years, dpi=dpi)
    expected = {(year, page, side) for year, page, side, panel in active}
    available = set()
    if not page_audit.empty:
        available = {(int(r.year), int(r.pdf_page), str(r.side))
                     for r in page_audit.itertuples()
                     if int(r.structured_tables_found) > 0}
    canton_checks = canton_subtotal_checks(raw, communes,
                                           expected_regions=expected,
                                           available_regions=available)
    export_frame(canton_checks, output / "situation_financiere_canton_validation.parquet")
    if communes.empty:
        print("No candidate commune rows. Review column_inventory and cached OCR.")
        return
    # Parse tax pairs offline; unassigned continuation lines remain review items.
    from extraordinary_tax_parser import enrich_with_extraordinary_taxes
    communes, tax_entries, detached_tax_rows = enrich_with_extraordinary_taxes(communes, raw)
    if not tax_entries.empty:
        export_frame(tax_entries, output / "situation_financiere_extraordinary_tax_entries.parquet")
    if not detached_tax_rows.empty:
        export_frame(detached_tax_rows, output / "situation_financiere_unassigned_tax_rows.parquet")
    if not args.no_match:
        communes = apply_commune_match(communes)
    else:
        communes["cassini_id"] = pd.Series([pd.NA] * len(communes), dtype="Int64")
    communes["year"] = pd.to_numeric(communes["year"], errors="raise").astype("int16")
    communes["verification_status"] = "ocr_unverified"
    counts = Counter((int(y), int(cid)) for y, cid in zip(communes["year"], communes["cassini_id"])
                     if pd.notna(cid))
    mismatched_cantons = set()
    if not canton_checks.empty:
        mismatched_cantons = set(zip(
            canton_checks.loc[canton_checks["status"] == "ocr_discrepancy_review_scan", "year"],
            canton_checks.loc[canton_checks["status"] == "ocr_discrepancy_review_scan", "canton_segment_id"]))
    qa = []
    for row in communes.to_dict("records"):
        flags = [x for x in str(row.get("extraction_flags") or "").split(";") if x]
        cid = row.get("cassini_id")
        if pd.isna(cid):
            flags.append("unmatched_commune")
        elif counts[(int(row["year"]), int(cid))] > 1:
            flags.append("duplicate_year_cassini_id")
        cm_status = str(row.get("cm_status", ""))
        if cm_status and not any(s in cm_status.lower() for s in ["exact", "matched"]):
            flags.append(f"check_match_status:{cm_status}")
        if row.get("ocr_page_confidence") is not None:
            try:
                if float(row["ocr_page_confidence"]) < args.min_confidence:
                    flags.append("low_page_ocr_confidence")
            except (ValueError, TypeError):
                pass
        tax_status = row.get("extraordinary_tax_pair_status", "")
        if tax_status and tax_status != "ocr_pairs_complete_unverified":
            flags.append(f"tax:{tax_status}")
        if pd.isna(row.get("canton_segment_id")):
            flags.append("missing_canton_context")
        elif (int(row["year"]), int(row["canton_segment_id"])) in mismatched_cantons:
            flags.append("canton_ocr_subtotal_discrepancy_review")
        qa.append({"row_id": row["row_id"], "year": row["year"],
                   "cassini_id": cid, "commune_raw": row["commune_raw"],
                   "arrondissement": row.get("arrondissement"),
                   "canton": row.get("canton"),
                   "canton_segment_id": row.get("canton_segment_id"),
                   "pdf_page": row["pdf_page"], "source_image": row["source_image"],
                   "review_flags": ";".join(sorted(set(flags)))})
    qa_frame = pd.DataFrame(qa)
    communes["review_flags"] = qa_frame["review_flags"].values
    export_frame(communes, output / "situation_financiere_commune_year_candidates.parquet")
    export_frame(qa_frame, output / "situation_financiere_qc.parquet")
    export_frame(qa_frame[qa_frame["review_flags"].ne("")], output / "situation_financiere_review_queue.parquet")
    diag = communes.groupby("year", dropna=False).agg(
        candidate_rows=("row_id", "size"),
        matched_communes=("cassini_id", "nunique"),
        unmatched=("cassini_id", lambda x: int(x.isna().sum())),
        flagged_rows=("review_flags", lambda x: int(x.ne("").sum())),
    ).reset_index()
    diag.to_csv(output / "situation_financiere_year_summary.csv", index=False)
    print(diag.to_string(index=False))
    print(f"Outputs: {output}")
    print("These are OCR CANDIDATES, not verified values. Review the queue, column layouts, and totals.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for cmd in ("plan", "ocr", "build"):
        q = sub.add_parser(cmd)
        q.add_argument("--years", type=int, nargs="+", choices=YEARS,
                       help="Limit to specified years (default: all PDFs present)")
        if cmd in ("plan", "ocr", "build"):
            q.add_argument("--dpi", type=int, default=200,
                           help="DPI of approved Stage 2 panel images (default: 200)")
        if cmd == "ocr":
            q.add_argument("--dry-run", action="store_true", help="Inspect images with no API calls")
            q.add_argument("--model", default=MODEL, help="Pinned Mistral OCR model ID")
            q.add_argument("--max-pages", type=int, default=0,
                           help="Pilot: max PDF scans per year, starting at confirmed range start")
            q.add_argument("--pages", type=int, nargs="+",
                           help="OCR selected 1-based PDF page numbers within each confirmed range")
            q.add_argument("--yes-all", action="store_true", help="Required for a paid full-range run")
            q.add_argument("--force", action="store_true", help="Replace cached responses")
            q.add_argument("--workers", type=int, default=1,
                           help="Concurrent OCR requests (1=original sequential; try 4)")
            q.add_argument("--rpm", type=float, default=0.0,
                           help="Optional maximum new OCR job starts/minute (0=unpaced)")
        if cmd == "build":
            q.add_argument("--no-match", action="store_true", help="Skip communematch temporarily")
            q.add_argument("--min-confidence", type=float, default=0.75)
    args = p.parse_args()
    if args.dpi <= 0:
        p.error("--dpi must be positive")
    if args.command == "plan":
        plan(args)
    elif args.command == "ocr":
        if args.max_pages < 0:
            p.error("--max-pages must be >= 0")
        if args.workers > 1:
            from parallel_ocr import parallel_ocr
            parallel_ocr(args)
        elif args.workers == 1:
            ocr(args)
        else:
            p.error("--workers must be positive")
    else:
        build(args)


if __name__ == "__main__":
    main()
