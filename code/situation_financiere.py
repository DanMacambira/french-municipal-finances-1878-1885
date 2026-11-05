#!/usr/bin/env python3
"""Extract Pas-de-Calais commune tables (1878-1885) from national PDFs.

Run from the project root: python code/situation_financiere.py --help

STAGES
  discover   Scan national PDFs locally; draft page ranges without any API calls.
  preview    Make contact sheets to check a page range visually.
  plan       Estimate confirmed regional pages and standard API call cost.
  ocr        Render ONLY confirmed Pas-de-Calais pages: full single-page scans
             or selected halves of double-page spreads, and cache OCR results.
  build      Parse all cached HTML tables, preserve original cells, optionally
             apply year/layout-specific column mappings, then match communes.

Never sends national PDFs to an API. Does not delete input PDFs. Does not
silently convert an uncertain OCR field into a research-ready number.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
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

import pymupdf as fitz  # Modern PyMuPDF import; keep fitz alias for internal uses
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


def scan_header(page: fitz.Page, use_tesseract: bool) -> str:
    top = fitz.Rect(0, 0, page.rect.width, page.rect.height * 0.30)
    txt = page.get_text("text", clip=top, sort=True)
    if use_tesseract:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as ex:
            raise RuntimeError("Install pytesseract and pillow for --header-ocr") from ex
        pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), clip=top, colorspace=fitz.csGRAY)
        image = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        txt += "\n" + pytesseract.image_to_string(image, lang="fra+eng", config="--psm 6")
    return normalized(txt)


def heading_kind(text: str) -> str | None:
    """Conservative heading detection; returns 'pas_de_calais' or 'other'."""
    # A PDF page may contain 'Pas-de-Calais' incidentally; require a heading.
    if re.search(r"DEPARTEMENT\s+(?:DU|DE|DES|D)\s+PAS\s+DE\s+CALAIS", text):
        return "pas_de_calais"
    if re.search(r"DEPARTEMENT\s+(?:DU|DE|DES|D)\s+[A-Z](?:[A-Z ]{1,40})", text):
        return "other"
    return None


def discover(args) -> None:
    pdfs = find_pdfs()
    existing = read_json(RANGES, {})
    suggestions = []
    for year in select_years(args, pdfs):
        with fitz.open(pdfs[year]) as pdf:
            # First try the existing PDF text layer, at essentially no OCR cost.
            heads = [(i + 1, heading_kind(scan_header(page, False)))
                     for i, page in enumerate(pdf)]
            starts = [i for i, typ in heads if typ == "pas_de_calais"]
            if args.header_ocr and not starts:
                print(f"{year}: no department heading found in the text layer; "
                      f"scanning {len(pdf)} page headers with local Tesseract.", flush=True)
                heads = []
                for i, page in enumerate(pdf):
                    heads.append((i + 1, heading_kind(scan_header(page, True))))
                    if (i + 1) % 100 == 0:
                        print(f"  {year}: checked {i + 1}/{len(pdf)} headers", flush=True)
                starts = [i for i, typ in heads if typ == "pas_de_calais"]
            if starts:
                start = starts[0]
                endings = [i for i, typ in heads if i > starts[-1] and typ == "other"]
                end = (endings[0] - 1) if endings else None
                status = "suggested" if end is not None and end >= start else "needs_manual_range"
            else:
                start = end = None
                status = "no_heading_found"
            entry = existing.get(str(year), {})
            # Existing reviewed ranges are never overwritten by auto-discovery.
            if not entry.get("confirmed", False):
                # Preserve hand-entered half-page exclusions across discovery runs.
                existing[str(year)] = {**entry, "pdf": pdfs[year].name,
                                       "start": start, "end": end,
                                       "confirmed": False, "status": status}
            suggestions.append({"year": year, "pages": len(pdf), "heading_hits": len(starts),
                                "first_hit": starts[0] if starts else None,
                                "last_hit": starts[-1] if starts else None,
                                "start_suggested": start, "end_suggested": end,
                                "status": status})
    atomic_json(RANGES, existing)
    DERIVED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(suggestions).to_csv(DERIVED / "discovery_report.csv", index=False)
    print(f"Draft ranges: {RANGES}")
    print(f"Discovery diagnostics: {DERIVED / 'discovery_report.csv'}")
    print("CHECK each range against the PDF and set confirmed=true BEFORE OCR.")
    if not args.header_ocr:
        print("If no headings are found (scanned PDF), retry discover --header-ocr,")
        print("or enter 1-based start/end page numbers in page_ranges.json manually.")


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


def sides_for_page(year: int, page_no: int) -> tuple[str, ...]:
    """Logical halves of a scanned PDF page belonging to the target department.

    Both halves are included by default. Configure exceptional boundary pages
    in page_ranges.json using, e.g., "page_sides": {"244": ["right"]}.
    This keeps the full source PDF and its 1-based page numbering intact.
    """
    spec = read_json(RANGES, {}).get(str(year), {})
    overrides = spec.get("page_sides", {})
    selected = overrides.get(str(page_no), ["left", "right"])
    if (not isinstance(selected, list) or not selected or
            any(side not in ("left", "right") for side in selected) or
            len(selected) != len(set(selected))):
        raise ValueError(f"{year} PDF page {page_no}: invalid page_sides {selected!r}")
    return tuple(side for side in ("left", "right") if side in selected)


def printed_page_regions(year: int, page_no: int, page: fitz.Page) -> tuple[str, ...]:
    """Choose a full scan or its left/right printed pages, without OCR.

    'auto' classifies using PDF page geometry, NOT the scanned image content.
    Use scan_layout / scan_layout_overrides to correct misclassified scans.
    The page_sides option continues to apply only to two-page spreads.
    """
    spec = read_json(RANGES, {}).get(str(year), {})
    overrides = spec.get("scan_layout_overrides", {})
    layout = overrides.get(str(page_no), spec.get("scan_layout", "auto"))
    if layout not in ("auto", "single", "spread"):
        raise ValueError(f"{year} PDF page {page_no}: invalid scan_layout {layout!r}")
    if layout == "auto":
        # Strongly landscape PDF sheets are likely to be scanned two-page spreads.
        # This is a heuristic: the operator must review preview --regions.
        layout = "spread" if page.rect.width > page.rect.height * 1.05 else "single"
    if layout == "single":
        if str(page_no) in spec.get("page_sides", {}):
            raise ValueError(f"{year} PDF page {page_no}: page_sides requires a spread; "
                             "set scan_layout_overrides for this page if needed")
        return ("full",)
    return sides_for_page(year, page_no)


def table_panels_for_page(year: int, page_no: int) -> int:
    """The number of parallel printed table blocks on each printed page.

    Set table_panels=2 (or a per-page override) ONLY after visually checking.
    Default 1 preserves previous extraction behaviour and cache names.
    """
    spec = read_json(RANGES, {}).get(str(year), {})
    overrides = spec.get("table_panel_overrides", {})
    count = overrides.get(str(page_no), spec.get("table_panels", 1))
    if type(count) is not int or count not in (1, 2):
        raise ValueError(f"{year} PDF page {page_no}: table_panels must be 1 or 2, got {count!r}")
    return count


def table_panel_split_for_page(year: int, page_no: int, region: str) -> float:
    """Position of the divider between two printed tables, as a width fraction.

    Defaults to 0.5; optional config overrides are keyed by PDF page + physical
    region, e.g. table_panel_split_overrides: {"p245_left": 0.521}.
    Inspect the actual crops before OCR; printed tables are not always centered.
    """
    spec = read_json(RANGES, {}).get(str(year), {})
    overrides = spec.get("table_panel_split_overrides", {})
    split = overrides.get(f"p{page_no}_{region}", spec.get("table_panel_split", 0.5))
    if isinstance(split, bool) or not isinstance(split, (int, float)) or not 0.35 < split < 0.65:
        raise ValueError(f"{year} PDF page {page_no} {region}: invalid table panel split {split!r}")
    return float(split)


def region_signature(year: int, page_no: int, region: str) -> str | None:
    """Detect whether the crop configuration has changed since the OCR run.

    Legacy unsplit regions have no signature, preserving their previous cache.
    """
    match = re.fullmatch(r"(full|left|right)_col[12]", region)
    if not match:
        return None
    payload = {"region": region,
               "split": table_panel_split_for_page(year, page_no, match.group(1)),
               "overlap_fraction": 0.007}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def regions_for_page(year: int, page_no: int, page: fitz.Page) -> tuple[str, ...]:
    """Physical printed pages, optionally split again into parallel table columns.

    Identifiers are stable provenance/cache keys, e.g. full_col1, right_col2.
    """
    base = printed_page_regions(year, page_no, page)
    if table_panels_for_page(year, page_no) == 1:
        return base
    return tuple(f"{region}_col{part}" for region in base for part in (1, 2))


def plan(args) -> None:
    """Count only approved regional pages; never send an API request."""
    pdfs = find_pdfs()
    totals = []
    for year in select_years(args, pdfs):
        with fitz.open(pdfs[year]) as pdf:
            a, b = valid_range(year, pdf)
            count = b - a + 1
            regions = [regions_for_page(year, p, pdf[p - 1]) for p in range(a, b + 1)]
            singles = sum(printed_page_regions(year, p, pdf[p - 1]) == ("full",)
                          for p in range(a, b + 1))
            spreads = count - singles
            requests = sum(len(r) for r in regions)
            totals.append((year, a, b, count, singles, spreads, requests))
    print("Year | First | Last | PDF pages | Single scans | Spreads | OCR requests")
    for year, a, b, count, singles, spreads, requests in totals:
        print(f"{year:4d} | {a:5d} | {b:4d} | {count:9d} | {singles:12d} | "
              f"{spreads:7d} | {requests:12d}")
    n_requests = sum(x[-1] for x in totals)
    print(f"Total: {sum(x[3] for x in totals)} source pages, {n_requests} image OCR requests")
    print("Auto layout is based on page aspect ratio, not an inspection of the printed content. "
          "Review make preview SF_ARGS='--regions ...' before paying for OCR.")
    print(f"Illustrative Mistral 4.1 standard list price at $4/1,000 pages: "
          f"USD {n_requests * .004:.2f} (not Batch API).")
    print("This assumes each separately submitted image is billed as one page; "
          "check actual provider usage/billing.")


def preview(args) -> None:
    from PIL import Image, ImageDraw
    pdfs = find_pdfs()
    for year in select_years(args, pdfs):
        with fitz.open(pdfs[year]) as pdf:
            if args.center:
                start, end = max(1, args.center - args.radius), min(len(pdf), args.center + args.radius)
            else:
                start, end = valid_range(year, pdf)
            pages = list(range(start, end + 1, max(1, args.every)))
            # --regions previews the ACTUAL images selected for the OCR call.
            # The ordinary preview continues to show unchanged source PDF pages.
            items = ([(p, side) for p in pages
                      for side in regions_for_page(year, p, pdf[p - 1])]
                     if args.regions else [(p, None) for p in pages])
            width, height = 380, 570
            columns = 4
            canvas = Image.new("RGB", (columns * width, ((len(items) + 3) // 4) * height), "white")
            draw = ImageDraw.Draw(canvas)
            for j, (p, side) in enumerate(items):
                if side is None:
                    pix = pdf[p - 1].get_pixmap(matrix=fitz.Matrix(0.75, 0.75), alpha=False)
                    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                else:
                    _, img_path = render_regions(pdf[p - 1], year, p, 96, (side,))[0]
                    with Image.open(img_path) as im:
                        image = im.convert("RGB")
                image.thumbnail((width - 20, height - 48))
                x, y = (j % columns) * width + 10, (j // columns) * height + 25
                canvas.paste(image, (x, y))
                suffix = f" / {side}" if side else ""
                draw.text((x, y - 18), f"{year} / PDF p.{p}{suffix}", fill="black")
            postfix = "_regions" if args.regions else ""
            out = DERIVED / "previews" / f"{year}_p{start}-{end}{postfix}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(out, quality=85)
            print(f"Wrote {out}")


def render_regions(page: fitz.Page, year: int, page_no: int, dpi: int,
                   sides: tuple[str, ...]) -> list[tuple[str, Path]]:
    """Render complete printed-page images; originals remain unchanged."""
    w, h = page.rect.width, page.rect.height
    mid = w / 2
    parts = {"full": page.rect,
             "left": fitz.Rect(0, 0, mid + w * .008, h),
             "right": fitz.Rect(mid - w * .008, 0, w, h)}
    rendered = []
    for side in sides:
        match = re.fullmatch(r"(full|left|right)(?:_col([12]))?", side)
        if not match:
            raise ValueError(f"Unknown OCR region {side!r}")
        printed_region, panel = match.groups()
        box = parts[printed_region]
        if panel is not None:
            # The divider is within the printed page, not necessarily at 50%.
            midpoint = box.x0 + box.width * table_panel_split_for_page(
                year, page_no, printed_region)
            pad = box.width * .007
            if panel == "1":
                box = fitz.Rect(box.x0, box.y0, midpoint + pad, box.y1)
            else:
                box = fitz.Rect(midpoint - pad, box.y0, box.x1, box.y1)
        out = DERIVED / "images" / str(year) / f"p{page_no:04d}_{side}_{dpi}dpi.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Rerender panels: their split fraction is configurable. The OCR cache
        # below compares hashes so an unchanged panel never causes a paid call.
        if panel is not None or not out.exists():
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=box,
                                  colorspace=fitz.csGRAY, alpha=False)
            pix.save(str(out))
        rendered.append((side, out))
    return rendered


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


def local_ocr(image: Path) -> dict:
    """Free transcription comparison; plain text does NOT reconstruct tables."""
    import pytesseract
    from PIL import Image
    txt = pytesseract.image_to_string(Image.open(image), lang="fra+eng", config="--psm 6")
    return {"model": "tesseract-local", "pages": [{"index": 0, "markdown": txt,
                                                       "tables": [], "confidence_scores": None}]}


def ocr(args) -> None:
    pdfs = find_pdfs()
    selected = select_years(args, pdfs)
    if args.max_pages == 0 and not getattr(args, "pages", None) and not args.yes_all:
        raise ValueError("Full run requires --yes-all. For a pilot use --max-pages 3 "
                         "or --pages 150 200 250")
    if args.engine == "mistral" and not os.environ.get("MISTRAL_API_KEY"):
        raise ValueError("Set MISTRAL_API_KEY before running paid OCR")
    count_sent = 0
    count_skipped = 0
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
                requested = range(start, end + 1)
            for page_no in requested:
                chosen_sides = regions_for_page(year, page_no, pdf[page_no - 1])
                for side, image in render_regions(pdf[page_no - 1], year, page_no,
                                                  args.dpi, chosen_sides):
                    out = DERIVED / "ocr" / args.engine / str(year) / f"p{page_no:04d}_{side}.json"
                    checksum = hashlib.sha256(image.read_bytes()).hexdigest()
                    signature = region_signature(year, page_no, side)
                    old = read_json(out, {})
                    chosen_model = args.model if args.engine == "mistral" else "tesseract-local"
                    if (not args.force and old.get("image_sha256") == checksum
                            and old.get("model") == chosen_model
                            and old.get("region_signature") == signature):
                        count_skipped += 1
                        continue
                    print(f"OCR {year}, PDF page {page_no}, {side}, engine={chosen_model}", flush=True)
                    response = api_ocr(image, args.model) if args.engine == "mistral" else local_ocr(image)
                    atomic_json(out, {"year": year, "pdf_page": page_no, "side": side,
                                      "source_pdf": pdfs[year].name, "image": str(image.relative_to(ROOT)),
                                      "image_sha256": checksum, "model": chosen_model,
                                      "region_signature": signature,
                                      "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
                                      "response": response})
                    count_sent += 1
    print(f"Done: {count_sent} OCR requests, {count_skipped} cache hits. ")
    if args.engine == "tesseract":
        print("Tesseract creates plain text only. For commune tables, Mistral HTML is needed for build.")


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


def active_ocr_regions(years: list[int] | None) -> dict[tuple[int, int, str], str]:
    """Return permitted current regions, excluding superseded crops from build.

    This prevents stale 'full' OCR from being mixed with replacement 'full_col1/2'
    results after changing the table_panels configuration.
    """
    pdfs = find_pdfs()
    active = {}
    for year in (years if years is not None else sorted(pdfs)):
        if year not in pdfs:
            raise FileNotFoundError(f"Missing original PDF for {year}: cannot validate OCR cache")
        with fitz.open(pdfs[year]) as pdf:
            start, end = valid_range(year, pdf)
            for page_no in range(start, end + 1):
                for region in regions_for_page(year, page_no, pdf[page_no - 1]):
                    active[(year, page_no, region)] = pdfs[year].name
    return active


def parse_cache(engine: str, years: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active = active_ocr_regions(years)
    mappings = read_json(MAPPINGS, {})
    raw_rows = []
    candidates = []
    inventory = []
    page_audit = []
    directory = DERIVED / "ocr" / engine
    for path in sorted(directory.glob("*/p*_*.json")):
        record = read_json(path, {})
        year, page_no, side = record["year"], record["pdf_page"], record["side"]
        # A previous OCR cache might be for an earlier crop/layout or an
        # excluded department. Keep it on disk but never use it in build.
        if active.get((year, page_no, side)) != record.get("source_pdf"):
            continue
        # Changing a table-panel split without rerunning OCR must not silently
        # incorporate obsolete content in a research-ready candidate dataset.
        if record.get("region_signature") != region_signature(year, page_no, side):
            print(f"Skipping stale OCR crop {path}; rerun OCR for this page", file=sys.stderr)
            continue
        image_path = ROOT / str(record.get("image", ""))
        if not image_path.is_file() or hashlib.sha256(image_path.read_bytes()).hexdigest() != record.get("image_sha256"):
            print(f"Skipping changed/missing OCR image {path}; rerun OCR", file=sys.stderr)
            continue
        response = record.get("response", {})
        ocr_page = (response.get("pages") or [{}])[0]
        sources = table_sources(ocr_page)
        page_context = str(ocr_page.get("markdown", ""))
        from territorial_validation import markdown_table_prefaces
        prefaces = markdown_table_prefaces(page_context)
        page_audit.append({"year": year, "pdf_page": page_no, "side": side,
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
            layout = layout_for(year, width, mappings, page_no, side)
            cols = layout.get("columns", {}) if layout else {}
            inventoried = {"year": year, "pdf_page": page_no, "side": side,
                           "table_id": table_id, "column_count": width,
                           "layout": layout.get("name") if layout else None,
                           "n_rows": len(rows), "html_header_excerpt": " | ".join(rows[0])[:350],
                           "image": record["image"]}
            inventory.append(inventoried)
            for i, cells in enumerate(rows, start=1):
                typ = row_type(cells)
                ident = f"{year}-p{page_no:04d}-{side}-{table_id}-r{i:03d}"
                base = {"row_id": ident, "year": year, "pdf_page": page_no,
                        "side": side, "table_id": table_id, "table_row": i,
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
    raw, communes, inventory, page_audit = parse_cache(args.engine, args.years)
    output = DERIVED / "output"
    if not page_audit.empty:
        export_frame(page_audit, output / "situation_financiere_page_audit.parquet")
    if raw.empty:
        raise RuntimeError(f"No structured HTML tables found under {DERIVED / 'ocr' / args.engine}. "
                           "Check OCR JSON responses; plain-text Tesseract is not enough.")
    export_frame(raw, output / "situation_financiere_all_rows.parquet")
    export_frame(inventory, output / "situation_financiere_column_inventory.parquet")
    from territorial_validation import canton_subtotal_checks
    expected = set(active_ocr_regions(args.years))
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
    for cmd in ("discover", "preview", "plan", "ocr", "build"):
        q = sub.add_parser(cmd)
        q.add_argument("--years", type=int, nargs="+", choices=YEARS,
                       help="Limit to specified years (default: all PDFs present)")
        if cmd == "discover":
            q.add_argument("--header-ocr", action="store_true",
                           help="Slow: Tesseract only page headers, useful for scanned PDFs")
        if cmd == "preview":
            q.add_argument("--center", type=int, help="PDF page for a local contact sheet")
            q.add_argument("--radius", type=int, default=5)
            q.add_argument("--every", type=int, default=1, help="Use 2, 3, etc. for long ranges")
            q.add_argument("--regions", action="store_true",
                           help="Preview full/left/right images as they would be sent to OCR")
        if cmd == "ocr":
            q.add_argument("--engine", choices=("mistral", "tesseract"), default="mistral")
            q.add_argument("--model", default=MODEL, help="Pinned Mistral model ID")
            q.add_argument("--max-pages", type=int, default=0,
                           help="Pilot: max pages per year, starting at confirmed range start")
            q.add_argument("--pages", type=int, nargs="+",
                           help="Instead, OCR selected 1-based PDF page numbers within each confirmed range")
            q.add_argument("--yes-all", action="store_true", help="Required for full range OCR")
            q.add_argument("--dpi", type=int, default=210)
            q.add_argument("--force", action="store_true", help="Replace cached responses")
        if cmd == "build":
            q.add_argument("--engine", choices=("mistral", "tesseract"), default="mistral")
            q.add_argument("--no-match", action="store_true", help="Skip communematch temporarily")
            q.add_argument("--min-confidence", type=float, default=0.75)
    args = p.parse_args()
    if args.command == "discover":
        discover(args)
    elif args.command == "preview":
        preview(args)
    elif args.command == "plan":
        plan(args)
    elif args.command == "ocr":
        if args.max_pages < 0:
            p.error("--max-pages must be >= 0")
        ocr(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
