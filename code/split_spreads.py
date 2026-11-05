#!/usr/bin/env python3
"""Conservatively split two-page PDF scan spreads along their dark central gutter.

This stage preserves single-page scans intact and splits only configured
two-page spreads. It never edits source PDFs or calls paid services. Uncertain
gutter detection is flagged, and no such split is produced by default.

Run from the repository root:
    .venv/bin/python code/split_spreads.py --year 1882 --pages 245 247 249
"""
from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentation_runtime import Progress, contact_sheets, summary_spreads

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "schema" / "situation_financiere" / "page_ranges.json"
DEFAULT_INPUT = ROOT / "raw_data" / "situation_financiere_communes"
DEFAULT_OUTPUT = ROOT / "derived" / "situation_financiere" / "split_pages"
VERSION = "multiprocess_stage1_v1_6"


@dataclass(frozen=True)
class GutterResult:
    x_ratio: float
    peak_darkness: float
    background_darkness: float
    prominence: float
    accepted: bool
    reason: str


def _box_smooth(values: np.ndarray, n: int) -> np.ndarray:
    """Mean filter with edge padding (unlike np.convolve(..., 'same'))."""
    n = max(3, int(n) | 1)
    pad = n // 2
    padded = np.pad(values.astype(np.float64), (pad, pad), mode="edge")
    cs = np.concatenate(([0.0], np.cumsum(padded)))
    return (cs[n:] - cs[:-n]) / n


def detect_gutter(image: Image.Image, *, center_min: float = .45,
                  center_max: float = .55) -> GutterResult:
    """Find a broad dark central strip, not an isolated printed table rule.

    Scale-independent scoring: near-black pixel fraction across many scan rows,
    then smooth horizontally at ~1% scan width to suppress narrow vertical rules.
    An indistinct maximum is refused and must be reviewed.
    """
    if not (0.0 < center_min < .5 < center_max < 1.0):
        raise ValueError("Search interval must straddle the middle")
    original_width = image.width
    if original_width < 100 or image.height < 100:
        raise ValueError("Image is too small to analyze")
    target_width = min(1400, original_width)
    target_height = max(200, round(image.height * target_width / original_width))
    small = image.convert("L").resize((target_width, target_height), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.uint8)
    # Ignore decorations/black printer borders at extreme top and bottom.
    top = int(target_height * .065)
    bottom = int(target_height * .94)
    if bottom - top < 150:
        raise ValueError("Not enough image height")
    region = arr[top:bottom]
    # A broad black crease survives horizontal smoothing; thin table rules do not.
    darkness = (region < 105).mean(axis=0)
    win = max(7, int(round(target_width * .010)) | 1)
    score = _box_smooth(darkness, win)
    lo = max(0, round(target_width * center_min))
    hi = min(target_width, round(target_width * center_max) + 1)
    candidates = score[lo:hi]
    peak_local = int(np.argmax(candidates))
    x = lo + peak_local
    peak = float(score[x])
    # Background median within the admissible band. The true crease should stand
    # out from the many thin printed vertical column rules.
    background = float(np.median(candidates))
    prominence = peak - background
    # Take the weighted centre of the dark peak plateau to avoid cutting on one
    # edge of a broad binding shadow.
    threshold = background + .68 * prominence
    inds = np.where(candidates >= threshold)[0]
    if inds.size:
        # Only use the contiguous plateau containing the best column.
        pieces = np.split(inds, np.where(np.diff(inds) > 1)[0] + 1)
        plateau = next((piece for piece in pieces if peak_local in piece), None)
        if plateau is not None:
            xs = plateau + lo
            weights = np.maximum(score[xs] - background, 0)
            if weights.sum() > 0:
                x = int(round(float(np.average(xs, weights=weights))))
    ratio = x / target_width
    # Require the darkness to extend through many rows, not just one fine rule.
    strong = peak >= .36 and prominence >= .16
    # If the maximum is right at a boundary, the actual crease might fall outside
    # our promised near-equal region; don't force a guess.
    edge = (x - lo) <= win // 2 or (hi - 1 - x) <= win // 2
    accepted = bool(strong and not edge)
    if edge:
        reason = "peak_at_search_boundary"
    elif not strong:
        reason = "weak_or_ambiguous_dark_gutter"
    else:
        reason = "dark_central_gutter"
    return GutterResult(float(ratio), peak, background, prominence, accepted, reason)


def split_image(image: Image.Image, x_ratio: float, overlap_fraction: float = 0.,
                inner_trim_fraction: float = 0.) -> tuple[Image.Image, Image.Image, int]:
    """Split a spread, optionally removing gutter residue from BOTH inner edges.

    The new OCR-safe default is to trim the dark gutter (a small gap) instead
    of duplicating neighbouring page content (an overlap). The old overlap
    option remains available explicitly for comparisons, but cannot coexist
    with trimming.
    """
    if not (0. <= overlap_fraction <= .04):
        raise ValueError("overlap_fraction must be between 0 and .04")
    if not (0. <= inner_trim_fraction <= .015):
        raise ValueError("inner_trim_fraction must be between 0 and .015")
    if overlap_fraction > 0 and inner_trim_fraction > 0:
        raise ValueError("Choose either inner trim or overlap, not both")
    if not (.4 <= x_ratio <= .6):
        raise ValueError("Unsafe split position")
    w, h = image.size
    x = round(w * x_ratio)
    pad = round(w * overlap_fraction)
    trim = round(w * inner_trim_fraction)
    left_end = min(w, x + pad - trim)
    right_start = max(0, x - pad + trim)
    if left_end <= .35 * w or right_start >= .65 * w:
        raise ValueError("Unsafe trimmed spread crop")
    return image.crop((0, 0, left_end, h)), image.crop((right_start, 0, w, h)), x


def _find_pdf(folder: Path, year: int, stated_name: str | None) -> Path:
    if stated_name:
        path = folder / stated_name
        if path.is_file():
            return path
    matches = [p for p in folder.glob("*.pdf") if re.search(rf"(?<!\d){year}(?!\d)", p.stem)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one PDF for {year} in {folder}; found {len(matches)}")
    return matches[0]


def _render_pdf_page(page: pymupdf.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72),
                          colorspace=pymupdf.csGRAY, alpha=False)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)


def _preview(image: Image.Image, x: int, min_ratio: float, max_ratio: float,
             result: GutterResult, label: str, inner_trim_fraction: float = 0.) -> Image.Image:
    visual = image.convert("RGB")
    visual.thumbnail((1300, 1050), Image.Resampling.LANCZOS)
    sx = visual.width / image.width
    line = round(x * sx)
    draw = ImageDraw.Draw(visual)
    if inner_trim_fraction:
        trim = round(visual.width * inner_trim_fraction)
        draw.rectangle((max(0, line-trim), 31, min(visual.width-1, line+trim),
                        visual.height-1), fill=(252, 215, 230))
    draw.line([(line, 31), (line, visual.height - 1)], fill=(255, 0, 180), width=3)
    for r in (min_ratio, max_ratio):
        xx = round(visual.width * r)
        for yy in range(0, visual.height, 14):
            draw.line([(xx, yy), (xx, min(visual.height, yy + 6))], fill=(0, 170, 230), width=2)
    draw.rectangle((0, 0, visual.width, 30), fill=(24, 44, 55))
    draw.text((8, 8), f"{label}: cut at {result.x_ratio:.2%}; "
              f"{'ACCEPTED' if result.accepted else 'REVIEW'}; {result.reason}", fill="white")
    return visual


def _single_preview(image: Image.Image, title: str) -> Image.Image:
    """A visibly labelled passthrough preview (no split line)."""
    view = image.convert("RGB")
    view.thumbnail((1300, 1050), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(view)
    draw.rectangle((0, 0, view.width, 30), fill=(24, 44, 55))
    draw.text((8, 8), f"{title}: SINGLE PAGE (kept intact)", fill="white")
    return view


def _contact_sheet(paths: list[Path], dest: Path) -> None:
    if not paths:
        return
    width, height = 480, 390
    canvas = Image.new("RGB", (width * 3, height * ((len(paths) + 2) // 3)), "white")
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            image = im.convert("RGB")
        image.thumbnail((width - 14, height - 25))
        x = i % 3 * width + 7
        y = i // 3 * height + 20
        canvas.paste(image, (x, y))
        ImageDraw.Draw(canvas).text((x, y - 16), path.stem, fill="black")
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, quality=85)


_PDF_WORKER_DOC = None


def _init_pdf_worker(pdf_path: str) -> None:
    """Each process owns its own PyMuPDF document; never share a document across workers."""
    global _PDF_WORKER_DOC
    _PDF_WORKER_DOC = pymupdf.open(pdf_path)


def _run_spread_page_task(task: tuple) -> dict:
    (year, pdf_page, dpi, spec, out, center_min, center_max,
     overlap_fraction, make_pdfs, force_review, inner_trim_fraction) = task
    if _PDF_WORKER_DOC is None:
        raise RuntimeError('PDF worker has not been initialized')
    page_id = f'p{pdf_page:04d}'
    accepted_dir, review_dir = out / 'accepted', out / 'needs_review'
    preview_dir = out / 'previews'
    for directory in (accepted_dir, review_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)
    # Remove the processed page's old outputs, including v1.5 root-level
    # images. A later failed detection must never leave a stale OCR-ready PNG.
    for folder in (out, accepted_dir, review_dir):
        for old in folder.glob(f'{page_id}_*'):
            if old.is_file() and old.suffix.lower() in {'.png', '.pdf', '.jpg'}:
                old.unlink()
    for old in preview_dir.glob(f'{page_id}_*.jpg'):
        old.unlink()
    layout = spec.get('scan_layout_overrides', {}).get(str(pdf_page),
                   spec.get('scan_layout', 'auto'))
    pg = _PDF_WORKER_DOC[pdf_page - 1]
    if layout == 'auto':
        is_spread = pg.rect.width > pg.rect.height * 1.05
    elif layout in {'spread', 'single'}:
        is_spread = layout == 'spread'
    else:
        raise ValueError(f'Invalid scan_layout {layout!r}')
    page_sides = spec.get('page_sides', {})
    if not isinstance(page_sides, dict):
        raise ValueError('page_sides must be a mapping')
    sides = page_sides.get(str(pdf_page), ['left', 'right'])
    if (not isinstance(sides, list) or not sides or len(sides) != len(set(sides))
            or any(side not in {'left', 'right'} for side in sides)):
        raise ValueError(f'Invalid page_sides at {pdf_page}: {sides}')
    if not is_spread and str(pdf_page) in page_sides:
        raise ValueError(f'page_sides is only valid for spreads: {pdf_page}')
    image = _render_pdf_page(pg, dpi)
    if not is_spread:
        dest = accepted_dir / f'{page_id}_single_{dpi}dpi.png'
        image.save(dest)
        if make_pdfs:
            image.convert('RGB').save(dest.with_suffix('.pdf'), resolution=dpi)
        preview = preview_dir / f'{page_id}_single.jpg'
        _single_preview(image, f'{year} PDF p.{pdf_page}').save(preview, quality=90)
        return {'year': year, 'pdf_page': pdf_page, 'scan_layout': 'single',
                'status': 'accepted_single', 'reason': 'configured_single_page_pass_through',
                'width_px': image.width, 'single_image': str(dest),
                'preview': str(preview), 'algorithm': VERSION}
    result = detect_gutter(image, center_min=center_min, center_max=center_max)
    x = round(result.x_ratio * image.width)
    good = result.accepted
    preview = ((preview_dir if good else review_dir) / f'{page_id}_cut.jpg')
    _preview(image, x, center_min, center_max, result, f'{year} PDF p.{pdf_page}',
             inner_trim_fraction=inner_trim_fraction).save(preview, quality=90)
    left_path = right_path = ''
    status = 'accepted' if good else 'needs_review'
    if good or force_review:
        left, right, _ = split_image(image, result.x_ratio, overlap_fraction,
                                    inner_trim_fraction=inner_trim_fraction)
        destination = accepted_dir if good else review_dir
        if 'left' in sides:
            target = destination / f'{page_id}_left_{dpi}dpi.png'
            left.save(target)
            left_path = str(target)
            if make_pdfs and good:
                left.convert('RGB').save(target.with_suffix('.pdf'), resolution=dpi)
        if 'right' in sides:
            target = destination / f'{page_id}_right_{dpi}dpi.png'
            right.save(target)
            right_path = str(target)
            if make_pdfs and good:
                right.convert('RGB').save(target.with_suffix('.pdf'), resolution=dpi)
        if not good:
            status = 'provisional_review_only'
    if not good:
        # Full original scan always available for inspecting a failed gutter.
        image.save(review_dir / f'{page_id}_source_{dpi}dpi.png')
    return {'year': year, 'pdf_page': pdf_page, 'scan_layout': 'spread',
            'status': status, 'reason': result.reason,
            'split_fraction': round(result.x_ratio, 6), 'split_x_px': x,
            'width_px': image.width, 'inner_trim_fraction': inner_trim_fraction,
            'overlap_fraction': overlap_fraction,
            'left_end_px': x + round(image.width * overlap_fraction)
                           - round(image.width * inner_trim_fraction),
            'right_start_px': x - round(image.width * overlap_fraction)
                              + round(image.width * inner_trim_fraction),
            'darkness_score': round(result.peak_darkness, 4),
            'background_score': round(result.background_darkness, 4),
            'prominence': round(result.prominence, 4),
            'left_image': left_path, 'right_image': right_path,
            'preview': str(preview), 'algorithm': VERSION}


def process_year(*, year: int, requested_pages: list[int] | None, dpi: int,
                 config: Path, input_dir: Path, output_dir: Path, center_min: float,
                 center_max: float, overlap_fraction: float, make_pdfs: bool,
                 force_review: bool, inner_trim_fraction: float = .001,
                 workers: int = 1) -> list[dict[str, str | int | float]]:
    if overlap_fraction and inner_trim_fraction:
        raise ValueError('overlap and inner trim cannot both be nonzero')
    if workers < 1:
        raise ValueError('workers must be >= 1')
    settings = json.loads(config.read_text(encoding='utf-8'))
    spec = settings.get(str(year), {})
    if not spec.get('confirmed'):
        raise ValueError(f'{year}: no confirmed page range in {config}')
    pdf_file = _find_pdf(input_dir, year, spec.get('pdf'))
    out = output_dir / str(year)
    out.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf_file) as doc:
        start, end = int(spec['start']), int(spec['end'])
        if not 1 <= start <= end <= len(doc):
            raise ValueError(f'Invalid page range: {start}..{end}, PDF has {len(doc)} pages')
    page_ids = (sorted(set(requested_pages)) if requested_pages
                else list(range(start, end + 1)))
    if any(p < start or p > end for p in page_ids):
        raise ValueError(f'Requested pages must be inside confirmed range {start}-{end}')
    tasks = [(year, p, dpi, spec, out, center_min, center_max, overlap_fraction,
              make_pdfs, force_review, inner_trim_fraction) for p in page_ids]
    rows: list[dict] = []
    progress = Progress(f'{year} | Stage 1', len(tasks))
    if workers == 1:
        _init_pdf_worker(str(pdf_file))
        try:
            for task in tasks:
                rows.append(_run_spread_page_task(task))
                progress.update()
        finally:
            global _PDF_WORKER_DOC
            if _PDF_WORKER_DOC is not None:
                _PDF_WORKER_DOC.close()
                _PDF_WORKER_DOC = None
    else:
        ctx = multiprocessing.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                  mp_context=ctx, initializer=_init_pdf_worker,
                  initargs=(str(pdf_file),)) as pool:
            futures = [pool.submit(_run_spread_page_task, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
                progress.update()
    fields = ['year', 'pdf_page', 'scan_layout', 'status', 'reason',
              'split_fraction', 'split_x_px', 'width_px', 'inner_trim_fraction',
              'overlap_fraction', 'left_end_px', 'right_start_px',
              'darkness_score', 'background_score', 'prominence', 'left_image',
              'right_image', 'single_image', 'preview', 'algorithm']
    report = out / 'split_review.csv'
    if requested_pages is not None and report.is_file():
        current = {int(r['pdf_page']) for r in rows}
        with report.open(newline='', encoding='utf-8') as handle:
            for previous in csv.DictReader(handle):
                if int(previous['pdf_page']) not in current:
                    rows.append(previous)
    rows.sort(key=lambda r: int(r['pdf_page']))
    with report.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    all_previews = [Path(r['preview']) for r in rows if r.get('preview')]
    review_previews = [Path(r['preview']) for r in rows
                       if r.get('status') not in {'accepted', 'accepted_single'} and r.get('preview')]
    contact_sheets(all_previews, out, name='review_contact_sheet', tile=(480,390))
    contact_sheets(review_previews, out / 'needs_review',
                   name='review_contact_sheet', tile=(480,390))
    summary_spreads(rows, progress.elapsed)
    print(f'Review CSV: {report}')
    print(f'Accepted PNGs: {out / "accepted"}')
    print(f'Needs-review PNGs/previews: {out / "needs_review"}')
    print('No OCR was performed.')
    return rows



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, help="e.g., 1882")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="parallel worker processes (default: up to 4, use 1 for serial)")
    parser.add_argument("--pages", nargs="+", type=int, help="1-based PDF page numbers; default: all confirmed pages")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--center-min", type=float, default=.45)
    parser.add_argument("--center-max", type=float, default=.55)
    parser.add_argument("--inner-trim", type=float, default=.001,
                        help="remove this fraction of full spread width from EACH inner page edge (default .001)")
    parser.add_argument("--overlap", type=float, default=0.,
                        help="legacy shared margin; requires --inner-trim 0")
    parser.add_argument("--pdf", action="store_true", help="also create one-page PDF files from the PNGs")
    parser.add_argument("--force-review", action="store_true", help="write provisional halves for uncertain gutters; not recommended for OCR")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 72 <= args.dpi <= 400:
        parser.error("--dpi must be between 72 and 400")
    if args.overlap and args.inner_trim:
        parser.error("--overlap requires --inner-trim 0")
    process_year(year=args.year, requested_pages=args.pages, dpi=args.dpi,
                 config=args.config, input_dir=args.input_dir, output_dir=args.output_dir,
                 center_min=args.center_min, center_max=args.center_max,
                 overlap_fraction=args.overlap, make_pdfs=args.pdf,
                 force_review=args.force_review, inner_trim_fraction=args.inner_trim,
                 workers=args.workers)


if __name__ == "__main__":
    main()
