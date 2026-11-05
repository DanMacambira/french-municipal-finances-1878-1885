#!/usr/bin/env python3
"""Offline visual layout diagnostics for historical municipal-finance tables.

No page-specific coordinates, no OCR requests, no changes to existing OCR data.
This is a geometry *proposal* generator: inspect its overlays before using any
bounds to crop pages or associate extraordinary-tax entries.

Usage:
  python code/automatic_layout.py --images scan1.png scan2.png --output derived/layout_diagnostics
  python code/automatic_layout.py --pdf raw_data/.../volume.pdf --pages 245,246 --output derived/layout_diagnostics
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Rule:
    x: int
    continuity: float


def _binary(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    # Otsu chooses a threshold from each scan, accommodating faded/uneven prints.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def vertical_rules(image: Image.Image) -> list[Rule]:
    """Long vertical ruling lines, independent of pixel dimensions."""
    b = _binary(image)
    h, w = b.shape
    length = max(16, round(h * 0.055))
    mask = cv2.morphologyEx(
        b, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)),
    )
    y0, y1 = round(h * .26), round(h * .90)
    support = (mask[y0:y1] > 0).mean(axis=0)
    occupied = np.flatnonzero(support >= .08)
    groups: list[list[int]] = []
    for x in occupied:
        if groups and x - groups[-1][-1] <= max(2, round(w * .003)):
            groups[-1].append(int(x))
        else:
            groups.append([int(x)])
    return [Rule(x=round(float(np.mean(g))), continuity=round(float(np.max(support[g])), 4))
            for g in groups]


def table_split(image: Image.Image, rules: list[Rule]) -> dict:
    """Proposal for the separator between two side-by-side tables."""
    w, h = image.size
    if w / h < .43:
        return {"status": "not_applicable", "x": None, "reason": "narrow image, not a full printed page"}
    # Search centered on the printed page, not a fixed midpoint position.
    plausible = [r for r in rules if .30 * w < r.x < .68 * w and r.continuity >= .12]
    if not plausible:
        return {"status": "unresolved", "x": None, "reason": "no central long ruling line"}
    best = min(plausible, key=lambda r: abs(r.x / w - .5) - .05 * r.continuity)
    return {"status": "candidate", "x": best.x,
            "relative_x": round(best.x / w, 5), "rule_continuity": best.continuity,
            "reason": "central long ruling line; not yet verified"}


def printed_page_regions(image: Image.Image, rules: list[Rule]) -> dict:
    """Potential two-page spread, guarded by page aspect ratio."""
    w, h = image.size
    if w / h < .96:
        return {"type": "single_printed_page", "split_x": None}
    # Wide volumes can include non-spread pages. This is only a diagnostic.
    b = _binary(image)
    band = b[int(h * .16):int(h * .88)]
    x0, x1 = round(.42 * w), round(.58 * w)
    radius = max(5, int(w * .007))
    candidates = range(x0, x1 + 1)
    # Favor gaps over inked text; the best candidate may be slightly off-center.
    scores = []
    for x in candidates:
        darkness = (band[:, max(0, x - radius):min(w, x + radius)] > 0).mean()
        scores.append((darkness + .12 * abs(x / w - .5), x))
    score, x = min(scores)
    # Do not silently split if there is no real whitespace evidence.
    status = "candidate" if score < .12 else "needs_review"
    return {"type": "possible_spread", "split_x": int(x),
            "relative_x": round(x / w, 5), "status": status,
            "ink_score": round(float(score), 5)}


def brace_proposals(image: Image.Image, *, tax_lane: tuple[float, float] | None = None) -> list[dict]:
    """Detect tall, narrow connected ink components (possible brace strokes).

    Deliberately returns component *proposals*, not commune assignments. A
    tax_lane may be inferred from column semantics in a future pipeline; when
    unknown the last 45% of the table is searched without fixed coordinates.
    """
    b = _binary(image)
    h, w = b.shape
    if tax_lane is None:
        left, right = .55, .97
    else:
        left, right = tax_lane
        if not 0 <= left < right <= 1:
            raise ValueError("invalid relative tax_lane")
    x0, x1 = round(w * left), round(w * right)
    # Avoid table borders: their continuous ink is not an accolade.
    long_rules = [r.x for r in vertical_rules(image) if r.continuity > .15]
    closed = cv2.morphologyEx(b, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4)))
    _, _, components, _ = cv2.connectedComponentsWithStats(closed, 8)
    candidates = []
    min_h, max_h = max(22, int(.009 * h)), int(.12 * h)
    max_w = max(8, int(.025 * w))
    for x, y, cw, ch, area in components[1:]:
        x, y, cw, ch, area = map(int, (x, y, cw, ch, area))
        xc = x + cw / 2
        if not (x0 <= xc < x1 and .22 * h <= y < .96 * h):
            continue
        if not (2 <= cw <= max_w and min_h <= ch <= max_h and ch / cw >= 3.5):
            continue
        if any(abs(xc - rx) < max(3, int(w * .003)) for rx in long_rules):
            continue
        if area < ch * .9:
            continue
        candidates.append({"x_min": x, "y_min": y, "x_max": x + cw,
                           "y_max": y + ch, "height": ch, "width": cw})
    # Curved braces differ from the straight strokes used for one-entry
    # communes. Preserve both kinds as evidence, but NEVER treat a straight
    # stroke as a tax-group boundary. Curvature is only a proposal: scan noise
    # and broken braces can look similar.
    for candidate in candidates:
        x, y = candidate["x_min"], candidate["y_min"]
        cw, ch = candidate["width"], candidate["height"]
        patch = closed[y:y + ch, x:x + cw] > 0
        rows, columns = np.where(patch)
        if len(columns) == 0:
            candidate["kind"] = "unknown_stroke"
            continue
        tertile = np.minimum(2, (rows * 3 // max(1, ch)).astype(int))
        medians = [float(np.median(columns[tertile == i])) if np.any(tertile == i)
                   else float(np.median(columns)) for i in range(3)]
        total_spread = float(np.percentile(columns, 95) - np.percentile(columns, 5))
        mid_bend = abs(medians[1] - (medians[0] + medians[2]) / 2)
        drift = abs(medians[2] - medians[0])
        candidate["curvature_px"] = round(max(mid_bend, drift), 2)
        candidate["horizontal_span_px"] = round(total_spread, 2)
        candidate["kind"] = ("curly_brace_candidate" if cw >= 5 and total_spread >= 2
                             and max(mid_bend, drift) >= .9
                             else "straight_or_fragment_candidate")

    # Repetition at the same x is the strongest cheap cue for a printed brace lane.
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: (c["x_min"] + c["x_max"]) / 2)
    lanes: list[list[dict]] = []
    tolerance = max(6, int(w * .016))
    for c in ordered:
        mid = (c["x_min"] + c["x_max"]) / 2
        if lanes and mid - np.median([(d["x_min"] + d["x_max"]) / 2 for d in lanes[-1]]) <= tolerance:
            lanes[-1].append(c)
        else:
            lanes.append([c])
    # Label plausible recurring lanes; leave isolated marks untrusted.
    for lane_idx, lane in enumerate(lanes):
        for c in lane:
            c["lane"] = lane_idx
            c["lane_support"] = len(lane)
            c["status"] = ("curved_shape_candidate" if c["kind"] == "curly_brace_candidate"
                           and len(lane) >= 3 else "unverified_stroke")
    return sorted([c for lane in lanes for c in lane], key=lambda c: (c["y_min"], c["x_min"]))


def diagnose(image: Image.Image) -> dict:
    rules = vertical_rules(image)
    page_info = printed_page_regions(image, rules)
    splits = []
    if page_info["type"] == "single_printed_page":
        split = table_split(image, rules)
        splits.append(split)
        if split["x"] is not None:
            # Detect braces within BOTH actual table panels, not fixed crops.
            x = split["x"]
            table_regions = [(0, x + 6), (max(0, x - 6), image.width)]
        else:
            table_regions = [(0, image.width)]
    else:
        table_regions = []
        if page_info.get("status") == "candidate":
            x = page_info["split_x"]
            table_regions = [(0, x + 6), (max(0, x - 6), image.width)]
    braces = []
    for i, (left, right) in enumerate(table_regions):
        tile = image.crop((left, 0, right, image.height))
        for b in brace_proposals(tile):
            b["x_min"] += left
            b["x_max"] += left
            b["region_index"] = i
            braces.append(b)
    return {"image_width": image.width, "image_height": image.height,
            "page_structure": page_info, "table_splits": splits,
            "vertical_rules": [asdict(r) for r in rules],
            "brace_proposals": braces,
            "status": "diagnostic_only_not_validated"}


def overlay(image: Image.Image, result: dict) -> Image.Image:
    out = image.convert("RGB")
    pen = ImageDraw.Draw(out)
    h = image.height
    for r in result["vertical_rules"]:
        pen.line([(r["x"], int(h*.25)), (r["x"], int(h*.9))], fill=(80, 155, 245), width=1)
    for s in result["table_splits"]:
        if s.get("x") is not None:
            pen.line([(s["x"], 0), (s["x"], h)], fill=(0, 195, 0), width=4)
    split_x = result["page_structure"].get("split_x")
    if split_x is not None:
        pen.line([(split_x, 0), (split_x, h)], fill=(20, 195, 20), width=4)
    for b in result["brace_proposals"]:
        color = (255, 100, 45) if b["status"] == "curved_shape_candidate" else (240, 200, 30)
        pen.rectangle((b["x_min"]-2, b["y_min"]-2,
                       b["x_max"]+2, b["y_max"]+2), outline=color, width=2)
    return out


def read_pdf_pages(path: Path, page_numbers: list[int], dpi: int):
    import fitz
    with fitz.open(str(path)) as pdf:
        for page_no in page_numbers:
            if not 1 <= page_no <= len(pdf):
                raise ValueError(f"page {page_no} out of 1..{len(pdf)}")
            pix = pdf[page_no - 1].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            yield f"{path.stem}_p{page_no:04d}", im


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", nargs="+", type=Path)
    source.add_argument("--pdf", type=Path)
    ap.add_argument("--pages", help="1-indexed PDF page numbers, comma-separated")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--output", type=Path, default=Path("derived/situation_financiere/automatic_layout"))
    opts = ap.parse_args()
    if opts.pdf:
        if not opts.pages:
            ap.error("--pdf needs --pages; this diagnostic never processes a full volume implicitly")
        inputs = list(read_pdf_pages(opts.pdf, [int(x) for x in opts.pages.split(",")], opts.dpi))
    else:
        inputs = []
        for path in opts.images:
            with Image.open(path) as image:
                inputs.append((path.stem, image.convert("RGB")))
    opts.output.mkdir(parents=True, exist_ok=True)
    report = []
    for identifier, image in inputs:
        result = diagnose(image)
        (opts.output / f"{identifier}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        overlay(image, result).save(opts.output / f"{identifier}_overlay.png")
        report.append({"image": identifier, "page": result["page_structure"],
                       "table_splits": result["table_splits"],
                       "brace_proposals": len(result["brace_proposals"])})
    (opts.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("OFFLINE, DIAGNOSTIC ONLY. Do not use the boundaries or braces as verified data.")


if __name__ == "__main__":
    main()
