#!/usr/bin/env python3
"""Split each previously extracted printed-page scan into its two parallel tables.

Rule-based image recognition: detect the long central double border followed by
an almost rule-free wide NOMS column. No OCR, model, templates, or API calls.

From the repository root:
  .venv/bin/python code/split_tables.py --year 1882 --pages 245 247 249
  .venv/bin/python code/split_tables.py --year 1882
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "derived" / "situation_financiere" / "split_pages"
DEFAULT_OUTPUT = ROOT / "derived" / "situation_financiere" / "table_panels"
VERSION = "vertical_rule_to_names_adaptive_v1_1"
PATTERN = re.compile(r"^p(?P<page>\d{4})_(?P<side>left|right)_(?P<dpi>\d+)dpi\.png$")


@dataclass(frozen=True)
class Separator:
    fraction: float
    strength: float
    left_header_rules: float
    right_header_rules: float
    second_strength: float
    confidence: float
    drift_fraction: float
    accepted: bool
    reason: str


def _resize(gray: np.ndarray, target_width: int = 1000) -> np.ndarray:
    h, w = gray.shape
    if w <= target_width:
        return gray
    return cv2.resize(gray, (target_width, round(h * target_width / w)), interpolation=cv2.INTER_AREA)


def _rule_profiles(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Long vertical ink strokes; reject printed characters and tiny speckles."""
    h, w = gray.shape
    if min(h, w) < 150:
        raise ValueError("Image too small for two-table detection")
    binary = np.uint8(gray < 145) * 255
    height = max(17, round(h * .035))
    # Open only in the vertical direction: letters/ink specks disappear while
    # long table rules remain. No absolute crop coordinates or page template.
    rules = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, height)))
    upper, lower = round(h * .07), round(h * .91)
    body = (rules[upper:lower] > 0).mean(axis=0).astype(np.float32)
    body = cv2.GaussianBlur(body.reshape(1, -1), (0, 0), sigmaX=1.7)[0]
    # Printed column rules run through the top of each table, but the NOMS
    # column is unusually wide and has no internal *long vertical* lines.
    header = (rules[round(h * .085):round(h * .19)] > 0).mean(axis=0).astype(np.float32)
    upper_body = (rules[round(h * .07):round(h * .32)] > 0).mean(axis=0).astype(np.float32)
    lower_body = (rules[round(h * .32):round(h * .91)] > 0).mean(axis=0).astype(np.float32)
    return body, header, np.minimum(upper_body, lower_body), rules


def _candidate_peaks(score: np.ndarray, first: int, last: int) -> list[int]:
    """Rank locally strong vertical lines, excluding near-duplicates."""
    raw: list[int] = []
    for x in range(max(first + 2, 2), min(last - 2, len(score) - 2)):
        if score[x] >= max(score[x - 2:x + 3]) and score[x] >= .035:
            raw.append(x)
    ranked = sorted(raw, key=lambda x: float(score[x]), reverse=True)
    result: list[int] = []
    exclusion = max(6, round(len(score) * .010))
    for x in ranked:
        if all(abs(x - other) > exclusion for other in result):
            result.append(x)
    return result[:18]


def _line_drift(rules: np.ndarray, x: int) -> float:
    """Estimate vertical-border curvature from four distant parts of the scan."""
    h, w = rules.shape
    xs: list[int] = []
    band_width = max(7, round(w * .023))
    for start, stop in ((.08, .22), (.26, .45), (.48, .68), (.71, .90)):
        region = (rules[round(h * start):round(h * stop)] > 0).mean(axis=0).astype(np.float32)
        smooth = cv2.GaussianBlur(region.reshape(1, -1), (0, 0), sigmaX=1.7)[0]
        left = max(0, x - band_width)
        right = min(w, x + band_width + 1)
        peak_x = left + int(np.argmax(smooth[left:right]))
        if smooth[peak_x] >= .085:
            xs.append(peak_x)
    if len(xs) < 3:
        return float('inf')  # Cannot confirm that the boundary is stable.
    return (max(xs) - min(xs)) / w


def detect_separator(image: Image.Image, *, min_fraction: float = .42,
                     max_fraction: float = .58) -> Separator:
    """Detect table1 -> table2 boundary; reject weak or non-unique candidates.

    Does not attempt to straighten curved columns. The selected x is the
    characteristic double vertical border, near but not necessarily at 50%.
    """
    if not (.35 <= min_fraction < .5 < max_fraction <= .65):
        raise ValueError("Search band must straddle midpoint and remain central")
    arr = np.asarray(image.convert("L"), dtype=np.uint8)
    arr = _resize(arr)
    h, w = arr.shape
    score, header, continuity, rules = _rule_profiles(arr)
    lo, hi = round(w * min_fraction), round(w * max_fraction)
    candidates = _candidate_peaks(score, lo, hi)
    margin = max(2, round(w * .013))
    radius = max(5, round(w * .09))
    entries: list[dict[str, float]] = []
    for x in candidates:
        # Count long header lines within a symmetric band on either side.
        # A central separator has several narrow fiscal-column rules BEFORE
        # it, and a wide rule-free commune-name area AFTER it.
        before = float(header[max(0, x - radius):max(0, x - margin)].sum())
        after = float(header[min(w, x + margin):min(w, x + radius)].sum())
        persist = float(np.max(continuity[max(0, x - 3):min(w, x + 4)]))
        line = float(score[x])
        if before < .30 or after > before * .55 or before - after < .28:
            continue
        if line < .075 or persist < .025:
            continue
        # The contrast provides a secondary discriminator, rather than being
        # a substitute for the long, continuous central dividing line.
        value = line * (1 + min(2.0, (before - after) / max(before, .01)))
        entries.append({"x": float(x), "strength": line, "left": before,
                        "right": after, "value": value})
    if not entries:
        return Separator(.5, 0., 0., 0., 0., 0., float("inf"), False,
                         "no_distinct_vertical_border_followed_by_wide_names_column")
    entries.sort(key=lambda e: e["value"], reverse=True)
    top = entries[0]
    second = entries[1]["value"] if len(entries) > 1 else 0.0
    dominance = top["value"] / max(second, .0001)
    # When two convincing positions compete, the page needs manual review.
    drift = _line_drift(rules, round(top["x"]))
    accepted = bool(dominance >= 1.18 and drift <= .017)
    if dominance < 1.18:
        reason = "ambiguous_competing_separators"
    elif drift > .017:
        reason = "curved_or_inconsistent_border_needs_review"
    else:
        reason = "detected_central_separator"
    return Separator(top["x"] / w, top["strength"], top["left"], top["right"],
                     float(second), min(1.0, (dominance - 1.) / .8), drift,
                     accepted, reason)


@dataclass(frozen=True)
class CutPath:
    """x-coordinate as a fraction of width, polynomial in normalized height [-1, 1]."""
    kind: str  # vertical, linear, quadratic
    coefficients: tuple[float, ...]  # highest power first (np.polyval)
    sample_count: int
    residual_fraction: float
    max_shift_fraction: float
    reason: str

    def fractions(self, height: int) -> np.ndarray:
        y = (np.arange(height, dtype=np.float64) + .5) * (2. / height) - 1.
        return np.polyval(self.coefficients, y)


def _fit_robust(ys: np.ndarray, xs: np.ndarray, degree: int
                ) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Discard isolated wrong column detections; fit only with broad support."""
    if len(xs) < max(9, 4 * degree + 2):
        return None
    keep = np.ones(len(xs), dtype=bool)
    for _ in range(3):
        if np.count_nonzero(keep) < 9:
            return None
        coef = np.polyfit(ys[keep], xs[keep], degree)
        err = np.abs(xs - np.polyval(coef, ys))
        # Do not reject tiny pixel-level fluctuations; tolerate 0.35% width.
        updated = err <= .0035
        if np.count_nonzero(updated) < max(9, round(len(xs) * .72)):
            break
        keep = updated
    if np.count_nonzero(keep) < max(9, round(len(xs) * .72)):
        return None
    if np.ptp(ys[keep]) < 1.35:  # Must span 67.5% of page height.
        return None
    coef = np.polyfit(ys[keep], xs[keep], degree)
    residual = float(np.sqrt(np.mean((xs[keep] - np.polyval(coef, ys[keep]))**2)))
    return coef, keep, residual


def detect_cut_path(image: Image.Image, separator: Separator, *, mode: str = "auto") -> CutPath:
    """Trace one printed rule independently down THIS page; reject unstable curves.

    The old vertical cut is the safety baseline. This method NEVER accepts an
    unrelated separator based merely on a good polynomial fit: the starting
    boundary must have passed the original table-header signature test.
    """
    if mode not in {"auto", "vertical"}:
        raise ValueError("mode must be 'auto' or 'vertical'")
    fallback = CutPath("vertical", (separator.fraction,), 0, float('nan'),
                       0., "original_vertical_cut")
    if mode == "vertical" or separator.strength <= .075 or separator.reason in {
        "no_distinct_vertical_border_followed_by_wide_names_column",
        "ambiguous_competing_separators",
    }:
        return fallback
    gray = _resize(np.asarray(image.convert("L"), dtype=np.uint8))
    h, w = gray.shape
    _, _, _, rules = _rule_profiles(gray)
    x0 = separator.fraction * w
    radius = max(6, round(w * .019))
    ys, xs = [], []
    # A local vertical-stroke detector is robust to different text in each row.
    for y_fraction in np.linspace(.09, .91, 15):
        cy = round(y_fraction * h)
        half = max(9, round(h * .030))
        region = rules[max(0, cy-half):min(h, cy+half)]
        density = (region > 0).mean(axis=0).astype(np.float32)
        density = cv2.GaussianBlur(density.reshape(1, -1), (0, 0), sigmaX=1.1)[0]
        lo = max(0, round(x0) - radius)
        hi = min(w, round(x0) + radius + 1)
        xx = np.arange(lo, hi)
        # Prefer the already identified rule if another narrow fiscal rule is
        # temporarily visible nearby; still allow gradual bending by ~2%.
        proximity = np.exp(-.5 * ((xx - x0)/(w * .012))**2)
        best = lo + int(np.argmax(density[lo:hi] * proximity))
        if density[best] >= .09:
            ys.append(2. * y_fraction - 1.)
            xs.append(best / w)
    if len(xs) < 10:
        return CutPath("vertical", fallback.coefficients, len(xs), float('nan'),
                       0., "too_few_consistent_rule_samples")
    yy = np.asarray(ys, dtype=np.float64)
    xx = np.asarray(xs, dtype=np.float64)
    fit_line = _fit_robust(yy, xx, 1)
    fit_curve = _fit_robust(yy, xx, 2)
    # On a strongly bowed page, a line may be a poor model even if a quadratic
    # fits all the observed rule positions. Evaluate both independently.
    candidates = []
    if fit_line is not None:
        coef_l, keep_l, err_l = fit_line
        baseline_l = float(np.sqrt(np.mean((xx[keep_l] - separator.fraction)**2)))
        linear_span = float(np.ptp(np.polyval(coef_l, [-1, 1])))
        if (linear_span >= .003 and err_l <= .0035
                and baseline_l - err_l >= .0007
                and err_l <= .78 * max(baseline_l, 1e-6)):
            candidates.append(("linear", coef_l, int(np.count_nonzero(keep_l)), err_l))
    if fit_curve is not None:
        coef_q, keep_q, err_q = fit_curve
        baseline_q = float(np.sqrt(np.mean((xx[keep_q] - separator.fraction)**2)))
        qs = np.polyval(coef_q, np.linspace(-1, 1, 101))
        curve_span = float(np.ptp(qs))
        if (curve_span >= .003 and err_q <= .0035
                and baseline_q - err_q >= .001
                and err_q <= .70 * max(baseline_q, 1e-6)
                and np.count_nonzero(keep_q) >= 10):
            # Prefer linear unless the higher-degree fit explains clearly
            # more of the observed curvature (with enough independent rows).
            use_quadratic = fit_line is None
            if fit_line is not None:
                _, _, err_l = fit_line
                straight = np.polyval(fit_line[0], np.linspace(-1, 1, 101))
                shape_delta = float(np.max(np.abs(qs - straight)))
                use_quadratic = (err_q <= .75 * err_l and
                                 err_l - err_q >= .0008 and
                                 shape_delta >= .002)
            if use_quadratic:
                candidates.append(("quadratic", coef_q, int(np.count_nonzero(keep_q)), err_q))
    if not candidates:
        return CutPath("vertical", fallback.coefficients, len(xs), float('nan'),
                       0., "curve_does_not_improve_vertical_baseline")
    kind, coef, sample_support, residual = candidates[-1]
    sampled = np.polyval(coef, np.linspace(-1, 1, 101))
    shift = float(np.max(np.abs(sampled - separator.fraction)))
    span = float(np.ptp(sampled))
    # Reject large departures even when interpolation fits well: they may be
    # another nearby fiscal column, a binding artefact, or true page warp.
    if (shift > .020 or span > .030 or np.min(sampled) <= .40
            or np.max(sampled) >= .60 or residual > .0035):
        return CutPath("vertical", fallback.coefficients, sample_support, residual,
                       shift, "unsafe_or_unstable_curved_path")
    return CutPath(kind, tuple(float(c) for c in coef), sample_support,
                   residual, shift, "verified_local_rule_trace")


def crop_tables_along_path(image: Image.Image, cut: CutPath,
                           overlap_fraction: float = .004
                           ) -> tuple[Image.Image, Image.Image, int]:
    """Trim each scan row at its own boundary without rotating text or pages.

    The output is rectangular: pixels outside the variable boundary are white.
    Purely vertical cuts remain byte-identical to the previous cropping logic.
    """
    if cut.kind == "vertical":
        return crop_tables(image, cut.coefficients[-1], overlap_fraction)
    if not (0 <= overlap_fraction <= .015):
        raise ValueError("Overlap must be between 0 and 1.5% of page width")
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    centers = np.rint(cut.fractions(h) * w).astype(np.int32)
    if np.any((centers <= round(w * .35)) | (centers >= round(w * .65))):
        raise ValueError("Unsafe slanted/curved cut")
    pad = round(overlap_fraction * w)
    left_end = np.clip(centers + pad, 0, w)
    right_start = np.clip(centers - pad, 0, w)
    left_width = int(left_end.max())
    right_origin = int(right_start.min())
    first = np.full((h, left_width), 255, dtype=np.uint8)
    second = np.full((h, w - right_origin), 255, dtype=np.uint8)
    for y in range(h):
        a, b = int(left_end[y]), int(right_start[y])
        first[y, :a] = gray[y, :a]
        second[y, b-right_origin:] = gray[y, b:]
    return Image.fromarray(first), Image.fromarray(second), int(round(np.median(centers)))


def crop_tables(image: Image.Image, fraction: float, overlap_fraction: float = .004
               ) -> tuple[Image.Image, Image.Image, int]:
    """Preserve a tiny shared margin at the dividing rule, not a large text crop."""
    if not (.35 < fraction < .65):
        raise ValueError("Cut is not approximately central")
    if not (0 <= overlap_fraction <= .015):
        raise ValueError("Overlap must be between 0 and 1.5% of page width")
    w, h = image.size
    x = round(w * fraction)
    pad = round(w * overlap_fraction)
    return (image.crop((0, 0, min(w, x + pad), h)),
            image.crop((max(0, x - pad), 0, w, h)), x)


def _preview(image: Image.Image, result: Separator, title: str,
             path: CutPath | None = None, approved: bool | None = None) -> Image.Image:
    view = image.convert("RGB")
    view.thumbnail((1100, 1100), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(view)
    if path is None:
        path = CutPath("vertical", (result.fraction,), 0, float('nan'), 0., "legacy")
    frac = path.fractions(view.height)
    points = [(round(view.width * float(frac[y])), y)
              for y in range(31, view.height, 2)]
    if len(points) >= 2:
        draw.line(points, fill=(220, 0, 210), width=3)
    if path.kind != "vertical":
        x0 = round(view.width * result.fraction)
        for y in range(35, view.height, 18):
            draw.line([(x0, y), (x0, min(y+7, view.height-1))],
                      fill=(245, 140, 0), width=2)
    for r in (.42, .58):
        xx = round(view.width * r)
        for y in range(31, view.height, 18):
            draw.line([(xx, y), (xx, min(y + 8, view.height - 1))], fill=(50, 155, 230), width=2)
    draw.rectangle((0, 0, view.width, 31), fill=(21, 45, 62))
    status = "ACCEPTED" if (result.accepted if approved is None else approved) else "REVIEW ONLY"
    draw.text((8, 8), f"{title} | {status} | x={result.fraction:.2%} | "
              f"{path.kind} | {result.reason}", fill="white")
    return view


def _contact_sheet(files: list[Path], target: Path) -> None:
    if not files:
        return
    tile_w, tile_h = 500, 600
    sheet = Image.new("RGB", (tile_w * 3, tile_h * ((len(files) + 2)//3)), "white")
    for i, file in enumerate(files):
        with Image.open(file) as src:
            image = src.convert("RGB")
        image.thumbnail((tile_w - 14, tile_h - 26), Image.Resampling.LANCZOS)
        x, y = (i % 3)*tile_w + 7, (i // 3)*tile_h + 21
        sheet.paste(image, (x, y))
        ImageDraw.Draw(sheet).text((x, y - 17), file.stem, fill="black")
    sheet.save(target, quality=86)


def split_year(*, year: int, pages: list[int] | None, dpi: int,
               input_root: Path = DEFAULT_INPUT, output_root: Path = DEFAULT_OUTPUT,
               overlap_fraction: float = .004, force_review: bool = False,
               cut_mode: str = "auto") -> list[dict]:
    folder = input_root / str(year)
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing first-stage outputs at {folder}; run code/split_spreads.py first")
    prior_status: dict[int, str] = {}
    prior_manifest = folder / "split_review.csv"
    if prior_manifest.exists():
        with prior_manifest.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("pdf_page") and row.get("status"):
                    prior_status[int(row["pdf_page"])] = row["status"]
    sources: list[tuple[int, str, Path]] = []
    wanted = set(pages) if pages else None
    for source in sorted(folder.glob(f"p????_*_{dpi}dpi.png")):
        m = PATTERN.fullmatch(source.name)
        if m:
            page = int(m.group("page"))
            if wanted is None or page in wanted:
                if prior_status.get(page, "accepted") != "accepted" and not force_review:
                    print(f"{year} PDF p.{page} {m.group('side')}: skip; stage 1 status={prior_status[page]}")
                    continue
                sources.append((page, m.group("side"), source))
    if not sources:
        raise FileNotFoundError(f"No first-stage PNGs in {folder} matching year={year}, pages={pages}, dpi={dpi}")
    out = output_root / str(year)
    preview_dir = out / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    previews: list[Path] = []
    for page, side, src in sources:
        with Image.open(src) as im:
            image = im.convert("L")
        result = detect_separator(image)
        path = detect_cut_path(image, result, mode=cut_mode)
        # Only the specific old rejection for excessive border drift may be
        # recovered by a well-supported fit. Never override weak/ambiguous
        # table identification.
        recovered = (not result.accepted and
                     result.reason == "curved_or_inconsistent_border_needs_review" and
                     path.kind != "vertical")
        safe = result.accepted or recovered
        stem = f"p{page:04d}_{side}"
        view = _preview(image, result, f"{year} PDF p.{page} {side}",
                        path, approved=safe)
        view_path = preview_dir / f"{stem}_separator.jpg"
        view.save(view_path, quality=90)
        previews.append(view_path)
        first_path = second_path = ""
        status = "accepted" if safe else "needs_review"
        if safe or force_review:
            first, second, _ = crop_tables_along_path(image, path, overlap_fraction)
            first_file = out / f"{stem}_col1_{dpi}dpi.png"
            second_file = out / f"{stem}_col2_{dpi}dpi.png"
            first.save(first_file)
            second.save(second_file)
            first_path, second_path = str(first_file), str(second_file)
            if not safe:
                status = "provisional_review_only"
        else:
            # If a previous run accepted a page but a new run does not, remove
            # stale output so it cannot be passed to Mistral inadvertently.
            for n in (1, 2):
                stale = out / f"{stem}_col{n}_{dpi}dpi.png"
                if stale.exists():
                    stale.unlink()
        row = {"year": year, "pdf_page": page, "printed_side": side,
               "status": status, "reason": ("recovered_curved_border" if recovered else result.reason),
               "cut_method": path.kind, "cut_path_coefficients": ";".join(f"{v:.8g}" for v in path.coefficients),
               "cut_path_points": path.sample_count,
               "cut_path_residual_fraction": (round(path.residual_fraction, 6)
                                               if np.isfinite(path.residual_fraction) else ""),
               "cut_path_max_shift_fraction": round(path.max_shift_fraction, 6),
               "cut_path_reason": path.reason,
               "cut_fraction": round(result.fraction, 6),
               "cut_x_px": round(image.width * result.fraction),
               "image_width_px": image.width,
               "vertical_strength": round(result.strength, 5),
               "header_rules_before": round(result.left_header_rules, 4),
               "header_rules_after": round(result.right_header_rules, 4),
               "second_candidate": round(result.second_strength, 5),
               "confidence_heuristic": round(result.confidence, 4),
               "border_drift_fraction": round(result.drift_fraction, 5),
               "input_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
               "source_image": str(src), "table_1": first_path, "table_2": second_path,
               "preview": str(view_path), "algorithm": VERSION}
        rows.append(row)
        print(f"{year} PDF p.{page} {side}: {status}, "+
              (f"cut={100*result.fraction:.2f}%, method={path.kind}, line={result.strength:.3f}"
               if result.strength else result.reason))
    fields = ["year", "pdf_page", "printed_side", "status", "reason", "cut_method",
              "cut_path_coefficients", "cut_path_points", "cut_path_residual_fraction",
              "cut_path_max_shift_fraction", "cut_path_reason", "cut_fraction",
              "cut_x_px", "image_width_px", "vertical_strength", "header_rules_before",
              "header_rules_after", "second_candidate", "confidence_heuristic",
              "border_drift_fraction", "input_sha256",
              "source_image", "table_1", "table_2", "preview", "algorithm"]
    report = out / "table_split_review.csv"
    with report.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _contact_sheet(previews, out / "review_contact_sheet.jpg")
    print(f"Review CSV: {report}")
    print(f"Contact sheet: {out / 'review_contact_sheet.jpg'}")
    print("No OCR was performed. Review the central separator before using these crops with Mistral.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--pages", nargs="+", type=int, default=None,
                        help="PDF page numbers in the first-stage outputs; default all")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Must equal the DPI used for code/split_spreads.py (default 200)")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overlap", type=float, default=.004,
                        help="Shared border margin as fraction of printed-page width")
    parser.add_argument("--cut-mode", choices=["auto", "vertical"], default="auto",
                        help="auto fits well-supported slanted/curved boundaries; vertical restores v1 cropping")
    parser.add_argument("--force-review", action="store_true",
                        help="Write provisional images for uncertain cuts; never treat them as validated")
    args = parser.parse_args()
    if args.dpi < 72 or args.dpi > 400:
        parser.error("--dpi must be between 72 and 400")
    split_year(year=args.year, pages=args.pages, dpi=args.dpi,
               input_root=args.input_root, output_root=args.output_root,
               overlap_fraction=args.overlap, force_review=args.force_review,
               cut_mode=args.cut_mode)


if __name__ == "__main__":
    main()
