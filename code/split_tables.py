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
import concurrent.futures
import contextlib
import multiprocessing
import csv
import hashlib
import re
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentation_runtime import Progress, contact_sheets, summary_tables

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "derived" / "situation_financiere" / "split_pages"
DEFAULT_OUTPUT = ROOT / "derived" / "situation_financiere" / "table_panels"
VERSION = "multiprocess_stage2_v1_6"
PATTERN = re.compile(r"^p(?P<page>\d{4})_(?P<side>left|right|single)_(?P<dpi>\d+)dpi\.png$")


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



def layout_bucket(page: int, side: str) -> str:
    """Use the side of a spread, or PDF-page parity on single-page scans."""
    if side in {"left", "right"}:
        return side
    return f"single_{page % 2}"


def layout_priors(sources: list[tuple[int, str, Path]],
                  initial: dict[Path, Separator]) -> dict[str, float]:
    """Robust per-year positions from the already-accepted pages only.

    A prior guides a local image search; it NEVER authorizes cropping by itself.
    A dispersion check prevents using a misleading page-parity assumption.
    """
    groups: dict[str, list[float]] = {}
    for page, side, path in sources:
        result = initial[path]
        if result.accepted and result.strength >= .08:
            groups.setdefault(layout_bucket(page, side), []).append(result.fraction)
    priors: dict[str, float] = {}
    for bucket, values in groups.items():
        if len(values) < 2:
            continue
        m = float(np.median(values))
        median_deviation = float(np.median(np.abs(np.asarray(values) - m)))
        if median_deviation <= .024 and .40 <= m <= .60:
            priors[bucket] = m
    return priors


def detect_separator_tolerant(image: Image.Image, expected: float) -> Separator | None:
    """Second detector, restricted to a previously learned local neighborhood.

    Unlike a coordinate template, the expected fraction is only a prior. A cut
    requires a detected long rule and independent left/right layout evidence.
    """
    if not .40 <= expected <= .60:
        return None
    gray = _resize(np.asarray(image.convert("L"), dtype=np.uint8))
    h, w = gray.shape
    enhanced = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(gray)
    binary = cv2.threshold(enhanced, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Shorter than the strict detector, but still longer than most characters.
    length = max(13, round(h * .017))
    rules = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
    profile = (rules[round(h*.08):round(h*.88)] > 0).mean(axis=0).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), sigmaX=1.6)[0]
    lower = max(0, round((expected - .032) * w))
    upper = min(w, round((expected + .032) * w))
    candidates = _candidate_peaks(profile, lower, upper)
    # Check several row bands because introductory pages have the column
    # heading substantially lower than ordinary pages.
    bands = [( .07, .25), (.21, .39), (.38, .56), (.55, .74)]
    results = []
    for x in candidates:
        if abs(x/w - expected) > .032:
            continue
        line_strength = float(profile[x])
        if line_strength < .040:
            continue
        left_support = 0.0
        right_support = 0.0
        good_bands = 0
        active_bands = 0
        for ya, yb in bands:
            line_profile = (rules[round(h*ya):round(h*yb)] > 0).mean(axis=0).astype(np.float32)
            # The separator is between dense fiscal rules on its left, and
            # the rule-free NOMS column for the next table on its right.
            lo1, hi1 = max(0, x-round(w*.083)), max(0, x-round(w*.018))
            lo2, hi2 = min(w, x+round(w*.018)), min(w, x+round(w*.083))
            left = float(line_profile[lo1:hi1].sum())
            right = float(line_profile[lo2:hi2].sum())
            local = float(line_profile[max(0,x-3):min(w,x+4)].max())
            if local > .06:
                active_bands += 1
            if local > .06 and left >= .22 and left >= 2.2 * max(right, .05) and left-right >= .18:
                good_bands += 1
            left_support += left
            right_support += right
        if good_bands < 2 or active_bands < 3:
            continue
        # Prefer rules close to the expected page-side position, not an
        # unrelated fiscal-column rule just inside the search band.
        distance = abs(x / w - expected)
        quality = (line_strength * (1. + .22 * good_bands)
                   * np.exp(-.5 * (distance / .020) ** 2))
        results.append((float(quality), x, line_strength,
                        left_support, right_support, good_bands))
    if not results:
        return None
    results.sort(reverse=True)
    top = results[0]
    competitor = next((r for r in results[1:] if abs(r[1] - top[1]) > w*.012), None)
    if competitor is not None and top[0] < competitor[0] * 1.20:
        return None
    cut_x = int(top[1])
    # A genuinely drifting printed rule is recovered by the already existing
    # robust linear/quadratic path fitter, not by increasing guessed margins.
    drift = _line_drift(rules, cut_x)
    if not np.isfinite(drift) or drift > .024:
        return None
    confidence = min(.95, max(.15, top[5]/4 * top[2]/.15))
    return Separator(cut_x/w, top[2], top[3], top[4],
                     float(competitor[0] if competitor else 0.), confidence,
                     drift, True, "recovered_tolerant_header_rule")


def detect_single_table_side(image: Image.Image) -> str | None:
    """Recognize a truly blank parallel half, never guess from a missing rule.

    Both activity and long printed vertical rules must be strongly one-sided.
    Ordinary pages with two tables (including half-filled last pages) remain
    two-table pages even if the bottom of both tables is blank.
    """
    gray = _resize(np.asarray(image.convert("L"), dtype=np.uint8))
    h, w = gray.shape
    bw = (gray < 165).astype(np.uint8) * 255
    rules = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT,
                                                      (1, max(15, round(h*.018)))))
    ink = []
    vertical = []
    for l, r in ((.10, .38), (.63, .91)):
        x1, x2 = round(w*l), round(w*r)
        area = bw[round(h*.10):round(h*.86), x1:x2]
        varea = rules[round(h*.10):round(h*.86), x1:x2]
        ink.append(float((area > 0).mean()))
        vertical.append(float((varea > 0).mean()))
    # A clean white half must differ dramatically from a ruled table half.
    # The thresholds deliberately prioritize not losing a genuine second table.
    for i, side in enumerate(("left", "right")):
        other = 1-i
        if (ink[i] >= .012 and vertical[i] >= .009
            and ink[other] <= .0015 and vertical[other] <= .0004
            and ink[i] >= 12 * max(ink[other], .0001)):
            return side
    return None


def aggregate_review_csvs(output_root: Path) -> Path:
    """Concatenate all year-level table reports into one reproducible CSV.

    The aggregate is refreshed after each run, contains every available year,
    and is written atomically. It never takes data from images or OCR caches.
    """
    files = sorted(output_root.glob("[0-9][0-9][0-9][0-9]/table_split_review.csv"))
    if not files:
        raise FileNotFoundError(f"No annual table_split_review.csv files under {output_root}")
    merged: list[dict[str, str]] = []
    fields: list[str] = []
    for file in files:
        with file.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or []:
                if name not in fields:
                    fields.append(name)
            for row in reader:
                if row.get("year") and row.get("pdf_page"):
                    merged.append(row)
    side_order = {"single":0, "left":1, "right":2}
    merged.sort(key=lambda r: (int(r["year"]), int(r["pdf_page"]),
                               side_order.get(r.get("printed_side", ""), 9)))
    target = output_root / "all_years_table_split_review.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".table_split_", suffix=".csv", dir=output_root)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(merged)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target



@dataclass(frozen=True)
class CutPath:
    """x-coordinate as a fraction of width, polynomial in normalized height [-1, 1]."""
    kind: str  # vertical, linear, quadratic
    coefficients: tuple[float, ...]  # highest power first (np.polyval)
    sample_count: int
    residual_fraction: float
    max_shift_fraction: float
    reason: str
    active_y_range: tuple[float, float] | None = None

    def fractions(self, height: int) -> np.ndarray:
        y = (np.arange(height, dtype=np.float64) + .5) * (2. / height) - 1.
        if self.active_y_range is not None:
            # Do not extrapolate a fitted curve through blank paper below a
            # short table. Hold the last observed boundary position instead.
            y = np.clip(y, *self.active_y_range)
        return np.polyval(self.coefficients, y)




def _extreme_fit(ys: np.ndarray, xs: np.ndarray, degree: int
                 ) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Robust fit for a *strongly slanted* detected rule, not generic pixels.

    This is deliberately separate from _fit_robust, so strict/tolerant
    detections continue to use exactly the v1.3 limits and behavior.
    """
    if len(xs) < 10:
        return None
    keep = np.ones(len(xs), dtype=bool)
    for _ in range(4):
        if keep.sum() < 9:
            return None
        coef = np.polyfit(ys[keep], xs[keep], degree)
        err = np.abs(xs - np.polyval(coef, ys))
        updated = err <= .007
        if updated.sum() < max(9, round(len(xs) * .65)):
            break
        keep = updated
    if keep.sum() < max(9, round(len(xs) * .65)) or np.ptp(ys[keep]) < 1.25:
        return None
    coef = np.polyfit(ys[keep], xs[keep], degree)
    residual = float(np.sqrt(np.mean((xs[keep] - np.polyval(coef, ys[keep]))**2)))
    return coef, keep, residual


def _extreme_path_from_samples(ys: np.ndarray, xs: np.ndarray,
                               expected: float) -> CutPath | None:
    """Test vertical/linear/quadratic hypotheses with expanded limits.

    No candidate is accepted simply because a high-degree polynomial fits:
    the source samples have to come from a traced printed rule and span
    most of the page. Quadratic requires a meaningful improvement on linear.
    """
    if len(xs) < 10 or np.ptp(ys) < 1.25:
        return None
    baseline = float(np.sqrt(np.mean((xs - np.median(xs))**2)))
    fits: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for name, degree in (("linear", 1), ("quadratic", 2)):
        fit = _extreme_fit(ys, xs, degree)
        if fit is not None:
            fits[name] = fit
    if not fits:
        return None
    straight = fits.get("linear")
    curved = fits.get("quadratic")
    # A genuinely near-vertical trace is preferable to any polynomial fit.
    if baseline <= .0028:
        coef = np.array([float(np.median(xs))])
        kind, residual, count = "vertical", baseline, len(xs)
    elif straight is not None and straight[2] <= .0065:
        coef, keep, residual = straight
        kind, count = "linear", int(keep.sum())
    elif curved is not None and curved[2] <= .0065:
        coef, keep, residual = curved
        kind, count = "quadratic", int(keep.sum())
    else:
        return None
    if curved is not None and curved[2] <= .0065:
        curve_coef, curve_keep, curve_error = curved
        if straight is None:
            better = baseline - curve_error > .002
        else:
            line_error = straight[2]
            sample_y = np.linspace(-1., 1., 80)
            curve_size = float(np.max(np.abs(np.polyval(curve_coef, sample_y)
                                               - np.polyval(straight[0], sample_y))))
            better = (line_error - curve_error > .0012 and
                      curve_error <= .7 * line_error and curve_size >= .003)
        if better:
            coef, residual = curve_coef, curve_error
            kind, count = "quadratic", int(curve_keep.sum())
    sampled = np.polyval(coef, np.linspace(-1., 1., 101))
    span = float(np.ptp(sampled))
    shift = float(np.max(np.abs(sampled - expected)))
    midpoint = float(np.polyval(coef, 0.))
    if (residual > .0065 or span > .145 or shift > .092
            or abs(midpoint-expected) > .040
            or np.min(sampled) <= .36 or np.max(sampled) >= .64):
        return None
    return CutPath(kind, tuple(float(x) for x in coef), count,
                   float(residual), shift, "verified_extreme_rule_trace")


def detect_separator_extreme(image: Image.Image, expected: float
                             ) -> tuple[Separator, CutPath] | None:
    """Third-pass *review-only* boundary detection from local rule evidence.

    Unlike v1.3's full-height vertical profile, this works on 17 independent
    horizontal strips: the same slanted rule may move across X between strips.
    A robust trace must have a fiscal-column-to-wide-NOMS transition near its
    top, long-rule evidence across the page, and a stable fitted geometry.
    A location prior is a search constraint, NEVER proof of a separator.
    """
    if not .40 <= expected <= .60:
        return None
    gray = _resize(np.asarray(image.convert("L"), dtype=np.uint8))
    h, w = gray.shape
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.threshold(enhanced, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Short rules survive gentle slant; text strokes are mostly suppressed.
    rules = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(
                                cv2.MORPH_RECT, (1, max(9, round(h * .012)))))
    centers = np.linspace(.105, .895, 17)
    band_data: list[list[tuple[int, float, float, float]]] = []
    left_limit = max(3, round((expected - .095) * w))
    right_limit = min(w-3, round((expected + .095) * w))
    for cy in centers:
        half = max(12, round(h * .033))
        y = round(cy * h)
        density = (rules[max(0, y-half):min(h, y+half)] > 0).mean(axis=0).astype(np.float32)
        smooth = cv2.GaussianBlur(density.reshape(1, -1), (0, 0), sigmaX=1.5)[0]
        # One detector applied afresh to every strip; no template transfer.
        peaks = _candidate_peaks(smooth, left_limit, right_limit)
        options = []
        for x in peaks:
            strength = float(smooth[x])
            if strength < .055:
                continue
            la, lb = max(0, x-round(w*.10)), max(0, x-round(w*.016))
            ra, rb = min(w, x+round(w*.016)), min(w, x+round(w*.105))
            before = float(smooth[la:lb].sum())
            after = float(smooth[ra:rb].sum())
            # This differentiates the table separator from internal fiscal rules.
            signature = (before >= .20 and before-after >= .16 and
                         before >= 1.7 * max(after, .07))
            if abs(x / w - expected) > .095:
                continue
            options.append((x, strength, before, after, bool(signature)))
        band_data.append(options)

    # Start from independent header-like evidence in the upper half of the
    # page; follow across bands with a predictor that tolerates strong drift.
    seeds = []
    for i, cy in enumerate(centers):
        if cy > .40:
            break
        for x, strength, before, after, signature in band_data[i]:
            if signature and abs(x/w - expected) <= .07:
                merit = strength * (1 + min(2., (before-after)/max(before, .1)))
                merit *= float(np.exp(-.5 * ((x/w-expected)/.06)**2))
                seeds.append((merit, i, x))
    if not seeds:
        return None
    seeds.sort(reverse=True)
    candidates = []
    for _, seed_i, seed_x in seeds[:12]:
        selected: dict[int, tuple[int, float, float, float, bool]] = {}
        selected[seed_i] = next(item for item in band_data[seed_i] if item[0] == seed_x)
        for direction in (-1, 1):
            last_points = [(seed_i, seed_x)]
            for i in range(seed_i+direction, len(centers) if direction > 0 else -1, direction):
                # Estimate local trend where possible, but heavily constrain
                # jumps to avoid switching to a neighbouring fiscal column.
                prev_i, prev_x = last_points[-1]
                slope = ((last_points[-1][1]-last_points[-2][1]) /
                         (last_points[-1][0]-last_points[-2][0])) if len(last_points)>=2 else 0.
                slope = float(np.clip(slope, -w*.019, w*.019))
                predicted = prev_x + slope*(i-prev_i)
                limit = w * .025 * max(1, abs(i-prev_i))
                local = [o for o in band_data[i] if abs(o[0]-predicted) <= limit]
                if not local:
                    continue
                def quality(option):
                    x, strength, before, after, signature = option
                    proximity = float(np.exp(-.5*((x-predicted)/(w*.014*max(1,abs(i-prev_i))))**2))
                    return strength*proximity*(1.18 if signature else 1.0)
                best = max(local, key=quality)
                if quality(best) < .032:
                    continue
                selected[i] = best
                last_points.append((i,best[0]))
        if len(selected) < 10:
            continue
        ordered = sorted(selected)
        ys = np.asarray([2*centers[i]-1 for i in ordered], dtype=np.float64)
        xs = np.asarray([selected[i][0]/w for i in ordered], dtype=np.float64)
        header = sum(bool(selected[i][4]) for i in ordered if centers[i] <= .40)
        if header < 2 or np.ptp(ys) < 1.3:
            continue
        path = _extreme_path_from_samples(ys, xs, expected)
        if path is None:
            continue
        # The path must remain close to observed rule pixels, not merely fit
        # a few isolated high-contrast features.
        residuals = np.abs(xs-np.polyval(path.coefficients, ys))
        if int((residuals <= .010).sum()) < 10:
            continue
        strength = float(np.median([selected[i][1] for i in ordered]))
        if strength < .07:
            continue
        score = (len(ordered)/len(centers) + header/10. + .6*strength
                 - 8*path.residual_fraction - 1.0*abs(float(np.median(xs))-expected))
        candidates.append((score, path, selected, ys, xs, header, strength))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[0]
    # Resolve genuinely different tracked separators; competing solutions are
    # evidence of ambiguity, not permission to guess.
    mid = float(np.polyval(top[1].coefficients, 0.))
    for other in candidates[1:]:
        other_mid = float(np.polyval(other[1].coefficients, 0.))
        if abs(other_mid-mid) > .014 and other[0] >= top[0]-.07:
            return None
    _, path, points, ys, xs, header, strength = top
    result = Separator(mid, strength, float(header), 0., 0.,
                       min(.85, .35+.035*len(points)),
                       float(np.ptp(xs)), True, "recovered_extreme_rule_trace")
    return result, path



def _short_table_active_region(image: Image.Image, expected: float
                               ) -> tuple[float, float] | None:
    """Find a well-supported *shared* table area ending before the page bottom.

    Count long printed rules in both tables, away from the page-edge shadows
    and away from the central separator. A dark crease, arbitrary text, or one
    populated half is insufficient to establish two short parallel tables.
    This is used ONLY after three existing passes failed.
    """
    if not .40 <= expected <= .60:
        return None
    gray = _resize(np.asarray(image.convert("L"), dtype=np.uint8))
    h, w = gray.shape
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.threshold(enhanced, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    rules = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(
                                 cv2.MORPH_RECT, (1, max(11, round(h * .012))))) > 0
    windows = ((round(w * .11), round(w * (expected - .04))),
               (round(w * (expected + .10)), round(w * .89)))
    bounds: list[tuple[int, int]] = []
    indices = np.arange(h)
    length = max(5, round(h * .012))
    kernel = np.ones(length, dtype=np.float64) / length
    for a, b in windows:
        if b-a < round(w * .14):
            return None
        density = rules[:, a:b].mean(axis=1)
        activity = np.convolve(density, kernel, mode="same")
        typical = float(np.quantile(activity[round(h*.09):round(h*.45)], .75))
        if typical < .008:  # No reliable collection of printed column rules.
            return None
        threshold = max(.0035, typical * .17)
        eligible = ((activity > threshold) & (indices >= round(h * .045))
                    & (indices <= round(h * .91)))
        active = np.flatnonzero(eligible)
        if len(active) < round(h * .23):
            return None
        first, last = int(active[0]), int(active[-1])
        if (last-first < round(h*.26) or
                float(eligible[first:last+1].mean()) < .58):
            return None
        bounds.append((first, last))
    top = min(pair[0] for pair in bounds)
    bottom = max(pair[1] for pair in bounds)
    # Require both sides to begin around the header and end together. A
    # full-height scan is NOT a short table even if one side is shorter.
    if (top > h*.23 or bottom < h*.37 or bottom > h*.79
            or abs(bounds[0][1]-bounds[1][1]) > h*.12):
        return None
    y0 = max(0, top-round(h*.018))
    y1 = min(h, bottom+round(h*.023))
    return y0/h, y1/h


def detect_separator_short_table(image: Image.Image, expected: float,
                                 original: Separator
                                 ) -> tuple[Separator, CutPath] | None:
    """Fourth pass: rerun familiar detectors on the observed printed region.

    It never alters results from passes 1-3. It also never accepts a separator
    based only on where earlier pages were split. The cropped region must
    independently demonstrate the fiscal-columns-to-wide-NOMS signature and
    yield at least ten consistent traced printed-rule samples.
    """
    span = _short_table_active_region(image, expected)
    if span is None:
        return None
    top, bottom = span
    y0, y1 = round(image.height * top), round(image.height * bottom)
    if y1-y0 < 150:
        return None
    region = image.crop((0, y0, image.width, y1))
    candidate = detect_separator(region)
    if not candidate.accepted:
        local_tolerant = detect_separator_tolerant(region, expected)
        if local_tolerant is not None:
            candidate = local_tolerant
    if (candidate.strength < .075 or
            abs(candidate.fraction-expected) > .045 or
            (original.strength >= .075 and
             abs(candidate.fraction-original.fraction) > .035)):
        return None
    local_path = detect_cut_path(region, candidate)
    reliable = ((candidate.accepted or
                 (candidate.reason == "curved_or_inconsistent_border_needs_review"
                  and local_path.kind != "vertical"))
                and local_path.sample_count >= 10
                and local_path.reason != "too_few_consistent_rule_samples")
    if not reliable:
        # The original extreme tracer is reused on the *short* image, where
        # its full-height sampling now corresponds to the actual table area.
        extreme = detect_separator_extreme(region, expected)
        if extreme is None:
            return None
        candidate, local_path = extreme
    # Convert a polynomial in cropped-region height to a polynomial in full
    # image height, then clamp it to the observed table interval so it cannot
    # bend unpredictably through the blank lower portion.
    scale = bottom-top
    u_to_local = np.poly1d([1./scale, -(top+bottom-1.)/scale])
    full_coef = np.poly1d(local_path.coefficients)(u_to_local).c
    if not np.all(np.isfinite(full_coef)):
        return None
    path = CutPath(local_path.kind, tuple(float(v) for v in full_coef),
                   local_path.sample_count, local_path.residual_fraction,
                   local_path.max_shift_fraction,
                   "verified_short_table_rule_trace", (2*top-1, 2*bottom-1))
    positions = path.fractions(image.height)
    if (positions.min() <= .36 or positions.max() >= .64 or
            np.max(np.abs(positions-expected)) > .09):
        return None
    midpoint = float(positions[len(positions)//2])
    result = Separator(midpoint, candidate.strength, candidate.left_header_rules,
                       candidate.right_header_rules, candidate.second_strength,
                       min(.85, candidate.confidence), candidate.drift_fraction,
                       True, "recovered_short_table_active_region")
    return result, path


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
             path: CutPath | None = None, approved: bool | None = None,
             single_table: str | None = None) -> Image.Image:
    view = image.convert("RGB")
    view.thumbnail((1100, 1100), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(view)
    if path is None:
        path = CutPath("vertical", (result.fraction,), 0, float('nan'), 0., "legacy")
    if single_table is None:
        frac = path.fractions(view.height)
        points = [(round(view.width * float(frac[y])), y)
                  for y in range(31, view.height, 2)]
        if len(points) >= 2:
            draw.line(points, fill=(220, 0, 210), width=3)
    if single_table is None and path.kind != "vertical":
        x0 = round(view.width * result.fraction)
        for y in range(35, view.height, 18):
            draw.line([(x0, y), (x0, min(y+7, view.height-1))],
                      fill=(245, 140, 0), width=2)
    if single_table is None:
        for r in (.42, .58):
            xx = round(view.width * r)
            for y in range(31, view.height, 18):
                draw.line([(xx, y), (xx, min(y + 8, view.height - 1))], fill=(50, 155, 230), width=2)
    draw.rectangle((0, 0, view.width, 31), fill=(21, 45, 62))
    status = "ACCEPTED" if (result.accepted if approved is None else approved) else "REVIEW ONLY"
    label = (f"SINGLE TABLE {single_table} | no dividing line" if single_table
             else f"x={result.fraction:.2%} | {path.kind} | {result.reason}")
    draw.text((8, 8), f"{title} | {status} | {label}", fill="white")
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


def _init_table_worker() -> None:
    # Four independent OpenCV processes should not each create dozens of
    # additional internal threads (oversubscription on an Intel Mac).
    cv2.setNumThreads(1)


def _detect_initial_task(task: tuple[int, str, Path]) -> tuple[Path, Separator]:
    _, _, src = task
    with Image.open(src) as image:
        return src, detect_separator(image)


def _run_table_task(task: tuple) -> dict:
    (year, page, side, src, original, expected, out, dpi,
     overlap_fraction, force_review, cut_mode, extreme, short_table) = task
    with Image.open(src) as im:
        image = im.convert('L')
    result = original
    single_table = detect_single_table_side(image)
    if single_table is None and not original.accepted and expected is not None:
        alternative = detect_separator_tolerant(image, expected)
        if alternative is not None:
            result = alternative
    path = detect_cut_path(image, result, mode=cut_mode)
    extreme_found = False
    recovered_base = (not result.accepted and
                      result.reason == 'curved_or_inconsistent_border_needs_review'
                      and path.kind != 'vertical')
    if (extreme and cut_mode == 'auto' and single_table is None
            and not result.accepted and not recovered_base and expected is not None):
        candidate = detect_separator_extreme(image, expected)
        if candidate is not None:
            result, path = candidate
            extreme_found = True
    short_found = False
    if (short_table and cut_mode == 'auto' and single_table is None
            and not result.accepted and not recovered_base
            and not extreme_found and expected is not None):
        short_candidate = detect_separator_short_table(image, expected, original)
        if short_candidate is not None:
            result, path = short_candidate
            short_found = True
    recovered = (not result.accepted and
                 result.reason == 'curved_or_inconsistent_border_needs_review'
                 and path.kind != 'vertical')
    safe = result.accepted or recovered or single_table is not None
    stem = f'p{page:04d}_{side}'
    accepted_dir, review_dir, preview_dir = (out / 'accepted', out / 'needs_review',
                                             out / 'previews')
    for directory in (accepted_dir, review_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)
    # Never leave an OCR-ready accepted crop if the same page is later rejected.
    for directory in (out, accepted_dir, review_dir,
                      out / 'extreme_candidates', out / 'short_table_candidates'):
        if directory.exists():
            for old in directory.glob(f'{stem}_*'):
                if old.is_file() and old.suffix.lower() in {'.png', '.jpg', '.pdf'}:
                    old.unlink()
    for old in preview_dir.glob(f'{stem}_*.jpg'):
        old.unlink()
    status = 'accepted' if safe else 'needs_review'
    if single_table is not None:
        status = 'accepted_single_table'
    elif short_found:
        status = 'accepted_short_table_pending_review'
    elif extreme_found:
        status = 'accepted_extreme_pending_review'
    elif not safe and force_review:
        status = 'provisional_review_only'
    is_review = status not in {'accepted', 'accepted_single_table'}
    target_dir = review_dir if is_review else accepted_dir
    label = (f'{year} PDF p.{page} {side} | SHORT TABLE PENDING REVIEW' if short_found else
             f'{year} PDF p.{page} {side} | EXTREME PENDING REVIEW' if extreme_found else
             f'{year} PDF p.{page} {side}')
    view = _preview(image, result, label,
                    path, approved=safe and not (extreme_found or short_found),
                    single_table=single_table)
    view_path = (review_dir if is_review else preview_dir) / f'{stem}_separator.jpg'
    view.save(view_path, quality=90)
    first_path = second_path = ''
    if single_table is not None:
        index = 1 if single_table == 'left' else 2
        file = accepted_dir / f'{stem}_col{index}_{dpi}dpi.png'
        image.save(file)
        if index == 1:
            first_path = str(file)
        else:
            second_path = str(file)
    elif safe or force_review:
        first, second, _ = crop_tables_along_path(image, path, overlap_fraction)
        first_file = target_dir / f'{stem}_col1_{dpi}dpi.png'
        second_file = target_dir / f'{stem}_col2_{dpi}dpi.png'
        first.save(first_file)
        second.save(second_file)
        first_path, second_path = str(first_file), str(second_file)
    if is_review:
        # Never leave an unresolved page invisible just because cutting failed.
        image.save(review_dir / f'{stem}_source_{dpi}dpi.png')
    return {'year': year, 'pdf_page': page, 'printed_side': side,
            'status': status,
            'reason': ('single_printed_table_only' if single_table else
                       'recovered_curved_border' if recovered else result.reason),
            'table_count': 1 if single_table else (2 if (safe or force_review) else 0),
            'single_table_side': single_table or '',
            'expected_cut_fraction': (round(expected, 6) if expected is not None else ''),
            'detection_pass': ('single_table' if single_table else
                               'short_table' if short_found else
                               'extreme' if extreme_found else
                               'tolerant' if result is not original else 'strict'),
            'cut_method': 'none' if single_table else path.kind,
            'cut_path_coefficients': ('' if single_table else
                                      ';'.join(f'{v:.8g}' for v in path.coefficients)),
            'cut_path_points': path.sample_count,
            'cut_path_residual_fraction': (round(path.residual_fraction, 6)
                                           if np.isfinite(path.residual_fraction) else ''),
            'cut_path_max_shift_fraction': round(path.max_shift_fraction, 6),
            'cut_path_reason': path.reason,
            'active_table_top_fraction': (round((path.active_y_range[0]+1)/2, 6)
                                          if path.active_y_range else ''),
            'active_table_bottom_fraction': (round((path.active_y_range[1]+1)/2, 6)
                                             if path.active_y_range else ''),
            'cut_fraction': '' if single_table else round(result.fraction, 6),
            'cut_x_px': '' if single_table else round(image.width * result.fraction),
            'image_width_px': image.width,
            'vertical_strength': round(result.strength, 5),
            'header_rules_before': round(result.left_header_rules, 4),
            'header_rules_after': round(result.right_header_rules, 4),
            'second_candidate': round(result.second_strength, 5),
            'confidence_heuristic': round(result.confidence, 4),
            'border_drift_fraction': round(result.drift_fraction, 5),
            'input_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
            'source_image': str(src), 'table_1': first_path, 'table_2': second_path,
            'preview': str(view_path), 'algorithm': VERSION}


def split_year(*, year: int, pages: list[int] | None, dpi: int,
               input_root: Path = DEFAULT_INPUT, output_root: Path = DEFAULT_OUTPUT,
               overlap_fraction: float = .004, force_review: bool = False,
               cut_mode: str = 'auto', extreme: bool = True,
               short_table: bool = True, workers: int = 1) -> list[dict]:
    if workers < 1:
        raise ValueError('workers must be >= 1')
    folder = input_root / str(year)
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing first-stage outputs at {folder}')
    prior_status: dict[int, str] = {}
    prior_manifest = folder / 'split_review.csv'
    if prior_manifest.exists():
        with prior_manifest.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                if row.get('pdf_page') and row.get('status'):
                    prior_status[int(row['pdf_page'])] = row['status']
    # Read only Stage 1 accepted images. Legacy root-level images are supported
    # during migration from the previous checkpoint, without double counting.
    images: dict[tuple[int, str], Path] = {}
    for src in sorted(folder.glob(f'p????_*_{dpi}dpi.png')):
        match = PATTERN.fullmatch(src.name)
        if match:
            images[(int(match.group('page')), match.group('side'))] = src
    for src in sorted((folder / 'accepted').glob(f'p????_*_{dpi}dpi.png')):
        match = PATTERN.fullmatch(src.name)
        if match:
            images[(int(match.group('page')), match.group('side'))] = src
    reference_sources: list[tuple[int, str, Path]] = []
    sources: list[tuple[int, str, Path]] = []
    wanted = set(pages) if pages else None
    for (page, side), src in sorted(images.items()):
        stage1_status = (prior_status.get(page, 'missing_from_manifest')
                         if prior_manifest.exists() else 'accepted')
        if stage1_status not in {'accepted', 'accepted_single'} and not force_review:
            continue
        reference_sources.append((page, side, src))
        if wanted is None or page in wanted:
            sources.append((page, side, src))
    if not sources:
        raise FileNotFoundError(f'No accepted first-stage PNGs for {year}, dpi={dpi}')
    out = output_root / str(year)
    out.mkdir(parents=True, exist_ok=True)
    prog_detect = Progress(f'{year} | Stage 2 detect', len(reference_sources))
    initial: dict[Path, Separator] = {}
    context = (concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
        initializer=_init_table_worker) if workers > 1 else contextlib.nullcontext())
    with context as pool:
        if pool is None:
            _init_table_worker()
            for item in reference_sources:
                src, result = _detect_initial_task(item)
                initial[src] = result
                prog_detect.update()
        else:
            futures = [pool.submit(_detect_initial_task, item) for item in reference_sources]
            for future in concurrent.futures.as_completed(futures):
                src, result = future.result()
                initial[src] = result
                prog_detect.update()
        priors = layout_priors(reference_sources, initial)
        jobs = [(year, page, side, src, initial[src],
                 priors.get(layout_bucket(page, side)), out, dpi,
                 overlap_fraction, force_review, cut_mode, extreme, short_table)
                for page, side, src in sources]
        rows: list[dict] = []
        prog_split = Progress(f'{year} | Stage 2 split', len(jobs))
        if pool is None:
            for job in jobs:
                rows.append(_run_table_task(job))
                prog_split.update()
        else:
            futures = [pool.submit(_run_table_task, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
                prog_split.update()
    fields = ['year', 'pdf_page', 'printed_side', 'status', 'reason', 'table_count',
              'single_table_side', 'expected_cut_fraction', 'detection_pass', 'cut_method',
              'cut_path_coefficients', 'cut_path_points', 'cut_path_residual_fraction',
              'cut_path_max_shift_fraction', 'cut_path_reason',
              'active_table_top_fraction', 'active_table_bottom_fraction', 'cut_fraction',
              'cut_x_px', 'image_width_px', 'vertical_strength', 'header_rules_before',
              'header_rules_after', 'second_candidate', 'confidence_heuristic',
              'border_drift_fraction', 'input_sha256', 'source_image',
              'table_1', 'table_2', 'preview', 'algorithm']
    report = out / 'table_split_review.csv'
    if wanted is not None and report.exists():
        keys = {(int(r['pdf_page']), r['printed_side']) for r in rows}
        with report.open(newline='', encoding='utf-8') as handle:
            for previous in csv.DictReader(handle):
                key = (int(previous['pdf_page']), previous['printed_side'])
                if key not in keys and key[0] not in wanted:
                    rows.append(previous)
    rows.sort(key=lambda r: (int(r['pdf_page']),
                             {'single':0, 'left':1, 'right':2}.get(r['printed_side'],9)))
    with report.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    all_previews = [Path(r['preview']) for r in rows if r.get('preview')]
    pending_previews = [Path(r['preview']) for r in rows
                        if r.get('status') not in {'accepted', 'accepted_single_table'} and r.get('preview')]
    contact_sheets(all_previews, out, name='review_contact_sheet', tile=(500,600))
    contact_sheets(pending_previews, out / 'needs_review',
                   name='review_contact_sheet', tile=(500,600))
    summary_tables(rows, prog_detect.elapsed + prog_split.elapsed)
    print(f'Review CSV: {report}')
    print(f'All-years review CSV: {aggregate_review_csvs(output_root)}')
    print(f'Accepted PNGs: {out / "accepted"}')
    print(f'Needs-review PNGs/previews: {out / "needs_review"}')
    print('No OCR was performed.')
    return rows



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="parallel worker processes (default: up to 4, use 1 for serial)")
    parser.add_argument("--combine-only", action="store_true",
                        help="Regenerate the all-years CSV from existing annual reports, without processing images")
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
    parser.add_argument("--no-extreme", dest="extreme", action="store_false",
                        help="Disable the review-only third pass; reproduce v1.3 decisions")
    parser.add_argument("--no-short-table", dest="short_table", action="store_false",
                        help="Disable fourth pass; reproduce v1.4 decisions")
    parser.add_argument("--force-review", action="store_true",
                        help="Write provisional images for uncertain cuts; never treat them as validated")
    args = parser.parse_args()
    if args.combine_only:
        print(f"All-years review CSV: {aggregate_review_csvs(args.output_root)}")
        return
    if args.year is None:
        parser.error("--year is required unless --combine-only is selected")
    if args.dpi < 72 or args.dpi > 400:
        parser.error("--dpi must be between 72 and 400")
    split_year(year=args.year, pages=args.pages, dpi=args.dpi,
               input_root=args.input_root, output_root=args.output_root,
               overlap_fraction=args.overlap, force_review=args.force_review,
               cut_mode=args.cut_mode, extreme=args.extreme,
               short_table=args.short_table, workers=args.workers)


if __name__ == "__main__":
    main()
