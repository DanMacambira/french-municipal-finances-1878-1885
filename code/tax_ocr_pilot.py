#!/usr/bin/env python3
"""Opt-in tax-only OCR comparison for the 1882 PDF-page-245 pilot.

The experiment never changes source images, the normal OCR cache, or build
outputs.  It puts two original-image strips (commune names and extraordinary
centimes/expiry) beside each other at their original y positions.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL = "mistral-ocr-4-1"
OCR_URL = "https://api.mistral.ai/v1/ocr"

# Visual measurements on 1882 PDF p.245, using the 210dpi panel crops.
# Fractions are relative to the width of each *existing panel image*, NOT PDF.
PRESETS = {
    "left_col2": (0.340, 0.808, 0.950),
    "right_col2": (0.310, 0.753, 0.890),
}


def crop_strips(image: Image.Image, name_right: float, tax_left: float,
                tax_right: float, gap: int = 16) -> Image.Image:
    """Preserve vertical alignment and all pixels within selected columns."""
    if not (0 < name_right < tax_left < tax_right <= 1):
        raise ValueError("Need 0 < name_right < tax_left < tax_right <= 1")
    if gap < 0:
        raise ValueError("gap must be nonnegative")
    source = image.convert("RGB")
    w, h = source.size
    name_end = round(w * name_right)
    tax_start = round(w * tax_left)
    tax_end = round(w * tax_right)
    if name_end <= 0 or tax_end <= tax_start:
        raise ValueError("Crop would be empty")
    new = Image.new("RGB", (name_end + gap + tax_end - tax_start, h), "white")
    new.paste(source.crop((0, 0, name_end, h)), (0, 0))
    new.paste(source.crop((tax_start, 0, tax_end, h)), (name_end + gap, 0))
    return new


def pilot_paths(root: Path, year: int, page: int, panel: str, dpi: int) -> tuple[Path, Path, Path]:
    if panel not in PRESETS:
        raise ValueError(f"No reviewed crop boundaries for {panel!r}")
    source = root / "derived/situation_financiere/images" / str(year) / f"p{page:04d}_{panel}_{dpi}dpi.png"
    base = root / "derived/situation_financiere/tax_ocr_pilot"
    image = base / "images" / str(year) / f"p{page:04d}_{panel}_tax_{dpi}dpi.png"
    cache = base / "ocr" / "mistral" / str(year) / f"p{page:04d}_{panel}_tax.json"
    return source, image, cache


def prepare(root: Path, year: int, page: int, panel: str, dpi: int) -> tuple[Path, Path, str]:
    source, image, cache = pilot_paths(root, year, page, panel, dpi)
    if not source.is_file():
        raise FileNotFoundError(f"Missing rendered panel: {source}\n"
                                "First run make preview SF_ARGS='--years 1882 --center 245 --radius 0 --regions'")
    with Image.open(source) as original:
        result = crop_strips(original, *PRESETS[panel])
        binary = io.BytesIO()
        result.save(binary, format="PNG")
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(binary.getvalue())
    return image, cache, hashlib.sha256(binary.getvalue()).hexdigest()


def call_ocr(image: Path, key: str, model: str) -> dict:
    import requests
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    payload = {"model": model,
               "document": {"type": "image_url", "image_url": f"data:image/png;base64,{encoded}"},
               "table_format": "html", "include_blocks": True,
               "confidence_scores_granularity": "word"}
    response = requests.post(OCR_URL, json=payload,
                             headers={"Authorization": f"Bearer {key}"}, timeout=(20, 240))
    if not response.ok:
        raise RuntimeError(f"Mistral OCR HTTP {response.status_code}: {response.text[:700]}")
    return response.json()


def run(args: argparse.Namespace, root: Path = ROOT) -> None:
    if (args.year, args.page) != (1882, 245):
        raise ValueError("This first pilot is calibrated ONLY for 1882 PDF page 245. "
                         "Inspect and calibrate other pages before using them.")
    if args.command == "ocr" and not args.yes:
        raise ValueError("Paid experiment requires --yes")
    key = os.environ.get("MISTRAL_API_KEY", "") if args.command == "ocr" else ""
    if args.command == "ocr" and not key:
        raise ValueError("Set MISTRAL_API_KEY in your terminal (not in any project file)")
    for panel in args.panels:
        image, cache, digest = prepare(root, args.year, args.page, panel, args.dpi)
        print(f"Prepared {image} [sha256 {digest[:12]}]")
        if args.command != "ocr":
            continue
        old = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
        if not args.force and old.get("image_sha256") == digest and old.get("model") == MODEL:
            print(f"Cache hit: {cache}")
            continue
        response = call_ocr(image, key, MODEL)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"year": args.year, "pdf_page": args.page,
                                    "panel": panel, "model": MODEL,
                                    "source_image": str(image.relative_to(root)),
                                    "image_sha256": digest, "response": response},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved independent OCR result: {cache}")
    if args.command != "ocr":
        print("PREVIEW ONLY: no API calls made. Inspect these crops before running ocr.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preview", "ocr"))
    p.add_argument("--year", type=int, default=1882)
    p.add_argument("--page", type=int, default=245)
    p.add_argument("--panels", nargs="+", choices=tuple(PRESETS),
                   default=list(PRESETS))
    p.add_argument("--dpi", type=int, default=210)
    p.add_argument("--yes", action="store_true", help="Confirm up to two paid OCR requests")
    p.add_argument("--force", action="store_true", help="Rerun an already cached OCR request")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
