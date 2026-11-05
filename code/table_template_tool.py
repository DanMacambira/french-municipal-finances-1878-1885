#!/usr/bin/env python3
"""Local, manual *reference-page* table-template annotator.

Use from project root:
    .venv/bin/python code/table_template_tool.py --year 1882 --page 245

Draw bounding boxes around complete parallel financial tables on one reference
PDF scan in the browser. The tool stores normalized coordinates in schema/ and
creates diagnostic *unverified* overlays for pages from the same year's range.

NO OCR, NO model, NO external API, NO editing of source PDFs. The current transfer
is normalized-position only: it does NOT assert that other pages are aligned.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import secrets
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont
from table_template_alignment import align_tables

ROOT = Path(__file__).resolve().parents[1]
RANGES = ROOT / "schema" / "situation_financiere" / "page_ranges.json"
TEMPLATES = ROOT / "schema" / "situation_financiere" / "table_templates.json"
PDF_DIR = ROOT / "raw_data" / "situation_financiere_communes"
OUT = ROOT / "derived" / "situation_financiere" / "reference_templates"


def config_for_year(year: int, ranges: Path = RANGES, pdf_dir: Path = PDF_DIR):
    settings = json.loads(ranges.read_text(encoding="utf-8"))
    spec = settings.get(str(year))
    if not spec:
        raise ValueError(f"Year {year} is absent from {ranges}")
    pdf = pdf_dir / spec["pdf"]
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    if not spec.get("confirmed"):
        raise ValueError(f"Year {year}: page range not confirmed")
    start, end = int(spec["start"]), int(spec["end"])
    if start < 1 or end < start:
        raise ValueError(f"Invalid page range: {start}..{end}")
    return spec, pdf


def render_page(pdf: Path, page_no: int, dpi: int = 125):
    with fitz.open(pdf) as doc:
        if not 1 <= page_no <= len(doc):
            raise ValueError(f"Page {page_no} outside PDF (1..{len(doc)})")
        p = doc[page_no - 1]
        pix = p.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), colorspace=fitz.csRGB, alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return image


def validate_boxes(boxes):
    if not isinstance(boxes, list) or not 1 <= len(boxes) <= 8:
        raise ValueError("Draw between 1 and 8 complete table rectangles")
    cleaned = []
    for i, box in enumerate(boxes, 1):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"Invalid rectangle #{i}")
        if not all(type(v) in (float, int) for v in box):
            raise ValueError("Coordinates must be numbers")
        x0, y0, x1, y1 = map(float, box)
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"Rectangle #{i} must stay inside the scan")
        if x1 - x0 < 0.03 or y1 - y0 < 0.05:
            raise ValueError(f"Rectangle #{i} is too small; draw the whole table")
        cleaned.append([round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)])
    return sorted(cleaned, key=lambda b: (b[0], b[1]))


def save_reference(year: int, page: int, pdf_name: str, boxes, path: Path = TEMPLATES):
    boxes = validate_boxes(boxes)
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
    else:
        doc = {"version": 1, "templates": {}}
    if doc.get("version") != 1 or not isinstance(doc.get("templates"), dict):
        raise ValueError(f"Unsupported template format in {path}")
    doc["templates"][str(year)] = {
        "pdf": pdf_name,
        "reference_page": int(page),
        "coordinate_system": "relative_to_entire_pdf_scan",
        "transfer_method": "normalized_only_unverified",
        "regions": [
            {"name": f"table_{i}", "box": box}
            for i, box in enumerate(boxes, 1)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return doc["templates"][str(year)]


def px_box(box, size):
    w, h = size
    return tuple(round(box[i] * (w if i % 2 == 0 else h)) for i in range(4))


def side_is_allowed(box, page: int, spec):
    """Skip excluded sides of a spread only where the configuration specifies it."""
    sides = spec.get("page_sides", {}).get(str(page))
    if not sides:
        return True
    midpoint = (box[0] + box[2]) / 2
    return ("left" in sides if midpoint < 0.5 else "right" in sides)


def draw_overlay(image: Image.Image, boxes, page: int, year: int, status: str):
    image = image.copy().convert("RGB")
    brush = ImageDraw.Draw(image)
    # Do not modify original scan. All markings are only for visual QA.
    for idx, box in enumerate(boxes, 1):
        coords = px_box(box, image.size)
        width = max(3, image.width // 340)
        brush.rectangle(coords, outline=(235, 38, 46), width=width)
        lab = f"TABLE {idx}"
        x, y = coords[:2]
        pad = 4
        brush.rectangle((x, max(0, y - 23), x + 96, y + 1), fill=(235, 38, 46))
        brush.text((x + pad, max(0, y - 21)), lab, fill="white")
    brush.rectangle((0, 0, min(image.width, 600), 24), fill=(18, 48, 69))
    brush.text((6, 4), f"{year} PDF p{page} | {status}", fill="white")
    return image


def generate_previews(year: int, spec: dict, pdf: Path, template: dict,
                      output_root: Path = OUT, dpi: int = 90, align: bool = True):
    """Preview translated boxes; uncertain predictions remain unverified."""
    dest = output_root / str(year)
    overlay_dir = dest / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    start, end = int(spec["start"]), int(spec["end"])
    ref = int(template["reference_page"])
    regions = template["regions"]
    reference_image = render_page(pdf, ref, dpi=dpi) if align else None
    rows = []
    with fitz.open(pdf) as doc:
        if end > len(doc):
            raise ValueError(f"Configured page range ends at {end}, but PDF has only {len(doc)} pages")
        for page in range(start, end + 1):
            image = render_page(pdf, page, dpi=dpi)
            boxes = [r["box"] for r in regions if side_is_allowed(r["box"], page, spec)]
            status = "reference" if page == ref else "UNVERIFIED POSITION TRANSFER"
            matches = []
            if align and page != ref and boxes:
                matches = align_tables(reference_image, image, boxes)
                boxes = [m.box for m in matches]
                status = "PROVISIONAL ALIGNED / REVIEW"
            out = overlay_dir / f"{year}_p{page:04d}_overlay.jpg"
            draw_overlay(image, boxes, page, year, status).save(out, quality=85)
            for i, box in enumerate(boxes):
                m = matches[i] if matches else None
                rows.append({
                    "year": year, "pdf_page": page, "reference_page": ref,
                    "table": i+1, "status": m.status if m else status,
                    "match_score": round(m.score, 4) if m else "",
                    "peak_gap": round(m.peak_gap, 4) if m else "",
                    "offset_x": m.offset_x if m else "",
                    "offset_y": m.offset_y if m else "",
                    "box": json.dumps(box),
                    "overlay": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
                    "visually_correct": "",
                })
    csv_path = dest / "review.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return csv_path, overlay_dir, len(set(r['pdf_page'] for r in rows))


def html_page(year: int, page: int, width: int, height: int, token: str, saved=None):
    # The UI never needs third-party JS or network requests.
    initial = json.dumps([r["box"] for r in saved["regions"]] if saved else [])
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Table template — {year} PDF p{page}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif; margin:16px; color:#202c35;}}
h1{{font-size:20px; margin-bottom:4px}}p{{line-height:1.45}}
button{{padding:9px 16px; margin:5px 8px 5px 0; cursor:pointer;}}
#frame{{position:relative; width:min(96vw,1400px); box-shadow:0 1px 8px #adb5bd;}}
#scan{{width:100%;height:auto;display:block;user-select:none;}}
#canvas{{position:absolute;left:0;top:0;width:100%;height:100%;touch-action:none;cursor:crosshair;}}
#message{{font-weight:600;white-space:pre-wrap}}.note{{color:#51616c}}
</style>
<h1>Year {year} — PDF page {page}</h1>
<p><strong>Draw one rectangle around each COMPLETE financial table.</strong> Include column headings and all commune rows. You are not drawing each individual column or tax brace.<br>
For a single printed page, that usually means <strong>2 rectangles</strong>; for a two-page spread, usually <strong>4 rectangles</strong>.</p>
<div><button id="undo">Undo last</button><button id="clear">Clear</button><button id="save">Save template and preview the year</button></div>
<p class="note">Drag from the top-left to bottom-right of a table. Red boxes are provisional. Existing PDF files and OCR outputs are never changed.</p>
<p id="message">Tables drawn: 0</p>
<div id="frame"><img id="scan" src="/image" width="{width}" height="{height}" alt="Scanned PDF page"><canvas id="canvas" width="{width}" height="{height}"></canvas></div>
<script>
const token={json.dumps(token)};
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),msg=document.getElementById('message');
let rects={initial}, start=null, preview=null;
function pos(e){{let b=canvas.getBoundingClientRect();return {{x:(e.clientX-b.left)/b.width,y:(e.clientY-b.top)/b.height}};}}
function norm(r){{return [Math.max(0,Math.min(r[0],r[2])),Math.max(0,Math.min(r[1],r[3])),Math.min(1,Math.max(r[0],r[2])),Math.min(1,Math.max(r[1],r[3]))];}}
function render(){{
 ctx.clearRect(0,0,canvas.width,canvas.height);
 let arr=preview?[...rects,preview]:rects;
 arr.forEach((r,i)=>{{let q=norm(r);ctx.strokeStyle='#e52429';ctx.lineWidth=Math.max(3,canvas.width/400);
 let x=q[0]*canvas.width,y=q[1]*canvas.height,w=(q[2]-q[0])*canvas.width,h=(q[3]-q[1])*canvas.height;
 ctx.strokeRect(x,y,w,h);ctx.fillStyle='#e52429';ctx.fillRect(x,Math.max(0,y-22),105,22);
 ctx.fillStyle='white';ctx.font='bold 17px Arial';ctx.fillText('TABLE '+(i+1),x+4,Math.max(16,y-5));}});
 msg.textContent='Tables drawn: '+rects.length;
}}
canvas.addEventListener('pointerdown',e=>{{start=pos(e);preview=null;canvas.setPointerCapture(e.pointerId);}});
canvas.addEventListener('pointermove',e=>{{if(!start)return;let p=pos(e);preview=[start.x,start.y,p.x,p.y];render();}});
canvas.addEventListener('pointerup',e=>{{if(!start)return;let p=pos(e),r=norm([start.x,start.y,p.x,p.y]);
 if(r[2]-r[0]>=.03&&r[3]-r[1]>=.05)rects.push(r); start=null;preview=null;render();}});
document.getElementById('undo').onclick=()=>{{rects.pop();render();}};
document.getElementById('clear').onclick=()=>{{rects=[];render();}};
document.getElementById('save').onclick=async()=>{{
 msg.textContent='Saving and generating overlays…';
 try{{let r=await fetch('/save',{{method:'POST',headers:{{'Content-Type':'application/json','X-Session-Token':token}},body:JSON.stringify({{boxes:rects}})}});
 let data=await r.json();if(!r.ok)throw new Error(data.error||'Unknown error');
 msg.textContent='SAVED. '+data.message+'\\nThe year overlays are in: '+data.overlays+'\\nYou may close this browser tab and press Ctrl+C in Terminal.';
 }}catch(e){{msg.textContent='ERROR: '+e.message;}}
}};
render();
</script></html>'''


def run_server(year: int, page: int, spec: dict, pdf: Path, port: int):
    img = render_page(pdf, page, dpi=130)
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_bytes = img_buf.getvalue()
    token = secrets.token_urlsafe(24)
    existing = None
    if TEMPLATES.exists():
        doc = json.loads(TEMPLATES.read_text(encoding="utf-8"))
        existing = doc.get("templates", {}).get(str(year))
        if existing and existing.get("reference_page") != page:
            existing = None
    html = html_page(year, page, *img.size, token=token, saved=existing).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data, mime="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            route = urlparse(self.path).path
            if route in ("/", "/index.html"):
                self.reply(200, html, "text/html; charset=utf-8")
            elif route == "/image":
                self.reply(200, img_bytes, "image/png")
            else:
                self.reply(404, b"Not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/save":
                self.reply(404, b'{"error":"Not found"}')
                return
            if self.headers.get("X-Session-Token") != token:
                self.reply(403, b'{"error":"Invalid session"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not (0 < length < 12000):
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                template = save_reference(year, page, pdf.name, payload.get("boxes"))
                review, overlay_dir, count = generate_previews(year, spec, pdf, template)
                msg = {"message": f"Created {count} diagnostic overlays; these are NOT verified crops.",
                       "overlays": str(overlay_dir.relative_to(ROOT)),
                       "review_csv": str(review.relative_to(ROOT))}
                self.reply(200, json.dumps(msg).encode("utf-8"))
                print(f"\nSaved template: {TEMPLATES}\nPreview overlays: {overlay_dir}\nReview sheet: {review}", flush=True)
            except (ValueError, KeyError, OSError, fitz.FileDataError) as exc:
                self.reply(400, json.dumps({"error": str(exc)}).encode("utf-8"))

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"\nOpen this address in your browser: {url}\nDraw 2 or 4 TABLE rectangles; click Save. Ctrl+C to stop.\n", flush=True)
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True, help="Year 1878..1885")
    parser.add_argument("--page", type=int, default=None, help="One-based PDF page for the reference scan")
    parser.add_argument("--port", type=int, default=0, help="Local port (0 chooses an available one)")
    parser.add_argument("--preview-only", action="store_true", help="Recreate overlays from a previously saved template")
    parser.add_argument("--no-align", action="store_true", help="Disable provisional header-based registration")
    args = parser.parse_args(argv)
    spec, pdf = config_for_year(args.year)
    if args.preview_only:
        if not TEMPLATES.exists():
            parser.error(f"Missing template file: {TEMPLATES}")
        doc = json.loads(TEMPLATES.read_text(encoding="utf-8"))
        template = doc.get("templates", {}).get(str(args.year))
        if not template:
            parser.error(f"No template saved for year {args.year}")
        review, overlays, count = generate_previews(args.year, spec, pdf, template, align=not args.no_align)
        print(f"Created {count} UNVERIFIED overlays: {overlays}\nReview CSV: {review}")
        return
    page = args.page or (int(spec["start"]) + int(spec["end"])) // 2
    run_server(args.year, page, spec, pdf, args.port)


if __name__ == "__main__":
    main()
