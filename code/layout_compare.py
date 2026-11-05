#!/usr/bin/env python3
"""Compare a pretrained table detector with volume-level table templates.

Never uses the old heuristic page/table splits. Both methods operate on the
*entire scanned PDF page*. The pretrained model is run once per image and its
boxes are cached. Templates are fitted using DEVELOPMENT pages only; they are
then transferred to unseen pages using normalized image coordinates plus a
bounded, conservative ruling-line registration.

This is a falsifiable *benchmark*, not a production segmenter. No OCR/API
requests, no handwritten page coordinates, no automatic changes to data.

Examples (from repository root):
    python code/layout_compare.py --plan-only
    python code/layout_compare.py --years 1878 1881 --per-year 5
    python code/layout_compare.py
    python code/layout_compare.py --score-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from layout_benchmark import ROOT, frozen_plan, _render

MODEL_ID = "microsoft/table-transformer-detection"
DEFAULT_OUTPUT = Path("derived/situation_financiere/layout_comparison_v2")
PLAN_DIR = Path("derived/situation_financiere/layout_benchmark")
FIELDS = (
    "year", "pdf_page", "role", "source_pdf", "page_shape", "model_count",
    "template_count", "template_status", "registration_status",
    "alignment_x", "alignment_y", "model_template_agreement",
    "proposed_gutter", "gutter_status", "overlay", "warning",
)
REVIEW_FIELDS = ("year", "pdf_page", "role", "overlay", "model_boxes_ok", "template_boxes_ok", "notes")
SCORE_VALUES = {"", "yes", "no", "na"}


class TableDetector(Protocol):
    def predict(self, image: Image.Image) -> list[dict]: ...


class TableTransformerDetector:
    """Optional Hugging Face model. Dependencies and weights loaded only at inference."""

    def __init__(self, model_id: str = MODEL_ID, threshold: float = .50, device: str = "cpu"):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as e:
            raise RuntimeError(
                "Pretrained model dependencies are absent. Install with: "
                "python -m pip install -r requirements-layout-model.txt"
            ) from e
        self.torch = torch
        self.threshold = threshold
        self.device = device
        self.model_id = model_id
        print(f"Loading table detector {model_id} on {device} (weights download on first use)", flush=True)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    def predict(self, image: Image.Image) -> list[dict]:
        torch = self.torch
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        det = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_sizes
        )[0]
        rows = []
        w, h = image.size
        for score, label, box in zip(det["scores"], det["labels"], det["boxes"]):
            label_name = str(self.model.config.id2label[int(label.item())]).lower()
            if "table" not in label_name:
                continue
            x0, y0, x1, y1 = [float(q) for q in box.tolist()]
            row = make_box([x0, y0, x1, y1], w, h, score=float(score.item()))
            if row is not None:
                rows.append(row)
        return deduplicate(rows)


def make_box(rect: list[float] | tuple[float, ...], w: int, h: int, score: float | None = None) -> dict | None:
    """Clamp to the source image; normalize for cross-page comparison."""
    if len(rect) != 4 or w <= 0 or h <= 0 or not all(math.isfinite(v) for v in rect):
        return None
    x0, y0, x1, y1 = rect
    x0, x1 = max(0., min(float(w), x0)), max(0., min(float(w), x1))
    y0, y1 = max(0., min(float(h), y0)), max(0., min(float(h), y1))
    if x1 - x0 < w * .025 or y1 - y0 < h * .035:
        return None
    return {
        "xyxy": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        "normalized": [round(x0/w, 6), round(y0/h, 6), round(x1/w, 6), round(y1/h, 6)],
        "score": round(float(score), 5) if score is not None else None,
    }


def box_iou(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = a["normalized"]
    bx0, by0, bx1, by1 = b["normalized"]
    area_a, area_b = (ax1-ax0)*(ay1-ay0), (bx1-bx0)*(by1-by0)
    intersection = max(0., min(ax1,bx1)-max(ax0,bx0))*max(0.,min(ay1,by1)-max(ay0,by0))
    return intersection / (area_a+area_b-intersection) if area_a+area_b-intersection > 0 else 0.


def deduplicate(boxes: list[dict], threshold: float = .8) -> list[dict]:
    kept = []
    for b in sorted(boxes, key=lambda r: r.get("score") or 0, reverse=True):
        if all(box_iou(b, previous) < threshold for previous in kept):
            kept.append(b)
    return sorted(kept, key=lambda b: (b["normalized"][0], b["normalized"][1]))


def proposed_gutter(boxes: list[dict], width: int, height: int) -> dict:
    """Only propose a spread gutter when detected TABLE REGIONS surround it.

    Unlike the previous algorithm, this never picks a vertical ruling line.
    If the pretrained detector sees an entire spread as one table, return
    unresolved; never use the central pixel as a fake answer.
    """
    if width / height < .96:
        return {"status": "single_page_by_aspect", "relative_x": None}
    if len(boxes) < 2:
        return {"status": "unresolved", "relative_x": None}
    best = None
    for left, right in zip(sorted(boxes, key=lambda b: b["normalized"][0])[:-1],
                           sorted(boxes, key=lambda b: b["normalized"][0])[1:]):
        gap_l = left["normalized"][2]
        gap_r = right["normalized"][0]
        gap = gap_r-gap_l
        mid = (gap_l+gap_r)/2
        if gap > .007 and .35 < mid < .65:
            candidate = (gap, mid)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        return {"status": "unresolved", "relative_x": None}
    return {"status": "table_gap_candidate", "relative_x": round(best[1], 5),
            "gap_fraction": round(best[0], 5)}


def page_shape(image: Image.Image) -> str:
    return "spread_like" if image.width / image.height >= .96 else "single_like"


def _json_write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv_write(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def _image_sha(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _cached_predict(image: Image.Image, detector: TableDetector, path: Path,
                    model_id: str, threshold: float, force: bool = False) -> dict:
    sha = _image_sha(image)
    if path.exists() and not force:
        old = json.loads(path.read_text(encoding="utf-8"))
        if (old.get("image_sha256") == sha and old.get("model_id") == model_id
                and old.get("threshold") == threshold):
            return old
    record = {"image_sha256": sha, "model_id": model_id, "threshold": threshold,
              "image_size": list(image.size), "boxes": detector.predict(image)}
    _json_write(path, record)
    return record


def _box_distance(a: dict, b: dict) -> float:
    x0,y0,x1,y1 = a["normalized"]
    X0,Y0,X1,Y1 = b["normalized"]
    return max(abs((x0+x1)-(X0+X1))/2, abs(x0-X0), abs(x1-X1),
               abs((y0+y1)-(Y0+Y1))/4)


def learn_volume_templates(samples: list[dict], records: dict[tuple[int,int], dict]) -> dict:
    """Learn repeated table locations using ONLY development detections.

    A template is NOT formed if detections are inconsistent. No holdout page
    can affect fitting, filtering or alignment reference selection.
    """
    buckets: dict[tuple[int, str], list[tuple[dict, dict]]] = defaultdict(list)
    for sample in samples:
        if sample["role"] != "development":
            continue
        key = int(sample["year"]), int(sample["pdf_page"])
        rec = records.get(key)
        if rec and rec["boxes"]:
            buckets[(key[0], rec["page_shape"])].append((sample, rec))
    templates: dict[str, dict] = {}
    for (year, kind), group in sorted(buckets.items()):
        slot = f"{year}:{kind}"
        # A template requires independent scanned pages; one-page calibration
        # would be a disguised manual exception.
        if len(group) < 2:
            templates[slot] = {"status": "unresolved_insufficient_development_pages",
                               "training_pages": [s["pdf_page"] for s, _ in group], "boxes": []}
            continue
        counts = Counter(len(r["boxes"]) for _, r in group)
        common_count, frequency = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))[0]
        qualifying = [(s,r) for s,r in group if len(r["boxes"]) == common_count]
        minimum_support = max(2, math.ceil(.6 * len(group)))
        if frequency < minimum_support:
            templates[slot] = {"status": "unresolved_inconsistent_table_counts",
                               "training_pages": [s["pdf_page"] for s,_ in group], "boxes": [],
                               "counts": dict(counts)}
            continue
        # Within any page, order by horizontal center; historic parallel
        # tables appear from left to right. Reject inconsistent placements.
        ordered = [sorted(rec["boxes"], key=lambda b: sum(b["normalized"][::2]))
                   for _, rec in qualifying]
        all_positions = np.array([[b["normalized"] for b in row] for row in ordered])
        medians = np.median(all_positions, axis=0)
        x_dispersion = float(np.max(np.ptp(all_positions[:, :, (0, 2)], axis=0)))
        if x_dispersion > .13:
            templates[slot] = {"status": "unresolved_unstable_horizontal_layout",
                               "training_pages": [s["pdf_page"] for s,_ in qualifying],
                               "boxes": [], "max_horizontal_dispersion": round(x_dispersion, 4)}
            continue
        template_boxes = []
        for box in medians:
            x0,y0,x1,y1 = [float(v) for v in box]
            if x1-x0 < .025 or y1-y0 < .035:
                continue
            template_boxes.append({"normalized": [round(v,6) for v in box],
                                   "xyxy": [], "score": None})
        templates[slot] = {"status": "fitted", "training_pages": [s["pdf_page"] for s,_ in qualifying],
                           "pages_available": len(group), "table_count": common_count,
                           "max_horizontal_dispersion": round(x_dispersion, 4),
                           "boxes": template_boxes,
                           "source": "pretrained detections on DEVELOPMENT pages only (template-stabilized)"}
    return templates


def _stable_rule_profiles(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Ruling-line projections for cautious image registration (not segmentation).

    A pattern must repeat across a volume for this to be useful. If the line
    evidence is weak, the template will not be transferred as if aligned.
    """
    gray = np.asarray(image.convert("L"))
    h, w = gray.shape
    # Fixed-size lower-resolution mask only for estimating small translations.
    scale = 900. / max(w, h)
    sw, sh = max(30, round(w*scale)), max(30, round(h*scale))
    reduced = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    ink = cv2.adaptiveThreshold(reduced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 29, 12)
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(24,sh//11))))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (max(30,sw//6), 1)))
    # Profiles from the whole page; no manually selected crop coordinates.
    xprof = (vertical > 0).sum(axis=0).astype(np.float64)
    yprof = (horizontal > 0).sum(axis=1).astype(np.float64)
    return xprof, yprof


def _offset(ref: np.ndarray, target: np.ndarray, max_shift_fraction: float=.05) -> tuple[float, str]:
    if len(ref) != len(target) or len(ref) < 40:
        return 0., "unresolved_size"
    if np.count_nonzero(ref) < 3 or np.count_nonzero(target) < 3:
        return 0., "unresolved_no_rules"
    ref = ref-ref.mean()
    target = target-target.mean()
    std = np.linalg.norm(ref)*np.linalg.norm(target)
    if std <= 1e-10:
        return 0., "unresolved_no_rules"
    radius = max(1, round(len(ref)*max_shift_fraction))
    choices = []
    for shift in range(-radius, radius+1):
        if shift >= 0:
            a, b = ref[:len(ref)-shift], target[shift:]
        else:
            a, b = ref[-shift:], target[:len(target)+shift]
        value = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
        choices.append((value, shift))
    best_score, shift = max(choices)
    # Correlation alone can be misleading on largely blank scans. Fail closed.
    if best_score < .45:
        return 0., "unresolved_weak_match"
    if abs(shift) >= radius:
        return 0., "unresolved_boundary_peak"
    return shift/len(ref), "matched"


def transfer_template(template: dict | None, target: dict, ref_profiles: tuple[np.ndarray,np.ndarray] | None,
                      target_profiles: tuple[np.ndarray,np.ndarray] | None) -> tuple[list[dict], str, float, float]:
    if not template or template.get("status") != "fitted":
        return [], "unresolved_template", 0., 0.
    if ref_profiles is None or target_profiles is None:
        return [], "unresolved_registration", 0., 0.
    dx, sx = _offset(ref_profiles[0], target_profiles[0])
    dy, sy = _offset(ref_profiles[1], target_profiles[1])
    if sx != "matched" or sy != "matched":
        return [], f"unresolved_registration_x_{sx}_y_{sy}", dx, dy
    w,h = target["image_size"]
    boxes = []
    for source in template["boxes"]:
        x0,y0,x1,y1 = source["normalized"]
        rect = [(x0+dx)*w,(y0+dy)*h,(x1+dx)*w,(y1+dy)*h]
        item = make_box(rect, w, h)
        if item:
            boxes.append(item)
    return boxes, "transferred", dx, dy


def box_agreement(a: list[dict], b: list[dict], threshold: float=.5) -> float | None:
    """Correspondence, NOT accuracy: both methods may make the same mistake."""
    if not a or not b:
        return None
    remaining = set(range(len(b)))
    matched = 0
    for target in sorted(a, key=lambda x: x.get("score") or 0, reverse=True):
        if not remaining:
            break
        idx = max(remaining, key=lambda i: box_iou(target,b[i]))
        if box_iou(target,b[idx]) >= threshold:
            matched += 1
            remaining.remove(idx)
    return round(matched/max(len(a),len(b)), 4)


def overlay(image: Image.Image, model_boxes: list[dict], template_boxes: list[dict],
            gutter: dict, *, label: str) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    line = max(2, int(min(image.size)/380))
    for item in model_boxes:
        xy = [round(v) for v in item["xyxy"]]
        draw.rectangle(xy, outline=(4,145,220), width=line)
    for item in template_boxes:
        xy = [round(v) for v in item["xyxy"]]
        draw.rectangle(xy, outline=(220,120,0), width=line)
    if gutter.get("relative_x") is not None:
        x = round(float(gutter["relative_x"])*out.width)
        draw.line([(x,0),(x,out.height)], fill=(130,50,185), width=line)
    draw.rectangle((0,0,min(out.width, max(250, len(label)*7+10)),20), fill=(255,255,255))
    draw.text((5,4), label, fill=(0,0,0))
    return out


def _contact_sheet(output: Path, shortlist: list[dict], columns: int=2) -> None:
    if not shortlist:
        return
    thumb_w, thumb_h = 540, 680
    margin, head = 16, 32
    nrows = math.ceil(len(shortlist)/columns)
    canvas = Image.new("RGB", (columns*(thumb_w+margin)+margin, nrows*(thumb_h+head+margin)+margin),"white")
    d = ImageDraw.Draw(canvas)
    for idx,row in enumerate(shortlist):
        with Image.open(output / row["overlay"]) as orig:
            thumb = orig.copy()
        thumb.thumbnail((thumb_w,thumb_h),Image.Resampling.LANCZOS)
        x = margin+(idx%columns)*(thumb_w+margin)
        y = margin+(idx//columns)*(thumb_h+head+margin)
        d.text((x,y), f'{row["year"]} p{int(row["pdf_page"]):04d} - {row["role"]}', fill=(0,0,0))
        canvas.paste(thumb,(x,y+head))
    canvas.save(output / "review_contact_sheet.jpg", quality=87)


def _shortlist(records: list[dict], output: Path) -> list[dict]:
    """At most one holdout example per year; no 40-page manual review request."""
    result = []
    for year in sorted({int(r["year"]) for r in records}):
        group = [r for r in records if int(r["year"]) == year]
        priority = sorted(group, key=lambda r: (r["role"] != "holdout", r["pdf_page"]))
        if not priority:
            continue
        row = priority[0]
        result.append({key: row.get(key, "") for key in ("year","pdf_page","role","overlay")})
    path = output / "review_shortlist.csv"
    if path.exists():
        with path.open(encoding="utf-8",newline="") as f:
            old = list(csv.DictReader(f))
        old_keys = [(int(x["year"]),int(x["pdf_page"])) for x in old]
        new_keys = [(int(x["year"]),int(x["pdf_page"])) for x in result]
        if old_keys != new_keys:
            raise ValueError("Review shortlist would change. Rename the old review CSV before altering the plan.")
        result = old
    else:
        _csv_write(path, REVIEW_FIELDS, result)
    _contact_sheet(output, result)
    return result


def score_shortlist(output: Path) -> dict:
    path = output / "review_shortlist.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8",newline="") as f:
        rows = list(csv.DictReader(f))
    report = []
    for method in ("model_boxes_ok", "template_boxes_ok"):
        values = [str(x.get(method, "")).strip().lower() for x in rows]
        if any(v not in SCORE_VALUES for v in values):
            raise ValueError(f"Invalid {method}; use yes, no, na, or blank")
        yes, no = values.count("yes"), values.count("no")
        report.append({"method": method, "checked": yes+no, "correct": yes,
                       "incorrect": no, "unreviewed": values.count(""),
                       "not_applicable": values.count("na"),
                       "accuracy": round(yes/(yes+no), 4) if yes+no else ""})
    _csv_write(output / "review_scores.csv",
               ("method","checked","correct","incorrect","unreviewed","not_applicable","accuracy"), report)
    return {"reviews":len(rows),"score_report":str(output / "review_scores.csv")}


def evaluate_plan(repo: Path, output: Path, plan: dict, *, dpi: int, threshold: float,
                  model_id: str, detector: TableDetector, force: bool=False,
                  max_pages: int | None=None) -> dict:
    """Run direct pretrained detection; learn only from development; test holdouts.

    Cache keyed by image SHA + model and threshold. The benchmark will stop with
    an explicit error if a source PDF is unavailable, not silently invent scores.
    """
    import fitz
    if not 72 <= dpi <= 300:
        raise ValueError("dpi must be between 72 and 300")
    selected = list(plan["samples"])
    if max_pages is not None:
        if max_pages < 1:
            raise ValueError("--max-pages must be positive")
        selected = selected[:max_pages]
    pdf_dir = repo / "raw_data" / "situation_financiere_communes"
    missing = sorted({str(pdf_dir / s["source_pdf"]) for s in selected
                      if not (pdf_dir / s["source_pdf"]).is_file()})
    if missing:
        raise FileNotFoundError("Missing PDFs:\n"+"\n".join(missing))
    by_year = defaultdict(list)
    for s in selected:
        by_year[int(s["year"])].append(s)
    recs: dict[tuple[int,int],dict] = {}
    images: dict[tuple[int,int],Image.Image] = {}
    for year in sorted(by_year):
        with fitz.open(pdf_dir / by_year[year][0]["source_pdf"]) as pdf:
            for s in by_year[year]:
                page = int(s["pdf_page"])
                print(f"{year} PDF page {page}: pretrained detector", flush=True)
                image = _render(pdf,page,dpi).convert("RGB")
                cached = _cached_predict(image, detector,
                                         output/"model_cache"/str(year)/f"p{page:04d}.json",
                                         model_id, threshold, force)
                cached["page_shape"] = page_shape(image)
                cached["image_size"] = list(image.size)
                recs[(year,page)] = cached
                images[(year,page)] = image
    templates = learn_volume_templates(selected,recs)
    _json_write(output/"templates.json", templates)
    # Registration references: medoid-development page from the pages actually
    # supporting the template; no holdout data used to fit the template itself.
    reference: dict[str,tuple[np.ndarray,np.ndarray]] = {}
    for slot,tpl in templates.items():
        if tpl["status"] != "fitted":
            continue
        year,kind = slot.split(":")
        pages = tpl["training_pages"]
        anchor = (int(year),int(pages[len(pages)//2]))
        if anchor in images:
            reference[slot] = _stable_rule_profiles(images[anchor])
    rows=[]
    for s in selected:
        key=(int(s["year"]),int(s["pdf_page"]))
        rec=recs[key]
        image=images[key]
        slot=f'{s["year"]}:{rec["page_shape"]}'
        template=templates.get(slot)
        template_boxes,registration_status,dx,dy = transfer_template(
            template,rec,reference.get(slot),_stable_rule_profiles(image)
        )
        model_boxes=rec["boxes"]
        agreement=box_agreement(model_boxes,template_boxes)
        gutter=proposed_gutter(model_boxes,image.width,image.height)
        stem=f'{s["year"]}_p{int(s["pdf_page"]):04d}'
        op=output/"overlays"/str(s["year"])/f'{stem}.jpg'
        op.parent.mkdir(parents=True,exist_ok=True)
        overlay(image,model_boxes,template_boxes,gutter,
                label=f'{stem} Blue=model Orange=template Purple=gutter').save(op,quality=88)
        warning=[]
        if not model_boxes: warning.append("no_model_tables")
        if registration_status != "transferred": warning.append(registration_status)
        if gutter["status"] == "unresolved": warning.append("gutter_unresolved")
        row={"year":s["year"],"pdf_page":s["pdf_page"],"role":s["role"],
             "source_pdf":s["source_pdf"],"page_shape":rec["page_shape"],
             "model_count":len(model_boxes),"template_count":len(template_boxes),
             "template_status":template["status"] if template else "no_template",
             "registration_status":registration_status,"alignment_x":round(dx,5),
             "alignment_y":round(dy,5),"model_template_agreement":agreement if agreement is not None else "",
             "proposed_gutter":gutter.get("relative_x") if gutter.get("relative_x") is not None else "",
             "gutter_status":gutter["status"],"overlay":str(op.relative_to(output)),
             "warning":";".join(warning)}
        rows.append(row)
        _json_write(output/"detections"/str(s["year"])/f'{stem}.json',
                    {"model_boxes":model_boxes,"template_boxes":template_boxes,
                     "gutter":gutter,"registration":registration_status,"role":s["role"]})
    _csv_write(output/"comparison_manifest.csv",FIELDS,rows)
    review=_shortlist(rows,output)
    summary={"pages":len(rows),"years":len(set(x["year"] for x in rows)),
             "model_pages_with_boxes":sum(x["model_count"]>0 for x in rows),
             "template_pages_transferred":sum(x["template_count"]>0 for x in rows),
             "gutter_candidates":sum(x["gutter_status"]=="table_gap_candidate" for x in rows),
             "review_shortlist_pages":len(review),
             "accuracy":None,
             "note":"Agreement is not accuracy. No page or table coordinates are trusted until reviewed."}
    _json_write(output/"comparison_summary.json",summary)
    return summary


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--source-plan",type=Path,default=PLAN_DIR/"sample_plan.json",
                        help="Use the frozen benchmark plan from layout_benchmark.py")
    parser.add_argument("--years",type=int,nargs="+",default=list(range(1878,1886)))
    parser.add_argument("--per-year",type=int,default=5)
    parser.add_argument("--seed",type=int,default=1882)
    parser.add_argument("--reset-plan",action="store_true")
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--score-only",action="store_true")
    parser.add_argument("--model-id",default=MODEL_ID)
    parser.add_argument("--threshold",type=float,default=.50)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--dpi",type=int,default=135)
    parser.add_argument("--force-model",action="store_true")
    parser.add_argument("--max-pages",type=int,default=None,
                        help="Smoke-test only; omit for full eight-year run")
    args=parser.parse_args(argv)
    repo=args.repo_root.resolve()
    output=args.output if args.output.is_absolute() else repo/args.output
    if args.score_only:
        print(json.dumps(score_shortlist(output),indent=2))
        return 0
    config=repo/"schema"/"situation_financiere"/"page_ranges.json"
    plan_path=args.source_plan if args.source_plan.is_absolute() else repo/args.source_plan
    if plan_path.is_file() and not args.reset_plan:
        plan=json.loads(plan_path.read_text(encoding="utf-8"))
        if sorted(set(args.years)) != sorted(plan.get("years",[])):
            parser.error("Existing frozen plan has different years; supply --years accordingly ")
        print(f"Reusing frozen sample from {plan_path}")
    else:
        plan=frozen_plan(config,output,years=sorted(set(args.years)),
                         per_year=args.per_year,seed=args.seed,reset=args.reset_plan)
    if args.plan_only:
        print(f"{len(plan['samples'])} PDF pages in frozen plan")
        for sample in plan["samples"]:
            print(f'  {sample["year"]} p{sample["pdf_page"]}: {sample["role"]}')
        return 0
    if not 0.05 <= args.threshold <= .95:
        parser.error("--threshold must be between 0.05 and 0.95")
    # Preflight missing PDF before downloading a model.
    selected=plan["samples"][:args.max_pages] if args.max_pages else plan["samples"]
    pdf_root=repo/"raw_data"/"situation_financiere_communes"
    missing=sorted({str(pdf_root/s["source_pdf"]) for s in selected
                    if not (pdf_root/s["source_pdf"]).is_file()})
    if missing:
        parser.error("Source PDFs missing:\n"+"\n".join(missing))
    detector=TableTransformerDetector(model_id=args.model_id,threshold=args.threshold,device=args.device)
    print(json.dumps(evaluate_plan(repo,output,plan,dpi=args.dpi,
                                   threshold=args.threshold,model_id=args.model_id,
                                   detector=detector,force=args.force_model,max_pages=args.max_pages),indent=2))
    print("Review review_contact_sheet.jpg and review_shortlist.csv; no OCR was run.")
    return 0

if __name__=="__main__":
    sys.exit(main())
