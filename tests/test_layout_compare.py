"""Offline unit and integration tests: no model download or Mistral calls."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from layout_compare import (
    _cached_predict, _offset, _shortlist, _stable_rule_profiles,
    box_agreement, box_iou, deduplicate, evaluate_plan, learn_volume_templates,
    make_box, page_shape, proposed_gutter, score_shortlist,
    transfer_template,
)


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        w, h = image.size
        return [make_box((.06*w, .32*h, .46*w, .90*h), w,h, .99),
                make_box((.54*w, .32*h, .94*w, .90*h), w,h, .99)]


def sample(year, page, role="development"):
    return {"year": year, "pdf_page": page, "role": role,
            "source_pdf": "example.pdf", "allowed_sides": ["left", "right"]}


def rec(x=0, shape="single_like", width=1000, height=1500):
    return {"page_shape": shape, "image_size": [width, height], "boxes": [
        make_box((60+x,400,460+x,1370),width,height,.9),
        make_box((540+x,400,940+x,1370),width,height,.95)]}


def test_boxes_dedup_and_iou_and_no_invented_gutter():
    a=make_box((0,0,490,800),1000,1000,.95)
    close=make_box((2,2,492,802),1000,1000,.85)
    other=make_box((520,0,990,800),1000,1000,.95)
    assert len(deduplicate([a,close,other])) == 2
    assert box_iou(a,close) > .97
    assert box_agreement([a,other],[close,other]) == 1
    assert proposed_gutter([a],1000,800)["status"] == "unresolved"
    assert proposed_gutter([a,other],1000,800)["status"] == "table_gap_candidate"
    assert proposed_gutter([a,other],1000,1700)["status"] == "single_page_by_aspect"
    assert make_box((100,100,99,199),1000,1000) is None


def test_training_uses_development_only():
    samples = [sample(1881,1),sample(1881,2),sample(1881,3,"holdout")]
    records = {(1881,1): rec(0), (1881,2): rec(12), (1881,3): rec(350)}
    templates=learn_volume_templates(samples,records)
    t=templates["1881:single_like"]
    assert t["status"]=="fitted"
    assert set(t["training_pages"]) == {1,2}
    assert t["table_count"]==2
    assert t["boxes"][0]["normalized"][0] < .08


def test_mismatched_templates_fail_closed():
    samples = [sample(1881,1),sample(1881,2),sample(1881,3)]
    a,b,c=rec(),rec(),rec()
    b["boxes"] = b["boxes"][:1]
    c["boxes"]=c["boxes"]+ [make_box((100,900,200,1000),1000,1500,.7)]
    result=learn_volume_templates(samples, {(1881,1):a,(1881,2):b,(1881,3):c})
    assert result["1881:single_like"]["status"] == "unresolved_inconsistent_table_counts"
    assert result["1881:single_like"]["boxes"] == []


def test_template_cannot_transfer_without_registration():
    tpl={"status":"fitted", "boxes":[{"normalized":[.1,.2,.4,.8]}]}
    boxes,status,dx,dy=transfer_template(tpl,{"image_size":[1000,1000]},None,None)
    assert boxes==[] and status=="unresolved_registration"
    assert dx==dy==0


def test_registration_projections_can_detect_shift_and_fail_on_empty():
    ref=np.zeros(100)
    ref[[16,36,78]]=1
    target=np.roll(ref,2)
    offset,status=_offset(ref,target,max_shift_fraction=.1)
    assert status=="matched" and offset==pytest.approx(.02)
    _,status2=_offset(np.zeros(100),target)
    assert status2=="unresolved_no_rules"


def test_predictions_cache_keyed_by_pixels_model_and_threshold(tmp_path):
    im=Image.new("RGB",(1000,1300),"white")
    detector=FakeDetector()
    path=tmp_path/"cache"/"test.json"
    a=_cached_predict(im,detector,path,"fake",.5)
    b=_cached_predict(im,detector,path,"fake",.5)
    assert a==b and detector.calls==1
    _cached_predict(im,detector,path,"fake",.6)
    assert detector.calls==2
    im.putpixel((3,3),(0,0,0))
    _cached_predict(im,detector,path,"fake",.6)
    assert detector.calls==3


def test_shortlist_one_holdout_per_year_and_score_preserves_edits(tmp_path):
    rows=[{"year":1881,"pdf_page":1,"role":"development","overlay":"o1.jpg"},
          {"year":1881,"pdf_page":2,"role":"holdout","overlay":"o2.jpg"},
          {"year":1882,"pdf_page":3,"role":"holdout","overlay":"o3.jpg"}]
    for row in rows:
        Image.new("RGB", (300,250), "white").save(tmp_path/row["overlay"])
    chosen=_shortlist(rows,tmp_path)
    assert [(r["year"],r["pdf_page"]) for r in chosen] == [(1881,2),(1882,3)]
    assert (tmp_path/"review_contact_sheet.jpg").exists()
    path=tmp_path/"review_shortlist.csv"
    with path.open(encoding="utf-8",newline="") as f:
        entries=list(csv.DictReader(f))
    entries[0]["model_boxes_ok"]="no"
    entries[1]["model_boxes_ok"]="yes"
    entries[0]["template_boxes_ok"]="na"
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=entries[0].keys());w.writeheader();w.writerows(entries)
    _shortlist(rows,tmp_path)
    result=score_shortlist(tmp_path)
    assert result["reviews"]==2
    scores=list(csv.DictReader((tmp_path/"review_scores.csv").open()))
    assert scores[0]["accuracy"]=="0.5"
    with pytest.raises(ValueError, match="Review shortlist would change"):
        _shortlist(rows[:1],tmp_path)


def test_pdf_end_to_end_with_fake_detector(tmp_path):
    fitz=pytest.importorskip("fitz")
    repo=tmp_path/"repo"
    folder=repo/"raw_data"/"situation_financiere_communes"
    folder.mkdir(parents=True)
    with fitz.open() as pdf:
        for p in range(4):
            page=pdf.new_page(width=600,height=860)
            # Same vertical and horizontal rules every page so registration
            # has actual independent geometric evidence.
            for x in (35,275,325,565):
                page.draw_line((x,180),(x,800),color=(0,0,0),width=2)
            for y in (180,400,800):
                page.draw_line((30,y),(570,y),color=(0,0,0),width=2)
            page.insert_text((60,230),f"Page {p+1}")
        pdf.save(folder/"example.pdf")
    plan={"samples":[sample(1881,1),sample(1881,2),
                     sample(1881,3,"holdout"),sample(1881,4,"holdout")]}
    out=tmp_path/"result"
    detector=FakeDetector()
    summary=evaluate_plan(repo,out,plan,dpi=130,threshold=.5,model_id="fake",
                          detector=detector)
    assert summary["pages"]==4
    assert summary["model_pages_with_boxes"]==4
    assert detector.calls==4
    assert (out/"templates.json").exists()
    assert (out/"comparison_manifest.csv").exists()
    assert (out/"review_contact_sheet.jpg").exists()
    assert (out/"review_shortlist.csv").exists()
    assert (out/"model_cache"/"1881"/"p0001.json").exists()
    # Test no paid calls / no new predictions on cached second execution.
    evaluate_plan(repo,out,plan,dpi=130,threshold=.5,model_id="fake",detector=detector)
    assert detector.calls==4
