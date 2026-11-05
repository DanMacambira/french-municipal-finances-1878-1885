import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))

import numpy as np
from PIL import Image, ImageDraw
from table_template_alignment import align_tables


def sample(shift=0):
    img=Image.new('RGB',(800,500),'white');d=ImageDraw.Draw(img)
    for x0 in (30,420):
        x=x0+shift
        for xline in [x,x+95,x+165,x+235,x+300]:
            d.line((xline,65,xline,135),fill='black',width=2)
        for y in (65,135):
            d.line((x,y,x+300,y),fill='black',width=2)
        # Distinct, repeatable synthetic header marks
        for j in range(3):
            d.rectangle((x+12+j*30, 85+j*3,x+22+j*30,91+j*3),fill='black')
    return img


def test_recovers_translation_without_manual_page_coordinates():
    a=sample();b=sample(13)
    boxes=[[30/800,65/500,330/800,470/500], [420/800,65/500,720/800,470/500]]
    matches=align_tables(a,b,boxes,min_score=.40,min_gap=.07)
    assert all(m.status=='provisional_aligned' for m in matches)
    assert all(abs(m.offset_x-13)<=1 for m in matches)


def test_cannot_trust_blank_scan():
    a=sample();b=Image.new('RGB',(800,500),'white')
    boxes=[[30/800,65/500,330/800,470/500]]
    assert align_tables(a,b,boxes)[0].status!='provisional_aligned'


def test_not_allows_out_of_bound_regions():
    a=sample();b=sample(13)
    boxes=[[.89,.13,.999,.94]]
    assert align_tables(a,b,boxes)[0].status!='provisional_aligned'
