"""Conservative registration of a manually selected table template to new scans.

This is a *preview* alignment. It never supplies verified crops. It matches the
repeated, printed column-header band independently for each table. It refuses
weak or overlapping matches; scan content hidden in a binding is unrecoverable.
"""
from __future__ import annotations

from dataclasses import dataclass
from PIL import Image
import cv2
import numpy as np


@dataclass(frozen=True)
class Match:
    box: tuple[float, float, float, float]
    score: float
    peak_gap: float
    offset_x: int
    offset_y: int
    status: str


def _gray(image: Image.Image, size=None):
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    a = np.asarray(image.convert('RGB')).copy()
    # Decorations from prior annotation screenshots do not form part of the
    # source PDF. Ignore them so the reference lines cannot become anchors.
    r, g, b = a[..., 0].astype(np.int16), a[..., 1].astype(np.int16), a[..., 2].astype(np.int16)
    colorful = (r > 100) & (r > 1.45*g) & (r > 1.45*b)
    colorful |= (b > 85) & (b > 1.3*g) & (b > 1.15*r)
    if np.any(colorful):
        colorful = cv2.dilate(colorful.astype(np.uint8), np.ones((3,3),np.uint8))>0
        a[colorful] = 255
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)[1]


def _px(box, width, height):
    return (int(round(box[0]*width)), int(round(box[1]*height)),
            int(round(box[2]*width)), int(round(box[3]*height)))


def align_tables(reference: Image.Image, target: Image.Image, boxes: list,
                 min_score: float = 0.43, min_gap: float = 0.10):
    """Return one Match per region, never change input or silently trust a crop.

    Each table's header is a stable printed pattern even when the communes
    differ. Search only in the vicinity of its normalized reference position.
    A match that creates new overlap with a neighbouring table is rejected.
    """
    width, height = target.size
    ref = _gray(reference, (width, height))
    dest = _gray(target)
    results = []
    for region in boxes:
        x0,y0,x1,y1 = _px(region,width,height)
        w,h=x1-x0,y1-y0
        # Header band, avoiding the outer frame and variable commune entries.
        ax=x0+round(.07*w); bx=x1-round(.07*w)
        ay=y0+round(.015*h); by=y0+round(.145*h)
        if bx-ax < 40 or by-ay < 30:
            results.append(Match(tuple(region), 0., 0., 0, 0, 'uncertain_small_region'))
            continue
        templ=ref[ay:by,ax:bx]
        dx=round(.065*width);dy=round(.065*height)
        sx0=max(0,ax-dx);sy0=max(0,ay-dy)
        sx1=min(width,bx+dx);sy1=min(height,by+dy)
        search=dest[sy0:sy1,sx0:sx1]
        scores=cv2.matchTemplate(search,templ,cv2.TM_CCOEFF_NORMED)
        if not np.isfinite(scores).all():
            results.append(Match(tuple(region), 0., 0., 0, 0, 'uncertain_invalid_correlation'))
            continue
        _,score,_,peak=cv2.minMaxLoc(scores)
        # Estimate uniqueness: exclude the local peak and its immediate shifts.
        copied=scores.copy();px,py=peak
        copied[max(0,py-10):py+11,max(0,px-10):px+11]=-1
        runner_up=float(np.max(copied)) if copied.size else -1.
        gap=float(score-runner_up)
        offset_x=int(sx0+px-ax);offset_y=int(sy0+py-ay)
        proposed=(x0+offset_x,y0+offset_y,x1+offset_x,y1+offset_y)
        inside=(proposed[0]>=0 and proposed[1]>=0 and proposed[2]<=width and proposed[3]<=height)
        plausible=(score>=min_score and gap>=min_gap and inside)
        if not plausible:
            status='uncertain_low_similarity' if score < min_score else ('uncertain_ambiguous' if gap < min_gap else 'uncertain_out_of_bounds')
            results.append(Match(tuple(region),float(score),gap,offset_x,offset_y,status))
            continue
        normalized=tuple(round(v/(width if i%2==0 else height),6) for i,v in enumerate(proposed))
        results.append(Match(normalized,float(score),gap,offset_x,offset_y,'provisional_aligned'))

    # The input order must be left-to-right. Reject newly overlapping boxes;
    # don't let a spurious header match cross into a neighbouring table.
    for i in range(1,len(results)):
        prev=results[i-1];now=results[i]
        if prev.status!='provisional_aligned' and now.status!='provisional_aligned':
            continue
        old_overlap=max(0.,boxes[i-1][2]-boxes[i][0]);
        new_overlap=max(0.,prev.box[2]-now.box[0]);
        if new_overlap>old_overlap+.007:
            # Prefer the earlier table when ambiguous; the offending box is
            # flagged and reverted to its original normalized position.
            results[i]=Match(tuple(boxes[i]),now.score,now.peak_gap,now.offset_x,now.offset_y,'uncertain_table_overlap')
    return results
