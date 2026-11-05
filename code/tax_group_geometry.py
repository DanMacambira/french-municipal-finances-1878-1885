"""Conservative geometry-only matching of extraordinary-tax lines to communes.

This module deliberately does NOT OCR, infer table coordinates, or treat printed
straight strokes as group boundaries. The caller supplies positions from a
coordinate-aware OCR/layout detector and curved-brace *proposals* from image
analysis. Results are candidates for QA, never validated financial data.
"""
from __future__ import annotations

from statistics import median


def _spacing(centres: list[float], default: float = 25.0) -> float:
    ordered = sorted(set(float(y) for y in centres))
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
    # Numeric tax lines often occur in tight stacks separated by large gaps
    # between communes; use the lower half to estimate local line spacing.
    return float(median(sorted(gaps)[:(len(gaps) + 1) // 2])) if gaps else default


def associate_tax_lines(communes: list[dict], tax_lines: list[dict],
                        brace_candidates: list[dict]) -> list[dict]:
    """Propose commune associations using curved braces OR single-row alignment.

    Parameters
    ----------
    communes: [{id, y_center, ...}] text-line anchors, coordinate OCR-derived
    tax_lines: [{id, y_center, ...}] paired centime/year rows (exclude subtotals)
    brace_candidates: [{y_min, y_max, kind, ...}] image-derived *curved* marks;
        `kind='curly_brace_candidate'` is required for grouping. Straight table
        rules are deliberately ignored even when they connect many entries.

    Auto-linked means geometrically unambiguous, NOT historically verified.
    If a curved brace is broken or the name is poorly aligned, entries remain
    unresolved instead of making an unsafe many-to-one assignment.
    """
    for group, objects in (("commune", communes), ("tax_line", tax_lines)):
        ids = [str(obj["id"]) for obj in objects]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {group} id")
        if any("y_center" not in obj for obj in objects):
            raise ValueError(f"missing {group} y_center")
    names = sorted(communes, key=lambda a: float(a["y_center"]))
    taxes = sorted(tax_lines, key=lambda a: float(a["y_center"]))
    associations = {str(t["id"]): {"tax_id": t["id"], "commune_id": None,
                                    "status": "needs_review", "reason": "not_yet_matched"}
                    for t in taxes}
    if not taxes or not names:
        return list(associations.values())
    tax_step = _spacing([float(t["y_center"]) for t in taxes])
    name_step = _spacing([float(c["y_center"]) for c in names], default=2 * tax_step)
    brace_pad = 0.2 * tax_step
    used_tax_ids = set()
    claimed_names = set()
    blocked_tax_ids = set()

    # First use a genuine curved mark spanning multiple numeric entries.
    # If the mark or name association is ambiguous, do NOT claim those rows.
    for brace in sorted(brace_candidates, key=lambda b: (float(b["y_min"]), float(b["y_max"]))):
        if brace.get("kind") != "curly_brace_candidate":
            continue
        low, high = sorted((float(brace["y_min"]), float(brace["y_max"])))
        inside = [t for t in taxes if low - brace_pad <= float(t["y_center"]) <= high + brace_pad]
        if len(inside) < 2:
            continue  # a brace candidate must actually explain multiple entries
        # In the printed tables a commune name can be at the top OR centre of
        # the brace, hence allow it to overlap the expanded brace interval.
        overlapping_names = [n for n in names if low - tax_step <= float(n["y_center"]) <= high + tax_step]
        ids = {str(t["id"]) for t in inside}
        if (len(overlapping_names) != 1 or ids & used_tax_ids
                or str(overlapping_names[0]["id"]) in claimed_names):
            blocked_tax_ids.update(ids)
            for t in inside:
                a = associations[str(t["id"])]
                if a["commune_id"] is None:
                    a["reason"] = "ambiguous_brace_or_name_overlap"
            continue
        commune = overlapping_names[0]
        for t in inside:
            associations[str(t["id"])] = {
                "tax_id": t["id"], "commune_id": commune["id"],
                "status": "proposed_brace_group", "reason": "curved_brace_vertical_extent"}
        used_tax_ids.update(ids)
        claimed_names.add(str(commune["id"]))

    # Remaining singles: a continuous straight line is not a grouping marker.
    # Only match an isolated tax row closely aligned with a SINGLE commune;
    # if 2+ rows compete for that same commune, leave all unassigned.
    unmatched = [t for t in taxes if str(t["id"]) not in used_tax_ids
                 and str(t["id"]) not in blocked_tax_ids]
    nearest_groups: dict[str, list[tuple[dict, float, float]]] = {}
    for tax in unmatched:
        y = float(tax["y_center"])
        distances = sorted((abs(y - float(n["y_center"])), str(n["id"])) for n in names)
        distance, name_id = distances[0]
        nearest_other = distances[1][0] if len(distances) > 1 else float('inf')
        nearest_groups.setdefault(name_id, []).append((tax, distance, nearest_other))
    for name_id, matches in nearest_groups.items():
        if len(matches) > 1:
            for t, _, _ in matches:
                associations[str(t["id"])]["reason"] = "multiple_tax_lines_without_verified_brace"
            continue
        t, distance, nearest_other = matches[0]
        # Do not use straight bars as multi-entry braces. A singleton can be
        # accepted only when position makes the match unambiguous.
        if (name_id not in claimed_names and distance <= 0.38 * tax_step
                and distance < 0.45 * nearest_other
                and distance <= 0.25 * name_step):
            associations[str(t["id"])] = {
                "tax_id": t["id"], "commune_id": next(n["id"] for n in names if str(n["id"]) == name_id),
                "status": "proposed_single_alignment", "reason": "one_to_one_vertical_alignment"}
        else:
            associations[str(t["id"])]["reason"] = "no_unique_nearby_commune"
    return list(associations.values())
