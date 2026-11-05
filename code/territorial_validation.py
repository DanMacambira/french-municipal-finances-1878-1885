"""Traceable geography and canton-level OCR accounting checks.

Input is the preserved Mistral table data, not a trusted historical database.
All checks compare OCR with OCR: a matching checksum is not proof of accuracy.
There is no API access and no inference of absent commune names.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata
from typing import Any

import pandas as pd


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _normalize(value: Any) -> str:
    s = unicodedata.normalize("NFKD", _text(value))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9]+", " ", s.upper()).strip()


def markdown_table_prefaces(markdown: str) -> dict[str, str]:
    """Return the text just preceding each linked Mistral HTML table.

    This matters when Mistral puts the canton title OUTSIDE the HTML table.
    Example: '[tbl-0.html](tbl-0.html)\nCANTON DE ...\n[tbl-1.html]'.
    """
    link = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    last = 0
    out = {}
    for match in link.finditer(markdown or ""):
        label, target = match.groups()
        out[target] = (markdown or "")[last:match.start()]
        out[label] = (markdown or "")[last:match.start()]
        last = match.end()
    return out


def heading_event(line: Any) -> tuple[str, str, bool] | None:
    """Recognise printed section headings, with conservative OCR normalization."""
    text = _normalize(line)
    if not text:
        return None
    arr = re.search(r"\bARRONDISSEMENT\s+(?:DE|D|DU|DES)\s+(.+)$", text)
    if arr:
        name = re.sub(r"\b(?:SUITE|CONTINUATION)\b.*$", "", arr.group(1)).strip()
        return ("arrondissement", name, "SUITE" in text) if name else None
    canton = re.search(r"\bCANTON\s+(?:DE|D|DU|DES)\s+(.+)$", text)
    if canton:
        name = canton.group(1)
        continuation = bool(re.search(r"\bSUITE\b", name))
        name = re.sub(r"\bSUITE\b.*$", "", name).strip()
        return ("canton", name, continuation) if name else None
    return None


def _parse_additive_int(value: Any) -> int | None:
    """Only unambiguous nonnegative integers, with printed thousands groups."""
    s = _text(value)
    if re.fullmatch(r"\d+", s):
        return int(s)
    if re.fullmatch(r"\d{1,3}(?:[., ]\d{3})+", s):
        return int(re.sub(r"[., ]", "", s))
    return None


def _is_subtotal(row: dict) -> bool:
    """An unnamed line with values in three additive financial columns.

    A continuation of tax expiry lines has values only in cols 7/8 and cannot
    satisfy this rule. Do not mistake a blank or ditto mark for zero.
    """
    return (not _text(row.get("raw_col_01"))
            and all(_parse_additive_int(row.get(f"raw_col_{i:02d}")) is not None
                    for i in (2, 3, 4)))


def annotate_territories(raw_rows: pd.DataFrame, communes: pd.DataFrame):
    """Carry arrondissement/canton across OCR table boundaries in reading order.

    Existing raw row IDs are retained. A new canton starts at an explicit
    heading; a '(suite)' without a previous matching canton remains incomplete.
    The section ID prevents accidental merging of two similarly named cantons.
    """
    if raw_rows.empty:
        return raw_rows.copy(), communes.copy()

    records = raw_rows.to_dict("records")
    state = {"year": None, "arrondissement": None, "canton": None,
             "canton_segment_id": None, "canton_started_here": False,
             "canton_heading_status": "missing"}
    next_segment = 0
    previous_table = None
    seen = []
    attached = {}

    def apply_event(event):
        nonlocal next_segment
        kind, label, continuation = event
        if kind == "arrondissement":
            state["arrondissement"] = label
            return
        previous = state["canton"]
        equivalent = previous == label
        fuzzy = bool(continuation and previous and not equivalent
                     and SequenceMatcher(None, previous, label).ratio() >= 0.85)
        # A new, non-continuation heading always identifies a new section when
        # the prior canton differs or was closed by a subtotal.
        if state["canton_segment_id"] is None or (not equivalent and not fuzzy):
            next_segment += 1
            state["canton_segment_id"] = next_segment
            state["canton"] = label
            state["canton_started_here"] = not continuation
            state["canton_heading_status"] = (
                "printed_start" if not continuation else "continuation_without_start")
        elif fuzzy:
            state["canton_heading_status"] = "continuation_ocr_name_disagrees"
        elif not continuation:
            state["canton_started_here"] = True
            state["canton_heading_status"] = "printed_start"
        # If equivalent and marked (suite), retain the existing identity.

    for row in records:
        year = int(row["year"])
        if state["year"] != year:
            state.update({"year": year, "arrondissement": None, "canton": None,
                          "canton_segment_id": None, "canton_started_here": False,
                          "canton_heading_status": "missing"})
            previous_table = None
        table_key = (year, int(row["pdf_page"]), _text(row["side"]), _text(row["table_id"]))
        if table_key != previous_table:
            # Markdown headings relate to the following linked HTML table,
            # not to every table on the same OCR image.
            for line in _text(row.get("ocr_table_preface")).splitlines():
                event = heading_event(line)
                if event:
                    apply_event(event)
            previous_table = table_key
        event = heading_event(row.get("raw_col_01"))
        if event:
            apply_event(event)
        if _is_subtotal(row):
            row["row_type"] = "subtotal"
        row["arrondissement"] = state["arrondissement"]
        row["canton"] = state["canton"]
        row["canton_segment_id"] = state["canton_segment_id"]
        row["canton_started_here"] = state["canton_started_here"]
        row["canton_heading_status"] = state["canton_heading_status"]
        seen.append(row)
        attached[row["row_id"]] = row
        if row["row_type"] == "subtotal":
            # A subtotal ends this canton, even if the next heading is missing.
            state.update({"canton": None, "canton_segment_id": None,
                          "canton_started_here": False,
                          "canton_heading_status": "missing"})

    candidates = []
    for row in communes.to_dict("records"):
        source = attached.get(row.get("row_id"), {})
        row.update({key: source.get(key) for key in
                    ("arrondissement", "canton", "canton_segment_id",
                     "canton_started_here", "canton_heading_status")})
        candidates.append(row)
    return pd.DataFrame(seen), pd.DataFrame(candidates)


def _region_order(region: str) -> int:
    """Physical reading order of OCR regions on a source PDF page."""
    order = {"full": 0, "full_col1": 0, "full_col2": 1,
             "left": 0, "left_col1": 0, "left_col2": 1,
             "right": 2, "right_col1": 2, "right_col2": 3}
    if region not in order:
        raise ValueError(f"Unknown OCR region for canton QA: {region!r}")
    return order[region]


def _region_position(key: tuple[int, int, str]) -> tuple[int, int, int]:
    year, page, region = key
    return (int(year), int(page), _region_order(region))


# The OCR table's column indices are 1-based; only additive quantities qualify.
ADDITIVE_FIELDS = (
    ("population", "population", 2),
    ("superficie_hectares", "superficie_hectares", 3),
    ("revenus_annuels_francs", "revenus_annuels_francs", 4),
    ("centimes_total", "centimes_total", 6),
)


def canton_subtotal_checks(raw_rows: pd.DataFrame, communes: pd.DataFrame,
                           expected_regions: set[tuple[int, int, str]] | None = None,
                           available_regions: set[tuple[int, int, str]] | None = None) -> pd.DataFrame:
    """Compare mapped commune values to canton subtotals found in OCR.

    Only compare when an explicit canton START and its subtotal were observed.
    Unknown or omitted observations yield NOT_CHECKED, never zero. Printed
    subtotal *itself* may be an OCR error: mismatches require source review.
    """
    output_columns = ["year", "arrondissement", "canton", "canton_segment_id",
                      "variable", "commune_count", "commune_sum",
                      "subtotal_ocr_raw", "subtotal_ocr_value", "difference",
                      "status", "canton_scope", "subtotal_source_row_id",
                      "n_missing_commune_values", "missing_ocr_regions_between"]
    if raw_rows.empty:
        return pd.DataFrame(columns=output_columns)
    source = raw_rows[raw_rows["canton_segment_id"].notna()].copy()
    if source.empty:
        return pd.DataFrame(columns=output_columns)
    out = []
    for (year, segment), group in source.groupby(["year", "canton_segment_id"], sort=False):
        sub = group[group["row_type"] == "subtotal"]
        rows = (communes[(communes["year"] == year) &
                         (communes["canton_segment_id"] == segment)]
                if not communes.empty else pd.DataFrame())
        first = group.iloc[0]
        scope_start = bool(group["canton_started_here"].fillna(False).any())
        missing_regions = []
        if expected_regions is not None and available_regions is not None and len(sub) == 1:
            first_key = (int(year), int(first["pdf_page"]), str(first["side"]))
            subtotal_row = sub.iloc[0]
            last_key = (int(year), int(subtotal_row["pdf_page"]), str(subtotal_row["side"]))
            a, b = _region_position(first_key), _region_position(last_key)
            missing_regions = sorted(
                (key for key in expected_regions if key not in available_regions
                 and a <= _region_position(key) <= b), key=_region_position)
        full_scope = scope_start and len(sub) == 1 and not missing_regions
        scope = ("start_not_observed" if not scope_start else
                 "missing_subtotal" if sub.empty else
                 "multiple_subtotals" if len(sub) > 1 else
                 "missing_ocr_regions" if missing_regions else
                 "complete_canton_in_cached_ocr")
        subtotal = sub.iloc[0] if len(sub) == 1 else None
        for name, field, column in ADDITIVE_FIELDS:
            sub_raw = (subtotal.get(f"raw_col_{column:02d}")
                       if subtotal is not None else None)
            sub_value = _parse_additive_int(sub_raw)
            n = len(rows)
            values = (pd.to_numeric(rows[field], errors="coerce")
                      if n and field in rows else pd.Series(dtype="float64"))
            missing = int(values.isna().sum()) if n and field in rows else n
            total = (int(values.sum()) if n and missing == 0
                     else None)
            delta = None
            if not full_scope:
                status = "not_checked_incomplete_canton"
            elif field not in rows:
                status = "not_checked_unmapped_column"
            elif missing or not n:
                status = "not_checked_missing_commune_values"
            elif sub_value is None:
                status = "not_checked_unreadable_subtotal"
            else:
                delta = total - sub_value
                status = "ocr_sums_match" if delta == 0 else "ocr_discrepancy_review_scan"
            out.append({"year": int(year), "arrondissement": first.get("arrondissement"),
                        "canton": first.get("canton"), "canton_segment_id": int(segment),
                        "variable": name, "commune_count": n,
                        "commune_sum": total, "subtotal_ocr_raw": _text(sub_raw),
                        "subtotal_ocr_value": sub_value, "difference": delta,
                        "status": status, "canton_scope": scope,
                        "subtotal_source_row_id": (subtotal.get("row_id") if subtotal is not None else None),
                        "n_missing_commune_values": missing,
                        "missing_ocr_regions_between": ";".join(
                            f"{y}:p{p:04d}_{side}" for y, p, side in missing_regions)})
    return pd.DataFrame(out, columns=output_columns)
