"""Conservative tax-pair extraction from preserved Mistral OCR table rows.

No OCR calls are made here. Raw OCR cells remain unchanged. Pairs within the
same commune row are linked; detached lines are exported *unlinked*, never
silently assigned to the nearest commune.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

C_FIELD = "centimes_extraordinaires_raw"
Y_FIELD = "duree_impositions_extraordinaires_raw"
KEYS = ("year", "pdf_page", "side", "table_id")
GEO = ("arrondissement", "canton", "canton_segment_id")
DITTO = {'"', "\u00bb", "\u00ab", "\u201d", "\u201c"}


def cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def split_tokens(value: Any) -> list[str]:
    """Split flattened cells; never convert a ditto sign to a number."""
    return re.findall(r'[^\s;|]+', cell_text(value))


def parse_pairs(centimes: Any, expiry: Any, fiscal_year: int) -> list[dict]:
    """One record per element position, including missing/unresolved elements."""
    cs = split_tokens(centimes)
    ys = split_tokens(expiry)
    out = []
    for i in range(max(len(cs), len(ys))):
        c_raw = cs[i] if i < len(cs) else ""
        y_raw = ys[i] if i < len(ys) else ""
        c_val = int(c_raw) if re.fullmatch(r'\d{1,3}', c_raw) else None
        y_val = int(y_raw) if re.fullmatch(r'\d{4}', y_raw) else None
        # Deliberately leave ditto, OCR punctuation and implausible years unresolved.
        if y_val is not None and not (1800 <= y_val <= 2100):
            y_val = None
        status = ("paired_numeric" if c_val is not None and y_val is not None
                  else "unresolved_ditto" if c_raw in DITTO or y_raw in DITTO
                  else "unequal_pair_counts" if len(cs) != len(ys)
                  else "unresolved_ocr")
        out.append({"entry_number": i + 1,
                    "centimes_extraordinaires_raw": c_raw,
                    "duree_impositions_extraordinaires_raw": y_raw,
                    "centimes_extraordinaires": c_val,
                    "expiry_year": y_val,
                    "remaining_years": max(y_val - fiscal_year, 0)
                        if y_val is not None else None,
                    "pair_status": status})
    return out


def tax_only_ocr_row(row: dict) -> bool:
    """Detect rows with tax cells but no commune/main financial fields."""
    if cell_text(row.get("row_type")) != "uncertain":
        return False
    if cell_text(row.get("raw_col_01")):
        return False
    if any(cell_text(row.get(f"raw_col_{i:02d}")) for i in range(2, 7)):
        return False
    return bool(cell_text(row.get("raw_col_07")) or
                cell_text(row.get("raw_col_08")))


def enrich_with_extraordinary_taxes(communes: pd.DataFrame, raw_rows: pd.DataFrame):
    """Return wide candidates, long pairs and unlinked continuation rows.

    The returned wide table is NOT a validated tax measure. Each pair carries
    its source row and pairing status, and panels with detached tax rows are
    flagged so no derived complete burden is silently calculated from them.
    """
    wide = communes.copy()
    if wide.empty:
        return wide, pd.DataFrame(), pd.DataFrame()

    # Keep detached rows in the source order and physical panel. Do not infer
    # their commune solely from their position in Mistral's HTML table.
    detached = []
    if not raw_rows.empty:
        for row in raw_rows.to_dict("records"):
            if not tax_only_ocr_row(row):
                continue
            entries = parse_pairs(row.get("raw_col_07"), row.get("raw_col_08"),
                                  int(row["year"]))
            detached.append({k: row.get(k) for k in (*KEYS, *GEO, "row_id", "table_row", "source_image")}
                            | {"centimes_extraordinaires_raw": cell_text(row.get("raw_col_07")),
                               "duree_impositions_extraordinaires_raw": cell_text(row.get("raw_col_08")),
                               "revenus_bureau_bienfaisance_raw": cell_text(row.get("raw_col_09")),
                               "pair_count": len(entries),
                               "pair_count_mismatch": len(split_tokens(row.get("raw_col_07"))) !=
                                                      len(split_tokens(row.get("raw_col_08"))),
                               "link_status": "unassigned_needs_review"})
    detached_df = pd.DataFrame(detached)
    affected = set()
    for row in detached:
        affected.add(tuple(row.get(k) for k in KEYS))

    all_entries = []
    slots = []
    panel_warning = []
    statuses = []
    complete_burden = []
    weighted_duration = []
    for row in wide.to_dict("records"):
        fiscal_year = int(row["year"])
        entries = parse_pairs(row.get(C_FIELD), row.get(Y_FIELD), fiscal_year)
        panel_has_detached = tuple(row.get(k) for k in KEYS) in affected
        panel_warning.append(panel_has_detached)
        counts_equal = len(split_tokens(row.get(C_FIELD))) == len(split_tokens(row.get(Y_FIELD)))
        all_numeric = bool(entries) and all(p["pair_status"] == "paired_numeric" for p in entries)
        if not counts_equal:
            status = "pair_count_mismatch"
        elif any(p["pair_status"] != "paired_numeric" for p in entries):
            status = "unresolved_tax_symbols_or_ocr"
        elif panel_has_detached:
            status = "panel_has_unassigned_tax_rows"
        elif not entries:
            status = "no_tax_pairs_in_row"
        else:
            status = "ocr_pairs_complete_unverified"
        statuses.append(status)
        # Numeric indicators are available ONLY when the OCR has no known
        # structural incompleteness in the table; values remain unverified.
        if all_numeric and not panel_has_detached:
            exposure = sum(p["centimes_extraordinaires"] * p["remaining_years"] for p in entries)
            centimes_sum = sum(p["centimes_extraordinaires"] for p in entries)
            complete_burden.append(exposure)
            weighted_duration.append(exposure / centimes_sum if centimes_sum else None)
        else:
            complete_burden.append(None)
            weighted_duration.append(None)
        slots.append(entries)
        for item in entries:
            all_entries.append({k: row.get(k) for k in (*KEYS, *GEO, "row_id", "commune_raw", "source_image")}
                               | {"source_row_id": row.get("row_id"),
                                  "link_status": "linked_same_ocr_row",
                                  "source_kind": "commune_row"} | item)

    # A detached tax-only OCR row is never dropped from the long output.
    for row in detached:
        for item in parse_pairs(row["centimes_extraordinaires_raw"],
                                row["duree_impositions_extraordinaires_raw"],
                                int(row["year"])):
            all_entries.append({k: row.get(k) for k in (*KEYS, *GEO, "source_image")}
                               | {"row_id": None, "commune_raw": None,
                                  "source_row_id": row["row_id"],
                                  "source_kind": "detached_ocr_row",
                                  "link_status": "unassigned_needs_review"} | item)

    ncols = max((len(p) for p in slots), default=0)
    for num in range(1, ncols + 1):
        wide[f"centimes_extraordinaires_raw_{num}"] = [
            pairs[num-1]["centimes_extraordinaires_raw"] if len(pairs) >= num else None
            for pairs in slots]
        wide[f"duree_impositions_extraordinaires_raw_{num}"] = [
            pairs[num-1]["duree_impositions_extraordinaires_raw"] if len(pairs) >= num else None
            for pairs in slots]
        wide[f"centimes_extraordinaires_{num}"] = [
            pairs[num-1]["centimes_extraordinaires"] if len(pairs) >= num else None
            for pairs in slots]
        wide[f"expiry_year_{num}"] = [
            pairs[num-1]["expiry_year"] if len(pairs) >= num else None
            for pairs in slots]

    wide["extraordinary_tax_pairs_in_commune_row"] = [len(pairs) for pairs in slots]
    wide["tax_panel_has_unassigned_rows"] = panel_warning
    wide["extraordinary_tax_pair_status"] = statuses
    wide["remaining_centime_years_ocr_unverified"] = complete_burden
    wide["weighted_remaining_years_ocr_unverified"] = weighted_duration
    return wide, pd.DataFrame(all_entries), detached_df
