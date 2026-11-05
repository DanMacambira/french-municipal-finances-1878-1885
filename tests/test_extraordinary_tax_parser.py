import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from extraordinary_tax_parser import (
    enrich_with_extraordinary_taxes, parse_pairs, tax_only_ocr_row)


def test_one_commune_multiple_pairs_keeps_original_order():
    got = parse_pairs("7 13", "1902 1888", 1882)
    assert len(got) == 2
    assert [(x["centimes_extraordinaires"], x["expiry_year"]) for x in got] == [
        (7, 1902), (13, 1888)]
    assert [x["remaining_years"] for x in got] == [20, 6]


def test_ditto_marks_not_invented_as_numeric_values():
    got = parse_pairs('" 2', '" 1902', 1882)
    assert got[0]["pair_status"] == "unresolved_ditto"
    assert got[0]["centimes_extraordinaires"] is None
    assert got[0]["expiry_year"] is None
    assert got[1]["centimes_extraordinaires"] == 2


def test_pair_count_mismatch_preserves_unpaired_entries():
    got = parse_pairs("7 13 6", "1902 1888", 1882)
    assert len(got) == 3
    assert got[2]["pair_status"] == "unequal_pair_counts"
    assert got[2]["expiry_year"] is None


def test_only_tax_cells_count_as_continuation():
    base = {"row_type": "uncertain", "raw_col_01": "", "raw_col_02": "", "raw_col_07": "2", "raw_col_08": "1910"}
    assert tax_only_ocr_row(base)
    assert not tax_only_ocr_row(base | {"raw_col_02": "13,683"})
    assert not tax_only_ocr_row(base | {"raw_col_01": "Morval"})


def test_no_automatic_detached_row_attachment_and_wide_columns():
    common = {"year": 1882, "pdf_page": 245, "side": "right_col2", "table_id": "tbl-1.html", "source_image": "test.png"}
    communes = pd.DataFrame([
        common | {"row_id": "c1", "commune_raw": "Hamelincourt", "centimes_extraordinaires_raw": "11", "duree_impositions_extraordinaires_raw": "1883"},
        common | {"row_id": "c2", "commune_raw": "Heninel", "centimes_extraordinaires_raw": '" 2', "duree_impositions_extraordinaires_raw": '" 1911'}])
    raw = pd.DataFrame([common | {"row_id": "tax-extra", "table_row": 7, "row_type": "uncertain", "raw_col_01": "", "raw_col_02": "", "raw_col_03": "", "raw_col_04": "", "raw_col_05": "", "raw_col_06": "", "raw_col_07": "2", "raw_col_08": "1910", "raw_col_09": ""}])
    wide, long, pending = enrich_with_extraordinary_taxes(communes, raw)
    assert len(pending) == 1
    assert len(long) == 1 + 2 + 1
    assert long.loc[long["source_kind"] == "detached_ocr_row", "row_id"].isna().all()
    assert wide.loc[0, "centimes_extraordinaires_1"] == 11
    assert wide.loc[0, "remaining_centime_years_ocr_unverified"] is None
    assert wide.loc[0, "extraordinary_tax_pair_status"] == "panel_has_unassigned_tax_rows"
    assert wide.loc[1, "extraordinary_tax_pair_status"] == "unresolved_tax_symbols_or_ocr"
    assert "centimes_extraordinaires_raw_2" in wide.columns
    assert wide.loc[1, "centimes_extraordinaires_raw_1"] == '"'


def test_no_detached_all_numeric_enables_provisional_burden():
    communes = pd.DataFrame([{
        "year": 1882, "pdf_page": 245, "side": "left_col1", "table_id": "tbl-0.html",
        "source_image": "test.png", "row_id": "c1", "commune_raw": "Morval",
        "centimes_extraordinaires_raw": "7 13", "duree_impositions_extraordinaires_raw": "1902 1888"}])
    wide, long, pending = enrich_with_extraordinary_taxes(communes, pd.DataFrame())
    assert pending.empty
    assert len(long) == 2
    assert wide.loc[0, "remaining_centime_years_ocr_unverified"] == 218
    assert wide.loc[0, "weighted_remaining_years_ocr_unverified"] == 10.9
    assert wide.loc[0, "extraordinary_tax_pair_status"] == "ocr_pairs_complete_unverified"
