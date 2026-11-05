import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from territorial_validation import (
    annotate_territories, canton_subtotal_checks, heading_event,
    markdown_table_prefaces,
)


def row(label, year=1882, panel="left_col1", table="tbl-0.html", i=1,
        values=(None, None, None, None), preface="", kind="candidate_commune"):
    r = {"row_id": f"{year}-p0245-{panel}-{table}-r{i:03d}",
         "year": year, "pdf_page": 245, "side": panel, "table_id": table,
         "table_row": i, "row_type": kind, "raw_col_01": label,
         "ocr_table_preface": preface}
    for k, v in zip((2, 3, 4, 6), values):
        r[f"raw_col_{k:02d}"] = v
    return r


def candidate(r, values):
    return {"row_id": r["row_id"], "year": r["year"],
            "population": values[0], "superficie_hectares": values[1],
            "revenus_annuels_francs": values[2], "centimes_total": values[3]}


def test_prefaces_link_correct_table_not_whole_page():
    text = "CALAIS\n[tbl-0.html](tbl-0.html)\nCANTON DE BERTINCOURT.\n[tbl-1.html](tbl-1.html)\nCANTON DE CROISILLES.\n[tbl-2.html](tbl-2.html)"
    p = markdown_table_prefaces(text)
    assert p["tbl-0.html"] == "CALAIS\n"
    assert "BERTINCOURT" in p["tbl-1.html"]
    assert "CROISILLES" in p["tbl-2.html"]
    assert "CROISILLES" not in p["tbl-1.html"]


def test_heading_parse_accents_and_suite():
    assert heading_event("Suite de l'arrondissement d'Arras.") == ("arrondissement", "ARRAS", True)
    assert heading_event("CANTON DE BEAUMETZ-LEZ-LOGES (suite).") == (
        "canton", "BEAUMETZ LEZ LOGES", True)
    assert heading_event("CANTON DE CROISILLES.") == ("canton", "CROISILLES", False)


def test_arrondissement_and_canton_cross_table_and_subtotal():
    a = row("Alpha", i=1,
            preface="ARRONDISSEMENT D'ARRAS.\nCANTON DE BEAUMETZ-LEZ-LOGES.",
            values=("10", "20", "30", "40"))
    b = row("Beta", panel="left_col2", i=1,
            preface="CANTON DE BEAUMETZ-LEZ-LOGES (suite).",
            values=("15", "25", "35", "45"))
    subtotal = row("", panel="left_col2", i=2,
                   values=("25", "45", "65", "85"), kind="uncertain")
    following = row("Gamma", panel="right_col1", i=1,
                    preface="CANTON DE BERTINCOURT.", values=("2", "3", "4", "5"))
    raw, comm = annotate_territories(
        pd.DataFrame([a, b, subtotal, following]),
        pd.DataFrame([candidate(a, [10, 20, 30, 40]),
                      candidate(b, [15, 25, 35, 45]),
                      candidate(following, [2, 3, 4, 5])]))
    assert list(comm["arrondissement"]) == ["ARRAS"] * 3
    assert list(comm["canton"]) == ["BEAUMETZ LEZ LOGES", "BEAUMETZ LEZ LOGES", "BERTINCOURT"]
    assert raw.iloc[2]["row_type"] == "subtotal"
    assert raw.iloc[0]["canton_segment_id"] == raw.iloc[2]["canton_segment_id"]
    assert raw.iloc[2]["canton_segment_id"] != raw.iloc[3]["canton_segment_id"]
    checks = canton_subtotal_checks(raw, comm)
    beaumetz = checks[checks.canton == "BEAUMETZ LEZ LOGES"]
    assert len(beaumetz) == 4
    assert set(beaumetz.status) == {"ocr_sums_match"}
    assert set(beaumetz.canton_scope) == {"complete_canton_in_cached_ocr"}
    bertincourt = checks[checks.canton == "BERTINCOURT"]
    assert set(bertincourt.status) == {"not_checked_incomplete_canton"}


def test_subtotal_ocr_discrepancy_flag_not_correction():
    a = row("Alpha", preface="CANTON DE CROISILLES.", values=("10", "20", "30", "40"))
    sub = row("", i=2, values=("11", "20", "30", "40"), kind="uncertain")
    raw, com = annotate_territories(pd.DataFrame([a, sub]),pd.DataFrame([candidate(a, [10, 20, 30, 40])]))
    checks = canton_subtotal_checks(raw, com)
    pop = checks[checks.variable == "population"].iloc[0]
    assert pop["difference"] == -1
    assert pop["status"] == "ocr_discrepancy_review_scan"
    assert (checks[checks.variable == "centimes_total"].iloc[0]["status"]
            == "ocr_sums_match")


def test_unobserved_canton_start_prevents_checks():
    a = row("Alpha", preface="CANTON DE CROISILLES (suite).", values=("10", "20", "30", "40"))
    sub = row("", i=2, values=("10", "20", "30", "40"), kind="uncertain")
    raw, com = annotate_territories(pd.DataFrame([a, sub]), pd.DataFrame([candidate(a, [10, 20, 30, 40])]))
    checks = canton_subtotal_checks(raw, com)
    assert set(checks.status) == {"not_checked_incomplete_canton"}
    assert set(checks.canton_scope) == {"start_not_observed"}


def test_missing_ocr_region_between_sections_blocks_checks():
    a = row("Alpha", preface="CANTON DE CROISILLES.", values=("10", "20", "30", "40"))
    sub = row("", panel="right_col1", table="tbl-1.html", values=("10", "20", "30", "40"), kind="uncertain")
    raw, com = annotate_territories(pd.DataFrame([a, sub]), pd.DataFrame([candidate(a, [10, 20, 30, 40])]))
    expected={(1882,245,"left_col1"),(1882,245,"left_col2"),(1882,245,"right_col1")}
    available={(1882,245,"left_col1"),(1882,245,"right_col1")}
    checks=canton_subtotal_checks(raw, com, expected, available)
    assert set(checks.canton_scope) == {"missing_ocr_regions"}
    assert "p0245_left_col2" in checks.iloc[0].missing_ocr_regions_between


def test_continuation_ocr_typo_keeps_canton_identity():
    a = row("Alpha", preface="ARRONDISSEMENT D'ARRAS.\nCANTON DE BERTINCOURT.", values=("10", "20", "30", "40"))
    b = row("Beta", panel="right_col2", preface="CANTON DE BERLINCOURT (suite).", values=("11", "21", "31", "41"))
    raw, comm=annotate_territories(pd.DataFrame([a,b]), pd.DataFrame([candidate(a,[10,20,30,40]), candidate(b,[11,21,31,41])]))
    assert comm.iloc[0].canton == comm.iloc[1].canton == "BERTINCOURT"
    assert comm.iloc[1].canton_heading_status == "continuation_ocr_name_disagrees"


def test_tax_only_continuation_not_misclassified_as_subtotal():
    a = row("Alpha", preface="CANTON DE MARQUION.", values=("10", "20", "30", "40"))
    ext = row("", i=2, kind="uncertain")
    ext.update({"raw_col_07": "3", "raw_col_08": "1911"})
    raw, _ = annotate_territories(pd.DataFrame([a,ext]), pd.DataFrame([candidate(a,[10,20,30,40])]))
    assert raw.iloc[1].row_type == "uncertain"
