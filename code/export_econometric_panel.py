#!/usr/bin/env python3
"""Make an auditable, provisional long commune-year analysis panel from build candidates.

Designed for any department and year with reviewed printed-column mappings.
Never imputes zero, invents commune IDs, or silently selects duplicate rows.
Extraordinary tax data are excluded from the analysis panel and retained separately.
"""
from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path
from collections import Counter
import pandas as pd

# These are semantic names supplied by the main pipeline's reviewed column mappings.
# They are not PDF column positions and are not year/page-specific.
FIELDS = {
    "population": "int",
    "superficie_hectares": "int",
    "revenus_annuels_francs": "int",
    "valeur_centime_francs": "decimal",
    "centimes_total": "int",
    "revenus_bureau_bienfaisance_raw": "int",
}
CORE_FIELDS = list(FIELDS)[:5]
EXPORTED_FIELDS = CORE_FIELDS + ["revenus_bureau_bienfaisance_francs"]
TAX_COLS = [
    "centimes_extraordinaires_raw", "duree_impositions_extraordinaires_raw",
    "extraordinary_tax_pairs_json", "extraordinary_tax_pair_status",
    "extraordinary_tax_pair_count", "extraordinary_tax_pair_notes",
]
META_FIELDS = [
    "row_id", "year", "cassini_id", "commune_cleaned", "commune_raw",
    "arrondissement", "canton", "canton_segment_id", "pdf_page", "side",
    "panel", "source_pdf", "source_image", "table_id", "table_row",
    "column_count", "layout", "ocr_model", "ocr_page_confidence",
    "verification_status", "cm_status", "extraction_flags", "review_flags",
]
SPECIAL_EMPTY = {"", "-", "—", "–", "»", "«", '"', "''", ".", "..", "NA", "N/A"}


def parse_number(value, kind: str):
    """(number or pd.NA, parse-status). Never use a guess to fix OCR digit errors."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return pd.NA, "missing"
    if isinstance(value, bool):
        return pd.NA, "invalid"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return pd.NA, "missing"
        if kind == "int":
            return (int(value), "parsed") if float(value).is_integer() else (pd.NA, "ambiguous")
        return float(value), "parsed"
    if not isinstance(value, str):
        return pd.NA, "invalid"
    s = value.strip().replace("\u00a0", " ").replace("\u202f", " ")
    if s.upper() in SPECIAL_EMPTY:
        return pd.NA, "missing"
    if kind == "int":
        if re.fullmatch(r"\d+", s):
            return int(s), "parsed"
        if re.fullmatch(r"\d{1,3}(?:[ ,.]\d{3})+", s):
            return int(re.sub(r"[ ,.]+", "", s)), "parsed"
        return pd.NA, "ambiguous"
    if re.fullmatch(r"\d+(?:[,.]\d{1,2})?", s):
        return float(s.replace(",", ".")), "parsed"
    if re.fullmatch(r"\d{1,3}(?:[ .]\d{3})+[,]\d{1,2}", s):
        return float(s.replace(" ", "").replace(".", "").replace(",", ".")), "parsed"
    return pd.NA, "ambiguous"


def core_flags(flags):
    """Drop ONLY tax warnings, never numeric/matching/territorial warnings."""
    if pd.isna(flags):
        return []
    result = []
    for token in str(flags).split(";"):
        token = token.strip()
        if not token or token.startswith("tax:"):
            continue
        result.append(token)
    return sorted(set(result))


def export_csv_and_parquet(df: pd.DataFrame, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8")
    try:
        df.to_parquet(stem.with_suffix(".parquet"), index=False)
    except ImportError:
        print("Note: pyarrow absent; CSV created, Parquet skipped for", stem.name)


def _read_candidate_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"No candidate file at {path}. First run: python code/situation_financiere.py build"
        )
    return pd.read_csv(path, dtype={"cassini_id": "string", "row_id": "string"}, low_memory=False)


def generate(input_df: pd.DataFrame):
    """Pure transformation, returning named result frames. Row loss is explicit and auditable."""
    df = input_df.copy()
    essentials = {"row_id", "year", "cassini_id", "commune_raw"}
    absent = essentials - set(df.columns)
    if absent:
        raise ValueError("Missing required candidate columns: " + ", ".join(sorted(absent)))
    if df["row_id"].isna().any() or df["row_id"].duplicated().any():
        raise ValueError("row_id must be nonmissing and unique: fix parser output before exporting")
    year = pd.to_numeric(df["year"], errors="coerce")
    if year.isna().any() or (year % 1 != 0).any():
        raise ValueError("year contains invalid values")
    df["year"] = year.astype("int32")
    # Treat the Cassini/noacass ID as a stable categorical ID, not a continuous numeric variable.
    id_str = df["cassini_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    good_id = id_str.str.fullmatch(r"\d+").fillna(False)
    df["cassini_id"] = pd.to_numeric(id_str.where(good_id), errors="coerce").astype("Int64")
    bad_ids = id_str.notna() & ~good_id
    if bad_ids.any():
        print(f"Warning: {int(bad_ids.sum())} invalid commune identifier(s), treated as unmatched")
    for field in FIELDS:
        if field not in df.columns:
            df[field] = pd.NA
    # Retain original text in candidate file, but emit strictly typed numeric columns.
    conversion_issues = []
    for field, kind in FIELDS.items():
        converted = [parse_number(value, kind) for value in df[field]]
        df[field] = pd.Series([value for value, _ in converted], index=df.index, dtype="Int64" if kind == "int" else "Float64")
        for idx, (_, status) in zip(df.index, converted):
            if status in {"ambiguous", "invalid"}:
                conversion_issues.append((idx, f"{field}:{status}"))
    issue_map = {}
    for idx, tag in conversion_issues:
        issue_map.setdefault(idx, []).append(tag)
    for c in ("extraction_flags", "review_flags"):
        if c not in df.columns:
            df[c] = ""
    duplicates = df["cassini_id"].notna() & df.duplicated(["year", "cassini_id"], keep=False)
    flags = []
    for idx, row in df.iterrows():
        tokens = core_flags(row["review_flags"]) + core_flags(row["extraction_flags"])
        tokens += issue_map.get(idx, [])
        if pd.isna(row["cassini_id"]):
            tokens.append("unmatched_commune")
        if duplicates.loc[idx]:
            tokens.append("duplicate_commune_year")
        flags.append(";".join(sorted(set(tokens))))
    df["core_review_flags"] = flags
    df["needs_core_review"] = df["core_review_flags"].ne("")
    df["n_core_fields_present"] = df[CORE_FIELDS].notna().sum(axis=1)
    df["complete_main_variables"] = df["n_core_fields_present"].eq(len(CORE_FIELDS))
    df["has_reviewed_layout"] = ~df["core_review_flags"].str.contains(r"(?:^|;)unmapped_columns(?:;|$)", regex=True)
    df["source_status"] = df.get("verification_status", pd.Series("ocr_unverified", index=df.index)).fillna("ocr_unverified")
    df["revenus_bureau_bienfaisance_francs"] = df["revenus_bureau_bienfaisance_raw"]
    df["panel_key"] = df["year"].astype(str) + ":" + df["cassini_id"].astype("string")
    if "commune_cleaned" in df.columns:
        df["commune_name"] = df["commune_cleaned"].fillna(df["commune_raw"])
    else:
        df["commune_name"] = df["commune_raw"]
    columns = [c for c in META_FIELDS if c in df.columns]
    ordered = columns + ["commune_name"] + EXPORTED_FIELDS + [
        "panel_key", "source_status", "n_core_fields_present", "complete_main_variables",
        "has_reviewed_layout", "core_review_flags", "needs_core_review",
    ]
    ordered = list(dict.fromkeys(ordered))
    candidates = df[ordered].copy()
    unmatch = candidates[df["cassini_id"].isna()].copy()
    dupl = candidates[duplicates].copy()
    matched_unique = df["cassini_id"].notna() & ~duplicates
    panel = candidates[matched_unique].copy().sort_values(["cassini_id", "year", "row_id"])
    # Strict is an optional clean subset, not a claim of manual transcription.
    strict = panel[panel["complete_main_variables"] & ~panel["needs_core_review"]].copy()
    tax_columns = [c for c in TAX_COLS if c in input_df.columns]
    tax = input_df[[c for c in ["row_id", "year", "cassini_id", "commune_raw", "source_image"] if c in input_df.columns] + tax_columns].copy()
    tax["tax_ocr_unverified"] = True
    diagnostics = []
    for y, block in df.groupby("year", sort=True):
        mask = df["year"].eq(y)
        sub = df.loc[mask]
        diagnostics.append({
            "year": y,
            "candidate_rows": len(sub),
            "unique_matched_communes": int(sub.loc[sub["cassini_id"].notna(), "cassini_id"].nunique()),
            "unmatched_rows": int(sub["cassini_id"].isna().sum()),
            "duplicate_candidate_rows": int(duplicates.loc[mask].sum()),
            "unique_panel_rows": int(matched_unique.loc[mask].sum()),
            "main_variables_complete": int((matched_unique.loc[mask] & sub["complete_main_variables"]).sum()),
            "strict_clean_rows": int((matched_unique.loc[mask] & sub["complete_main_variables"] & ~sub["needs_core_review"]).sum()),
            "core_review_rows": int(sub["needs_core_review"].sum()),
        })
    diag = pd.DataFrame(diagnostics)
    review = candidates[candidates["needs_core_review"] | ~matched_unique].copy()
    return {
        "commune_year_candidates": candidates,
        "commune_year_panel_provisional": panel,
        "commune_year_panel_strict": strict,
        "excluded_unmatched": unmatch,
        "excluded_duplicate_ids": dupl,
        "review_queue_core": review,
        "extraordinary_taxes_unverified": tax,
        "year_quality_summary": diag,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path("."), help="Path to main git repository")
    ap.add_argument("--input", type=Path, help="Override candidates CSV path")
    ap.add_argument("--out", type=Path, help="Override output folder (defaults to gitignored derived/)")
    args = ap.parse_args()
    root = args.repo.expanduser().resolve()
    source = args.input or root / "derived/situation_financiere/output/situation_financiere_commune_year_candidates.csv"
    dest = args.out or root / "derived/situation_financiere/econometric_export"
    results = generate(_read_candidate_file(source))
    for stem, frame in results.items():
        export_csv_and_parquet(frame, dest / stem)
    report = {
        "input_file": str(source), "outputs_dir": str(dest),
        "warning": "Unverified OCR-derived observations. Not a validated archival data release.",
        "candidate_rows": len(results["commune_year_candidates"]),
        "matched_unique_panel_rows": len(results["commune_year_panel_provisional"]),
        "strict_clean_subset_rows": len(results["commune_year_panel_strict"]),
        "unmatched_rows": len(results["excluded_unmatched"]),
        "duplicate_candidate_rows": len(results["excluded_duplicate_ids"]),
        "years": sorted(results["year_quality_summary"]["year"].astype(int).tolist()),
        "main_fields": CORE_FIELDS,
        "tax_data_policy": "Unverified, excluded from econometric panel, retained in separate file",
    }
    (dest / "export_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(results["year_quality_summary"].to_string(index=False))
    print(json.dumps(report, indent=2))
    if report["matched_unique_panel_rows"] == 0:
        print("WARNING: no matched IDs. Run build with commune matching enabled (without --no-match).")

if __name__ == "__main__":
    main()
