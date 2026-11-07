# Provisional commune-year export (post-processing v2)

This add-on is a **non-destructive, provisional** post-processing pass of the
already-matched OCR candidates for *Situation financière des communes*. It
makes no additional Mistral calls and makes no changes to `communematch`.

## Why this exists

The existing exporter excludes all duplicate Cassini/year observations and all
unmapped table layouts. This loses otherwise usable numbers when the same
commune has repeated OCR rows because its extraordinary taxes span lines.

The new exporter:

1. Reads `derived/situation_financiere/output/situation_financiere_commune_year_candidates.csv`.
2. Validates the positions of the five main financial columns from already
   mapped rows **separately for every year**. Requires at least 25 numeric
   anchors per field and >= 99.5% agreement; if not satisfied, it refuses to
   infer values for unmapped rows in that year.
3. Recovers plausible leading-column values for *unmapped* rows only in years
   that pass this validation. It does not assume that the extraordinary-tax
   columns have fixed positions. OCR ambiguity is never guessed as a number.
4. Groups by `(year, cassini_id)` and merges duplicate rows **only when every
   nonblank main numeric field agrees**. If any main value conflicts, **the
   entire ID-year group is excluded from the panel and written to review**.
5. Saves all unmatched/ambiguous rows; does not drop them silently. Does not
   apply sample selection or attempt to reassign wrong historical names.
6. Preserves original row IDs and image references, and keeps all results
   marked provisional/unverified.

## Install (automatic update installer)

Put `commune_postprocessing_update.pyz` in Downloads and run:

```bash
python3 ~/Downloads/commune_postprocessing_update.pyz --run --cleanup
```

It locates `~/dev/french-municipal-finances-1878-1885`, adds:

- `code/postprocess_communes.py`
- `tests/test_postprocess_communes.py`
- `README_POSTPROCESSING_V2.md`

It backs up existing files when replacing them and runs offline tests. It
**never** modifies the original candidate CSV or the original exporter.

## Run again without reinstalling

```bash
cd ~/dev/french-municipal-finances-1878-1885
.venv/bin/python code/postprocess_communes.py --repo .
```

Output directory:

`derived/situation_financiere/econometric_export_v2/`

- `provisional_panel.csv/.parquet`: one row per nonconflicting matched ID/year;
  may include missing main fields.
- `complete_core_panel.csv/.parquet`: subset with all five main financial
  variables. **Still OCR-unverified**.
- `year_quality_v2.csv`: by-year coverage and recovery.
- `conflicting_ids.csv`: groups with conflicting core numbers.
- `conflicting_candidate_rows.csv`: original candidate rows for those groups.
- `unmatched_candidate_rows.csv`: NO_MATCH and AMBIGUOUS rows.
- `unmapped_layout_review.csv`: all original unmapped-layout candidate rows,
  including recovered values for audit.
- `numeric_parse_discrepancies.csv`: mapped values differing from a conservative
  read of the corresponding raw cell (do not auto-correct without checking).
- `layout_validation.csv`: proof that leading numeric fields are aligned in
  each year before any recovery.
- `report.json`: machine-readable diagnostics.

## Important limitations

- **No values have been checked against original scans by this script.**
  `complete_core_panel` does not mean *verified*.
- A matched Cassini ID can still be wrong if the OCR name was wrong; repeated
  names assigned to several different numeric profiles are excluded as
  conflicts, but single misidentifications may remain.
- The six pending segmentation-review pages remain outside the OCR caches.
- The extraordinary-tax data are not repaired or used here.
- For a new department/year/layout, the inference rule must pass the internal
  anchor agreement check. It does not hardcode any individual page number.
- When 1878-1900 data become available, also validate temporal continuity,
  commune territorial changes, and financial plausibility before analysis.
