# Geographic labels and canton subtotal checks

The build step is **offline**. It consumes existing cached OCR JSON files; it does not send a new request to Mistral.

## Outputs (in `derived/situation_financiere/output/`)

- `situation_financiere_commune_year_candidates.csv`: candidate financial rows now carry `arrondissement`, `canton`, `canton_segment_id`, and numbered extraordinary-tax pairs. Geographic labels are normalized OCR headings, **not independently verified historical identifiers**.
- `situation_financiere_all_rows.csv`: every OCR table row, including canton headings, subtotal rows, and tax-only continuations, with the geographic context inferred from source order.
- `situation_financiere_canton_validation.csv`: one check per canton and additive numerical field, comparing sums of commune OCR rows with the **OCR-transcribed** printed subtotal. Does not correct values automatically.
- `situation_financiere_extraordinary_tax_entries.csv`: one row per OCR tax pair, with original row ID, geographic context, and linkage status.
- `situation_financiere_unassigned_tax_rows.csv`: detached tax continuations, preserved but not automatically attributed to a commune.
- Existing column-inventory, page-audit, QC and review-queue files remain available.

## Interpretation of canton checks

A check is attempted only when the OCR contains both an explicit canton beginning and a single subtotal, with no known gaps in the intervening OCR regions.

- `ocr_sums_match`: the OCR sum equals the **OCR** subtotal. This is not proof that either is correctly read from the scan.
- `ocr_discrepancy_review_scan`: the numbers differ. Inspect the source scan, because the subtotal itself may be misread.
- `not_checked_incomplete_canton`: the canton start or subtotal is unavailable, multiple subtotals were identified, or an intervening OCR region is missing.
- `not_checked_unmapped_column`, `not_checked_missing_commune_values`, `not_checked_unreadable_subtotal`: no valid comparison for that field.

Only `population`, `superficie_hectares`, `revenus_annuels_francs`, and `centimes_total` are automatically added. The *valeur du centime* and tax expiry years are **not additive**. Do not use extraordinary-tax subtotals until tax continuation rows are reliably assigned. A canton spanning more than one printed page can be validated when all relevant OCR regions are present.

A start marked `(suite)` without an earlier printed start is treated as an incomplete canton. In the 1882 PDF page 245 pilot, Bapaume begins earlier and Marquion continues later, so neither should be reported as a complete validated canton.

The recognition of headings relies on Mistral text. An absent or hallucinated heading can misassign territorial context; the row ID, source image, PDF page and `canton_heading_status` are retained to support checking. Similar names in `(suite)` headings may be associated with a previous canton only when the match is very close; this is identified as `continuation_ocr_name_disagrees`.

## Run

```bash
make test
make build SF_ARGS="--years 1882 --no-match"
```

Check `situation_financiere_canton_validation.csv` and the corresponding scan crops, without treating any matched totals as sufficient verification of individual commune values.
