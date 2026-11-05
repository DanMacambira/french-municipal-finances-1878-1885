# Historical financial tables: image segmentation checkpoint

This checkpoint contains **two separate, local, deterministic stages**. Neither
calls OCR, cloud models, or paid services, and neither changes the source PDFs.

The accepted pilot layout for 1882 is one PDF spread -> two printed pages ->
two independent financial tables per printed page. The scan layout is
configured explicitly per year in
`schema/situation_financiere/page_ranges.json` (`single` or `spread`). The
original page ranges and exceptional `page_sides` are retained.

## Stage 1: identify the dark book gutter

Script: `code/split_spreads.py` (tests: `tests/test_split_spreads.py`)

```bash
.venv/bin/python code/split_spreads.py --year 1882
```

Configured spreads are split only when the dark gutter is convincingly detected within
45-55% of full scan width. The selected `--inner-trim .001` removes **0.1% of
full spread width from each inner edge** (the default). It avoids duplication
from the opposite printed page while trying not to clip content near the
binding. You can use `--inner-trim 0` to test without trimming.

Inputs: `raw_data/situation_financiere_communes/*.pdf` and the existing year
ranges in `schema/situation_financiere/page_ranges.json`.

Outputs: `derived/situation_financiere/split_pages/{year}/pXXXX_left_200dpi.png`,
`pXXXX_right_200dpi.png` for spreads, or `pXXXX_single_200dpi.png` for single
pages; plus `split_review.csv`, `review_contact_sheet.jpg`. Configured absent
sides are excluded. A flagged gutter requires human review.

## Stage 2: identify the two parallel tables on each printed page

Script: `code/split_tables.py` (tests: `tests/test_split_tables.py`)

```bash
.venv/bin/python code/split_tables.py --year 1882
```

The detector uses the printed central double rule and the wide NOMS column
in each image. In auto mode it tries a conservative linear or quadratic cut;
when evidence is insufficient, it keeps the original vertical cut. To disable
all fitting and reproduce the vertical method:

```bash
.venv/bin/python code/split_tables.py --year 1882 --cut-mode vertical
```

Inputs: all accepted stage-1 PNGs (`left`, `right`, or `single`) and `split_review.csv`.

Outputs: `derived/situation_financiere/table_panels/{year}/pXXXX_{side}_col{1,2}_200dpi.png`
(where side is `left`, `right`, or `single`), plus `table_split_review.csv`
and `review_contact_sheet.jpg`. A `needs_review` result
is **not** a validated crop, even if an earlier attempt was accepted.

## Verification before OCR

```bash
.venv/bin/python -m pytest -q tests/test_split_spreads.py tests/test_split_tables.py
open derived/situation_financiere/split_pages/1882/review_contact_sheet.jpg
open derived/situation_financiere/table_panels/1882/review_contact_sheet.jpg
```

Check the **actual table PNGs** near the binding and table separator, not only
the overlays. Page 244 of the 1882 pilot may still be flagged in stage 2: it
must not be silently accepted. Review all years before OCR: an automatic
`accepted` result is still a heuristic, not visual proof that no text was clipped. The legacy `table_panel_overrides` and
`table_panel_split_overrides` in the JSON belong to the older OCR workflow,
not to these two segmentation scripts.

The downstream Mistral OCR parser is intentionally **not wired into this
checkpoint**. Commit only the two scripts, their tests, and these docs, not
`derived/`, PDF scans or OCR data.

### Stage 2 robustness and all-years review (v1.3)

When the normal line-and-header detector fails, Stage 2 can try a more tolerant
local rule search near the median accepted boundary of similarly oriented
pages in the same year. This uses image evidence, not transferred coordinates.
Weak detections still fail closed. A strongly one-sided printed page is
preserved as a single table; missing table halves are not invented.

The year-level CSVs are automatically combined into
`derived/situation_financiere/table_panels/all_years_table_split_review.csv`.
Rebuild without processing scans with:

```bash
.venv/bin/python code/split_tables.py --combine-only
```

Review `detection_pass`, `status`, `table_count` and the preview paths in the
consolidated CSV. Run Stage 2 for all eight years after installing the patch.
The detector was regression-tested on synthetic variants and the available
full-size sample scans, **not** all historical page images; visually inspect
any newly recovered detections before OCR.


### Stage 2 extreme review-only fallback (v1.4)

Only after the strict and tolerant detectors fail, the new third pass follows
short local rule fragments through 17 horizontal bands. It tests vertical,
linear and quadratic paths with wider geometry limits, while still requiring
printed-rule evidence and a verified fiscal-columns-to-NOMS transition.
Successful extreme detections are **not** automatically promoted to OCR:
`accepted_extreme_pending_review` crops are stored separately in
`table_panels/{year}/extreme_candidates/` and logged in the annual and
all-years review CSVs. Existing `accepted` crops are left untouched.

Use `--no-extreme` to reproduce the previous v1.3 decisions. Review actual
full-resolution extreme images, not only the preview or algorithm score.

### Stage 2: short printed-area fallback (v1.5)

A fourth, review-only pass is attempted only when the previous three passes
fail. It detects the bottom of the two printed tables (which may occupy only
the upper part of a page), reruns the existing vertical/linear/quadratic rule
tracers within that area, and holds the separator constant through blank
paper below. The output is labelled `accepted_short_table_pending_review`
and placed in `table_panels/{year}/short_table_candidates/`, never in the
ordinary OCR-ready folder. A page without matching printed-rule evidence
remains `needs_review`.

The per-year and consolidated CSVs record the active table bounds. For the two
known short pages, run `--year 1881 --pages 466` and `--year 1883 --pages 502`.
Use `--no-short-table` to restore the v1.4 decision for a selected page.
The first three detectors and Stage 1 are unchanged.
