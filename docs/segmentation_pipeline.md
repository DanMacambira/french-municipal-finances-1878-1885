# Historical financial tables: image segmentation checkpoint

This checkpoint contains **two separate, local, deterministic stages**. Neither
calls OCR, cloud models, or paid services, and neither changes the source PDFs.

The accepted pilot layout for 1882 is one PDF spread -> two printed pages ->
two independent financial tables per printed page. The underlying year/page
mapping remains in `schema/situation_financiere/page_ranges.json` and may need
further verification before running on every year. No mapping is changed here.

## Stage 1: identify the dark book gutter

Script: `code/split_spreads.py` (tests: `tests/test_split_spreads.py`)

```bash
.venv/bin/python code/split_spreads.py --year 1882
```

Scans are split only when the dark gutter is convincingly detected within
45-55% of full scan width. The selected `--inner-trim .001` removes **0.1% of
full spread width from each inner edge** (the default). It avoids duplication
from the opposite printed page while trying not to clip content near the
binding. You can use `--inner-trim 0` to test without trimming.

Inputs: `raw_data/situation_financiere_communes/*.pdf` and the existing year
ranges in `schema/situation_financiere/page_ranges.json`.

Outputs: `derived/situation_financiere/split_pages/{year}/pXXXX_{side}_200dpi.png`,
`split_review.csv`, `review_contact_sheet.jpg`. Configured absent sides must
not be processed. A flagged gutter requires human review.

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

Inputs: stage-1 PNGs and `split_review.csv`.

Outputs: `derived/situation_financiere/table_panels/{year}/pXXXX_{side}_col{1,2}_200dpi.png`,
`table_split_review.csv`, `review_contact_sheet.jpg`. A `needs_review` result
is **not** a validated crop, even if an earlier attempt was accepted.

## Verification before OCR

```bash
.venv/bin/python -m pytest -q tests/test_split_spreads.py tests/test_split_tables.py
open derived/situation_financiere/split_pages/1882/review_contact_sheet.jpg
open derived/situation_financiere/table_panels/1882/review_contact_sheet.jpg
```

Check the **actual table PNGs** near the binding and table separator, not only
the overlays. Page 244 of the 1882 pilot may still be flagged in stage 2: it
must not be silently accepted. The future year-wide run needs a separate
review of source page ranges, spread/single-page layouts and boundary pages.

The downstream Mistral OCR parser is intentionally **not wired into this
checkpoint**. Commit only the two scripts, their tests, and these docs, not
`derived/`, PDF scans or OCR data.
