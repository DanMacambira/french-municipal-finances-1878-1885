# French municipal finances, 1878-1885

**Status: work in progress. No validated data are published.**

A reproducible workflow for digitizing commune-level fiscal tables in the French *Situation financière des communes* (1878-1885), starting with **Pas-de-Calais** and designed to support later extension to other departments.

Source: *Situation financière des communes*, Bibliothèque nationale de France (Gallica). **Source gallica.bnf.fr / Bibliothèque nationale de France.**

Original collection: https://gallica.bnf.fr/ark:/12148/cb412872434/date
Gallica reuse terms: https://gallica.bnf.fr/accueil/fr/html/conditions-dutilisation-de-gallica

## Repository policy

- Original national PDFs live in `raw_data/situation_financiere_communes/`, which is gitignored.
- The `derived/` folder (page images, API responses, candidate rows, review files) is gitignored.
- `data/` is for **reviewed and versioned** public research data only; currently empty except for documentation.
- No Mistral keys or environment files are committed.
- Do not commit source scans or OCR text dumps without reviewing the BnF terms (including terms specific to AI processing, noncommercial research, and commercial reuse).

## Installation

Create a local virtual environment (Python 3.12 or newer recommended):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Copy the annual PDF files into `raw_data/situation_financiere_communes/`, one file per year 1878-1885, with the year in the filename.

## Extraction

**Discover candidate Pas-de-Calais pages locally, without paid OCR:**

```sh
.venv/bin/python code/situation_financiere.py discover
```

If PDF text is poor or missing, install a local Tesseract binary with French language data, run `pip install pytesseract`, then:

```sh
.venv/bin/python code/situation_financiere.py discover --header-ocr
```

Open `derived/situation_financiere/discovery_report.csv` and review each range in `schema/situation_financiere/page_ranges.json`. **Manually confirm PDF-page start and end, then set `confirmed: true` for each year.** Page numbering is 1-based PDF index, not printed volume pagination.

**Preview proposed pages:**

```sh
.venv/bin/python code/situation_financiere.py preview --years 1881 --center 250 --radius 6
```

**Estimate requests (after confirming ranges):**

```sh
.venv/bin/python code/situation_financiere.py plan
```

**Run a small Mistral pilot:**

```sh
export MISTRAL_API_KEY='YOUR_KEY'
.venv/bin/python code/situation_financiere.py ocr --years 1881 --max-pages 3
.venv/bin/python code/situation_financiere.py build --years 1881 --no-match
```

Avoid pasting keys into shared terminal logs or committing them. The workflow sends **cropped Pas-de-Calais page images**, never the full national PDFs, and caches OCR responses locally to avoid duplicate charges. It uses the standard synchronous OCR endpoint, **not** the discounted batch API.

**Add historical commune matching (optional).** The file `code/commune_matcher.py` is copied from the related school project. It depends on an external `communematch` package and a locally provisioned `raw_data/COMMUNES_ref.xlsx` containing the `noacass` identifier and historical name variants. Both are intentionally *not* bundled here. You can use `--no-match` while validating the table parser. Once those prerequisites are installed and sourced correctly, run `build` without `--no-match`.

**Process all confirmed pages:**

```sh
.venv/bin/python code/situation_financiere.py ocr --yes-all
.venv/bin/python code/situation_financiere.py build --no-match
```

Review `derived/situation_financiere/output/situation_financiere_review_queue.csv` and the complete `situation_financiere_all_rows.csv` before publishing. `column_mappings.json` intentionally starts with *no active mappings*; this prevents incorrect year-specific labels being silently assigned.

## Year-specific tables

Every OCR row retains raw cells, page number, region (`full`/`left`/`right`), table ID, and original OCR payload. Map the actual printed fiscal headings *year by year* in `schema/situation_financiere/column_mappings.json` after visual inspection. Distinct layouts can coexist within one year. Unmapped fields remain preserved.

**Important:** OCR candidate values must not be treated as verified research data. They can contain wrong digits, misplaced columns, and row collisions. Build a manually transcribed benchmark before any public data release.

## Tests

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q tests/
```

## Data citation and licensing

The original scans come from the BnF; their reuse is governed by Gallica's terms, not by a software license placed on this repository. Gallica requires source attribution and sets special conditions for certain reuses. Research-oriented AI processing can be free subject to its conditions; commercial exploitation may require additional permission. Verify how these terms apply to *redistributed extracted data* before selecting a dataset license. The project code is available for inspection; an explicit software license and dataset license should be selected by the repository owner before public reuse/release.

For archival dataset releases, use versioned GitHub Releases and optionally connect the repository to [Zenodo](https://zenodo.org/) to obtain a DOI. A `CITATION.cff` should be added once the correct author(s) and citation details are decided.

## Single-page scans versus double-page spreads

PDF pages containing a single printed page are sent as **one complete image**.
PDF pages containing a two-page spread are split into **left** and **right**
printed-page images. The originals remain unchanged and stay out of Git.

By default, the pipeline guesses the type from the PDF page's aspect ratio:
width > 1.05 * height is treated as a spread; otherwise as a single page.
This heuristic does **not** inspect the scanned content. Inspect the output of
`make plan` and the actual crop previews before running paid OCR:

```sh
make plan
make preview SF_ARGS="--years 1882 --center 244 --radius 2 --regions"
```

The `--regions` option creates a contact sheet showing exactly the `full`,
`left`, or `right` images the script would send to OCR, without making API
calls. It is saved under `derived/situation_financiere/previews/`.

Override the automatic detection, if needed, with the optional keys below
in your **existing confirmed** `schema/situation_financiere/page_ranges.json`
entry. Keep your manually verified `start` and `end` values unchanged:

```json
"1882": {
  "pdf": "La_Situation_financière_des_communes_1882.pdf",
  "start": 244,
  "end": 252,
  "confirmed": true,
  "scan_layout": "spread",
  "page_sides": {"244": ["right"]},
  "scan_layout_overrides": {"250": "single"}
}
```

- `scan_layout`: `auto` (default), `single`, or `spread`, applied to that year.
- `scan_layout_overrides`: change the layout for individual **1-based PDF pages**.
- `page_sides`: on a spread, specify the printed pages that actually belong
  to Pas-de-Calais; e.g. `["right"]` when the left side belongs to Orne.

The override example is illustrative: do **not** classify page 250 as single
unless you have verified it visually. `page_sides` is invalid on single scans.
Unspecified spreads use both halves. Source provenance keeps the original PDF
page number and OCR region (`full`, `left`, `right`).

## Parallel tables within a printed page (two-column OCR)

A *printed page* may contain two independent side-by-side commune tables. OCR
of the complete printed page can merge them into a single 18-column table or
repeat/lose commune names. Set `table_panels: 2` in the corresponding **year's
existing confirmed** `schema/situation_financiere/page_ranges.json` entry:

```json
"1881": {
  "pdf": "La_Situation_financière_des_communes_1881.pdf",
  "start": 452,
  "end": 466,
  "confirmed": true,
  "table_panels": 2
}
```

These numbers are illustrative of the current project; **retain your own
verified ranges and all existing page_sides entries**. `table_panels` defaults
to 1. Optionally set `table_panel_overrides: {"455": 1}` if a specific PDF
page has only one table block.

When `table_panels` is 2:

- A single-page scan produces `full_col1`, `full_col2` image/JSON pairs.
- A double-page spread produces `left_col1`, `left_col2`, `right_col1`,
  `right_col2`, subject to `page_sides` selection.
- `make plan` counts these as separate OCR requests.
- `make preview SF_ARGS="--years 1881 --center 455 --radius 0 --regions"`
  shows **exactly** the new crops and makes **no API requests**.
- After updating the config, **old whole-page OCR JSON is retained but
  automatically excluded from `build`** to avoid duplicate/contaminated rows.

Try the new layout on PDF page 455 before processing other pages:

```sh
make plan
make preview SF_ARGS="--years 1881 --center 455 --radius 0 --regions"
make ocr SF_ARGS="--years 1881 --pages 455 --dpi 300"
make build SF_ARGS="--years 1881 --no-match"
```

For research reliability, compare the resulting HTML data against the
**original image**: count unique communes in each column and manually check
names plus numerical cells. Model confidence is not a substitute for audit.
Do not launch the full run until the pilot is satisfactory.

## Two independent commune tables per printed page

The eighteenth- and nineteenth-century table pages can have **two parallel
nine-column commune blocks**. Mistral sometimes merges both blocks into an
18-column HTML table, losing rows and misaligning fiscal values. This is a
**separate problem** from two-page PDF scans: a `spread` contains two printed
pages, each of which can itself contain two table panels.

After visually verifying, add `"table_panels": 2` to a year's **existing**
confirmed entry in `schema/situation_financiere/page_ranges.json`; do **not**
overwrite your start/end ranges or `page_sides`. The optional overrides are:

```json
"table_panels": 2,
"table_panel_overrides": {"245": 1},
"table_panel_split_overrides": {"p245_left": 0.521, "p245_right": 0.485}
```

The values above illustrate a configuration; `table_panel_overrides` should
only be supplied for exceptional PDF pages with a single printed table. The
split fractions are measured from the **start of each printed page**, not the
full two-page spread. Default split is `0.5`; use a custom fraction only if
`preview --regions` shows columns being cropped incorrectly.

- Single PDF page with two table panels => `full_col1` and `full_col2`.
- Two-page spread with two panels per printed page => `left_col1`,
  `left_col2`, `right_col1`, `right_col2`, except excluded `page_sides`.
- Original PDFs are never changed, and prior JSON responses remain cached.
- `build` excludes old, now-invalid whole-page OCR and changed crop versions.
- A preview can regenerate panel images without making API calls.

Pilot procedure (one 1882 PDF spread):

```sh
make plan
make preview SF_ARGS="--years 1882 --center 245 --radius 0 --regions"
make ocr SF_ARGS="--years 1882 --pages 245"
make build SF_ARGS="--years 1882 --no-match"
```

**The pilot is not yet research-grade.** Examine the HTML output, record count,
and the unusual *centimes* columns (often stacked lines/braces) against the
original scans. In particular, OCR confidence scores cannot certify numerical
alignment or recover physically illegible values.
