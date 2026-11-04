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

Every OCR row retains raw cells, page number, side (`left`/`right`), table ID, and original OCR payload. Map the actual printed fiscal headings *year by year* in `schema/situation_financiere/column_mappings.json` after visual inspection. Distinct layouts can coexist within one year. Unmapped fields remain preserved.

**Important:** OCR candidate values must not be treated as verified research data. They can contain wrong digits, misplaced columns, and row collisions. Build a manually transcribed benchmark before any public data release.

## Tests

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q tests/
```

## Data citation and licensing

The original scans come from the BnF; their reuse is governed by Gallica's terms, not by a software license placed on this repository. Gallica requires source attribution and sets special conditions for certain reuses. Research-oriented AI processing can be free subject to its conditions; commercial exploitation may require additional permission. Verify how these terms apply to *redistributed extracted data* before selecting a dataset license. The project code is available for inspection; an explicit software license and dataset license should be selected by the repository owner before public reuse/release.

For archival dataset releases, use versioned GitHub Releases and optionally connect the repository to [Zenodo](https://zenodo.org/) to obtain a DOI. A `CITATION.cff` should be added once the correct author(s) and citation details are decided.
