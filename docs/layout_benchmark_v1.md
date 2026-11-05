# Automatic layout: reproducible multi-year offline benchmark

This is the next diagnostic step after `automatic_layout_and_tax_grouping_v1.patch`.
It adds a **batch benchmark runner** without changing Mistral OCR, the extraction
parser, the existing page-range configurations, or any production data.

It is designed to detect and report when geometry *looks* plausible. It **does
not** measure true correctness until a human reviews the diagnostic overlays.

## Install the optional image-processing dependency

```bash
.venv/bin/python -m pip install -r requirements-layout.txt
```

## Run from the repository root

```bash
# Select and freeze 5 PDF pages per year (40 total; 3 development and 2 holdout/year)
.venv/bin/python code/layout_benchmark.py --plan-only

# Offline: render those pages, diagnose gutters/table rules/braces, save overlays
.venv/bin/python code/layout_benchmark.py
```

The runner reads the already-confirmed `schema/situation_financiere/page_ranges.json`
and the corresponding files from `raw_data/situation_financiere_communes/`.
It uses **no manual crop coordinates, API keys, internet requests, or paid OCR**.
Every year contributes start, midpoint, end, and deterministically selected
additional PDF pages. The selection is written to `sample_plan.json` before any
layout detection, and is reused on subsequent runs. A subset is labeled
`holdout`: do not adjust detection thresholds using those pages.

If you want a smaller local smoke test first, use a separate output directory:

```bash
.venv/bin/python code/layout_benchmark.py --years 1881 1882 --per-year 2 \
  --output derived/situation_financiere/layout_benchmark_smoke
```

Do not use smoke-test pages as evidence of eight-year accuracy.

## Outputs

Under `derived/situation_financiere/layout_benchmark/`:

- `sample_plan.json` - frozen page selection including development/holdout role.
- `overlays/<year>/..._page.jpg` - whole-page view with detected spread gutter.
- `overlays/<year>/..._<side>_overlay.jpg` - independent table/brace
  proposals for each printed page; **both sides of spreads** are inspected.
- `diagnostics/<year>/*.json` - original pixel-space geometry proposals.
- `diagnostic_manifest.csv` - one row per printed region, normalized split
  coordinates, detection status, candidate brace counts and overlay paths.
- `diagnostic_summary.json` - counts of candidate/unresolved splits, **not**
  ground-truth accuracy.
- `human_review.csv` - one row per PDF page to assess page/gutter splits,
  printed-table separator, clipping and brace proposals. This CSV is **never
  overwritten** when rerunning with the same page sample.

`page_sides` exclusions from the existing configuration are recorded in the
manifest; excluded regions are not counted as successful table detection.

## Human verification and real accuracy

Inspect the `*_page.jpg` and `*_overlay.jpg` images. For each PDF page, fill
`human_review.csv` using `yes`, `no`, or `na`; leave unreviewed cells blank.
For a single printed page, `gutter_ok` is normally `na`. Set the remaining
metrics by examining the full page and both table panels. The brace field can
be `na` if no true brace is present.

```bash
.venv/bin/python code/layout_benchmark.py --score-only
```

This writes `human_validation_summary.csv` with the denominator of reviewed
checks and separate development vs holdout and per-year success rates. Missing entries are
not counted as correct; the result is blank if no entries are reviewed.

**Caution:** The prototype finds long vertical rules and candidate brace-like
marks, not final column names or definitive brace-to-commune assignments.
The detector must show good holdout results before its proposals are used for
production crops. Do not report these scans as fully extracted data.

## Reproducibility

Use the same sample and output folder for follow-up testing. An existing
sample plan is frozen; the runner refuses different years, seed, sample count,
or page-range configuration. A deliberate new experiment may use a different
`--output` directory (preferred), or `--reset-plan` after archiving the original.
No PDF is copied to Git, and everything in `derived/` should remain ignored.
