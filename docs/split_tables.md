# Stage 2: split each printed page into its two tables

**Purpose:** use the printed-page PNGs saved by stage 1 (`code/split_spreads.py`)
and identify the **central double vertical separator** between the parallel
tables. No model, paid OCR, coordinate templates, or internet access required.

## Run from your repository root

This second-stage tool is independent of stage 1 and existing OCR. The only
shared interface is the printed-page PNG output plus the stage-1 review CSV.

Test a small set of 1882 scans:

```bash
.venv/bin/python code/split_tables.py --year 1882 --pages 245 247 249
```

Review:

```bash
open derived/situation_financiere/table_panels/1882/review_contact_sheet.jpg
```

If the magenta lines fall between the two printed tables, run all 1882 scans:

```bash
.venv/bin/python code/split_tables.py --year 1882
```

Run the included tests (without doing any OCR):

```bash
.venv/bin/python -m pytest -q tests/test_split_tables.py
```

## Why this differs from our failed layout/template tools

- The same fixed rectangle is **not** transferred from page to page.
- The detector runs **independently** on each already separated printed page.
- It finds long vertical table rules within 42-58% of each image width.
- It checks a distinguishing visual signature: **narrow fiscal columns to the
  left** of the divider, then **the wide 'NOMS' column to the right**, with no
  internal long vertical rules in its upper header.
- It rejects weak or competing matches, rather than guessing a crop.
- It also checks the vertical rule in four height bands and flags strong
  curvature/position changes instead of cropping characters with a bad straight cut.
- It will not process stage-1 pages marked uncertain in `split_review.csv`
  unless explicitly forced for review.
- It uses a small 0.4% overlap at the divider to avoid deleting border ink while
  minimizing text from the adjacent table in each OCR image.
- Printed-page curvature can still cause a **straight cut** to cross characters.
  The acceptance flag is a heuristic, not a guarantee that a scan is readable.

## Output paths

- `derived/situation_financiere/table_panels/1882/p0245_left_col1_200dpi.png`
- `derived/situation_financiere/table_panels/1882/p0245_left_col2_200dpi.png`
- `derived/situation_financiere/table_panels/1882/p0245_right_col1_200dpi.png`
- `derived/situation_financiere/table_panels/1882/p0245_right_col2_200dpi.png`
- `derived/situation_financiere/table_panels/1882/previews/` (magenta cut lines)
- `derived/situation_financiere/table_panels/1882/review_contact_sheet.jpg`
- `derived/situation_financiere/table_panels/1882/table_split_review.csv`

The stage 1 `page_sides` convention is retained automatically: if a source has
only a right printed page, only that source is processed, creating two tables.

**Important:** this tool deliberately does not call Mistral or modify
`code/situation_financiere.py`. Evaluate 1882 and at least one other year
before adopting the outputs for OCR. Change source segmentation only after QA.

## Version 1.1: adaptive slanted / gently curved separator

`--cut-mode auto` (the default) now samples the previously detected central
printed rule in 15 horizontal bands and compares a robust linear and quadratic
fit against the **original straight vertical cut**. A fitted curve is used only
when it is supported by many observations, materially improves the fit, stays
near the baseline boundary, and does not compete with another possible table
separator. A quadratic must improve noticeably on a line. No model inference,
external API, reference template, or per-image hand tuning is involved.

The code does **not rotate or dewarp** the page; it copies each row up to its
locally fitted cut and pads the remainder white to form rectangular OCR images.
This keeps all printed figures at their original pixel positions while
allowing the separator to slant or curve. The 0.4%-of-printed-page-width overlap
at the table border is unchanged; it preserves a small safety margin around
rules rather than the neighboring text.

The old algorithm remains available *byte for byte* when the fit is not
reliable. To force the old strictly vertical table cut on all accepted pages:

```bash
.venv/bin/python code/split_tables.py --year 1882 --cut-mode vertical
```

Normal test after re-running stage 1:

```bash
.venv/bin/python code/split_tables.py --year 1882 --pages 245 247 249
open derived/situation_financiere/table_panels/1882/review_contact_sheet.jpg
```

Previews show the **magenta chosen cut** and, when a fitted path is used,
**orange dashes for the former vertical cut**. CSV fields `cut_method`,
`cut_path_coefficients`, `cut_path_points`, `cut_path_residual_fraction`,
`cut_path_max_shift_fraction` and `cut_path_reason` explain each decision.
If the original detector said only that the rule drift was too large, a highly
consistent fitted path may recover it. Unidentified/competing separators remain
`needs_review`, regardless of polynomial fit. Review crops visually before OCR:
none of these confidence checks guarantees that text is uncut.
