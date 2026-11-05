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

- `derived/situation_financiere/table_panels/1882/accepted/p0245_left_col1_200dpi.png`
- `derived/situation_financiere/table_panels/1882/accepted/p0245_left_col2_200dpi.png`
- `derived/situation_financiere/table_panels/1882/accepted/p0245_right_col1_200dpi.png`
- `derived/situation_financiere/table_panels/1882/accepted/p0245_right_col2_200dpi.png`
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

The input glob accepts `pXXXX_single_<dpi>dpi.png` alongside spread half-images.
Both formats use the same independent two-table separator detector.

## Version 1.3: conservative second pass and consolidated review

The existing strict divider detector remains **first**. When it cannot locate the
separator, the script learns a **position prior** from successful pages in the
same annual volume and orientation: the printed `left` or `right` spread side,
or the PDF-page parity for scanned single pages. This prior merely restricts
where a *new image-based search* takes place. It does not itself justify a
crop. The second search looks for a nearby printed vertical rule and the
transition from multiple narrow fiscal columns to the wide `NOMS` column,
using a shorter line detector and several vertical bands (including lower
headers on introductory pages). Weak, conflicting or unconfirmed evidence
remains `needs_review`. An originally accepted cut is never replaced by the
second detector. The existing slanted/quadratic fit and vertical fallback
remain unchanged.

When one parallel half is **demonstrably almost blank** and the other contains
printed long rules and ink, the script may set `accepted_single_table` and
save only the existing table (as `col1` or `col2`). It saves the full printed
page image for that table instead of trimming at an unsupported boundary.
This safeguard does **not** infer a single table merely because the ordinary
separator detector fails; ambiguous pages remain `needs_review`.

The report adds `detection_pass` (`strict`, `tolerant`, `single_table`),
`expected_cut_fraction`, `table_count`, and `single_table_side`. In particular,
a provisional 50% `cut_fraction` in a `needs_review` row is **not** a verified
cut. `accepted` is still a detector decision, not manual ground truth.

Every yearly run also refreshes:

`derived/situation_financiere/table_panels/all_years_table_split_review.csv`

This CSV is simply all available annual `table_split_review.csv` rows together,
sorted by year, PDF page and printed side. It includes rejected pages. To
rebuild this file from existing annual CSVs *without rerunning any images*:

```bash
.venv/bin/python code/split_tables.py --combine-only
```

Rerunning a subset with `--pages` preserves previous CSV rows for pages not
selected. The contact sheet from that subset shows only pages processed in
that run; run the full year again for a complete year contact sheet. To update
all years and rebuild the aggregate:

```bash
for year in 1878 1879 1880 1881 1882 1883 1884 1885; do
  .venv/bin/python code/split_tables.py --year "$year"
done
.venv/bin/python code/split_tables.py --combine-only
```

**No Mistral calls are made.** Check actual full-size crops for any newly
recovered pages before feeding them to OCR. Some genuinely exceptional or
occluded pages should remain flagged; do not force them into two outputs.


## Version 1.4: extreme third pass (pending visual approval)

If both the original detector and the tolerant second pass fail, **and** a
stable same-year/same-orientation prior exists, an optional third pass can
trace the double rule through 17 short horizontal bands. This permits the
rule to move considerably from left to right with height. It compares
**vertical, linear, and quadratic** fits, with wider thresholds than stages
1-2; it still needs (a) local printed-rule pixels over most of the page,
(b) at least two upper-page indications of narrow fiscal columns followed by
the wide `NOMS` column, and (c) a stable fit that does not match a competing
central rule. An estimated position by itself **never** authorizes a cut.
The previous decisions of already-accepted pages are not recomputed or
modified by the extended pass.

A recovered page receives:

- `status=accepted_extreme_pending_review`
- `detection_pass=extreme`
- `cut_method=vertical`, `linear` or `quadratic`

This is **not OCR approval**. Its two unverified images are saved under
`derived/situation_financiere/table_panels/{year}/needs_review/`, outside
the ordinary production crop directory. The contact-sheet preview is tagged
`EXTREME PENDING REVIEW` and the annual/combined CSVs give their paths.
If the extreme fit is unsupported or ambiguous, the page stays `needs_review`.
The detector does not force an invented second table on a one-table page.

Try the six previously unresolved cases without overwriting other annual CSV
rows (run Stage 1 first, if needed):

```bash
.venv/bin/python code/split_tables.py --year 1881 --pages 466
.venv/bin/python code/split_tables.py --year 1883 --pages 495 497 499 502
.venv/bin/python code/split_tables.py --year 1884 --pages 501
.venv/bin/python code/split_tables.py --combine-only
```

Inspect each contact-sheet preview and **full-resolution image in
`needs_review/`** before deciding it can be used for Mistral.
To return to the v1.3 detection behavior, use `--no-extreme`. To disable all
curve fitting as well, use `--cut-mode vertical` (which also disables the
extreme fit). Re-running a page with `--no-extreme` clears stale experimental
candidate crops for that page. No Mistral, PDFs, or schema settings are
modified by this patch.

## Version 1.5: fourth pass for tables ending above the page bottom

The first three detectors are **unchanged**. Only a page that is still
`needs_review` may enter the short-table fallback. The fallback must:

1. Find long printed column-rule activity on **both** sides of the expected
   separator, excluding dark page margins. Both sides must begin near the top
   and end together well above the physical page bottom.
2. Crop **in memory** to that observed table height and retry the existing
   strict/tolerant vertical, linear and quadratic algorithms. A positional
   prior alone cannot authorize an output.
3. Require at least ten coherent separator samples in this printed region.
   Extend the selected boundary constantly through the blank remainder,
   instead of extrapolating an unsupported slope or quadratic.

Recovered pages receive `status=accepted_short_table_pending_review` and
`detection_pass=short_table`. Their two crops are saved under
`table_panels/{year}/needs_review/` instead of the ordinary
production output directory. They require a visual check **before OCR**.
The annual and all-years CSVs also record `active_table_top_fraction` and
`active_table_bottom_fraction` for these pages. All other rows leave those
fields empty.

To test the two known remaining cases (without rerunning Stage 1):

```bash
.venv/bin/python code/split_tables.py --year 1881 --pages 466
.venv/bin/python code/split_tables.py --year 1883 --pages 502
.venv/bin/python code/split_tables.py --combine-only
```

If a short table is not independently supported by printed rules, it remains
`needs_review`. Use `--no-short-table` to reproduce the v1.4 decisions for a
selected page (and remove stale short-table candidate crops). Previously
accepted standard and extreme results are left untouched. `--cut-mode vertical`
also disables this fourth-pass experiment.

No OCR or schema changes occur in this release. The algorithm has synthetic
regression tests, but the **original full-resolution scans of these two pages
still need human visual verification**.


## v1.6: parallel runs, visual review queues and statistics

`--workers N` parallelizes the unchanged separator detectors across source
pages (default up to four worker processes). The initial strict detections
are evaluated across the whole year first so that the tolerant detector's
layout priors remain identical to a serial run. Each worker limits OpenCV
internal threads to one. This keeps outputs reproducible and avoids oversubscribing
CPU resources. On memory-constrained Macs, use `--workers 1`.

Directory layout for each year:

- `table_panels/{year}/accepted/`: automatically accepted table crops **only**.
- `table_panels/{year}/needs_review/`: full original printed-page PNGs for
  unresolved cases, plus separately held provisional/extreme/short-table crops,
  and their own individual previews and `review_contact_sheet.jpg`.
- `table_panels/{year}/previews/`: diagnostic overlays for accepted cuts.
- `table_panels/{year}/review_contact_sheet.jpg`: visual overview, paginated
  after 48 items (`review_contact_sheet_002.jpg`, etc.).
- `table_panels/{year}/table_split_review.csv` and
  `table_panels/all_years_table_split_review.csv`: unchanged audit reports.

A page marked `accepted_extreme_pending_review` or
`accepted_short_table_pending_review` is **not** treated as OCR-ready. Its crop
is placed in `needs_review/` until a separate manual approval process is
implemented. `--pages` reruns preserve unaffected review CSV rows. The terminal
shows only progress bars/timers and an end-of-run table with status, cut method
(vertical/linear/quadratic) and detection pass (strict/tolerant/extreme/short).
The scan detection and path-fitting thresholds are unchanged from v1.5.
