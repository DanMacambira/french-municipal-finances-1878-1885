# Automatic layout detection v1 (offline, diagnostic only)

This replaces the direction of manually calibrated per-page crop presets. It is a **first-stage geometry benchmark**, not a change to the financial parser.

It uses OpenCV to detect long vertical ruling lines, propose a separator between the two parallel printed tables, diagnose likely two-page scans, and mark tall repeated strokes that may be printed **braces/accolades** grouping extraordinary-tax entries.

All image coordinates are detected from the pixels; there are **no year/page-specific dimensions** in this module.

## Install

```bash
.venv/bin/python -m pip install 'opencv-python-headless>=4.8,<5'
```

Pillow and numpy are already present in the project. `opencv-python-headless` is an optional dependency for the diagnostic module; it is not required for the existing OCR/build pipeline.

## Usage

```bash
.venv/bin/python code/automatic_layout.py \
  --pdf raw_data/situation_financiere_communes/your_volume.pdf \
  --pages 245,246 \
  --output derived/situation_financiere/automatic_layout
```

You can also pass rendered images:

```bash
.venv/bin/python code/automatic_layout.py \
  --images derived/situation_financiere/images/1882/p0245_left_210dpi.png \
           derived/situation_financiere/images/1882/p0245_right_210dpi.png
```

Each image produces a `.json` proposal manifest and a `*_overlay.png`. Blue = candidate vertical rules, green = proposed split, orange/yellow = possible brace components. This command makes **zero API calls** and leaves all existing OCR cache/data unchanged.

## Current limitations and required independent benchmark

- A drawn brace may be disconnected or broken across scan damage; v1 returns component **proposals**, not complete braces, verified tax groups or commune assignments.
- The two-table split needs to be manually reviewed on a representative sample before being used for automatic paid OCR.
- Wide pages can be cover images rather than spreads; the spread detector is provisional.
- Columns are geometrical; semantic labels such as `centimes_extraordinaires` must later be assigned from headers or established repeated volume layouts.
- Source OCR HTML does **not** provide trustworthy individual word rectangles. A subsequent coordinate-aware OCR pass (e.g. Tesseract word boxes) is needed to associate recognized tax figures with brace vertical ranges and commune labels.
- Do not manually optimize this detector for a handful of pages. Establish a fixed multi-year holdout before changing any thresholds; report missed boundaries, cut-off text, and brace group accuracy.

Do **not** delete working `page_ranges.json` overrides yet. The next stage is comparing this detector with the existing checked crop coordinates on a broad holdout; only replace the old system once it demonstrably improves reliability.

## Single-entry straight bars versus multi-entry curved braces

This distinction is explicit in the experimental geometry matcher
`code/tax_group_geometry.py`:

* A straight vertical stroke is **not** a commune grouping marker. For a single
  tax, matching requires one-to-one vertical alignment between an independently
  located commune name and one numeric tax/year row.
* A **curved** brace can propose a multi-entry group if at least two paired
  tax/year rows lie within its span and exactly one commune-name position
  overlaps it. A curved-stroke candidate is not enough on its own: an erroneous
  or broken brace remains unresolved.
* If a bracket is missing and several tax lines are near the same commune,
  they remain unresolved rather than forcing multiple pairs into one row.
* Canton subtotals must be excluded from the tax-entry coordinates before
  grouping and used later as independent validation.

The matcher takes coordinate observations, *not manual crop bounds*. A future
coordinate-aware OCR/layout stage must supply the positions of commune-name
anchors and tax/year pairs. The current Mistral HTML output is insufficient
for safe automated attribution because its cell grouping can be wrong.

All statuses begin with `proposed_` or `needs_review`. This is not a production
corrector and does not modify existing extracted datasets. First benchmark
curly-brace detection and one-entry matching on untouched pages from multiple
volumes; then enable automatic writing only when error rates justify it.
