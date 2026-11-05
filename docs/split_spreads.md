# Split scanned spreads into two printed-page images (experimental)

This first-stage tool is independent of Mistral OCR and table extraction. It never
changes source PDFs or existing OCR caches.

## From your repository root (Intel Mac; no Docker)

Install no extra packages if your original pipeline works: it uses PyMuPDF,
Pillow and NumPy, all already present in the OCR project's Python environment.

To test three 1882 PDF pages (PDF page numbers, not printed page numbers):

```bash
.venv/bin/python code/split_spreads.py --year 1882 --pages 245 247 249
```

To process all confirmed 1882 pages after reviewing the pilot:

```bash
.venv/bin/python code/split_spreads.py --year 1882
```

To also emit one-page PDFs (PNG is sufficient for OCR):

```bash
.venv/bin/python code/split_spreads.py --year 1882 --pdf
```

## How the split works

* The candidate cut is restricted to the **central 45-55%** of scan width.
* For every candidate column, the tool measures the proportion of dark pixels
  across most of the page height. Horizontal smoothing reduces false positives
  from narrow printed column rules. A distinctive dark vertical crease must
  be visible; otherwise the page is flagged `needs_review` and not split.
* By default, each half is trimmed by **0.1% of the original spread width**
  at the binding; no pixels from the opposite page are intentionally duplicated.
* Source PDF page numbers and `page_sides` exclusions from the existing
  `schema/situation_financiere/page_ranges.json` are preserved.
* Portrait/single-page scans are skipped, never bisected.
* No attempt is made to straighten distorted pages or split each printed page
  into its two financial tables; that is a separate next step.

## Outputs

`derived/situation_financiere/split_pages/1882/` contains:

- `p0245_left_200dpi.png` and `p0245_right_200dpi.png` (and other PDF pages)
- `previews/p0245_cut.jpg`: magenta cutting line; dashed cyan permissible band
- `review_contact_sheet.jpg`: all selected cuts at a glance
- `split_review.csv`: cut percentage, darkness score, accepted/review status

For a page flagged `needs_review`, no half-page outputs are created unless the
explicit `--force-review` flag is provided. Such outputs are labeled provisional
and should not be used for paid OCR without verification.

**Limitations:** This is a single straight split, not book dewarping. It is
expected to work for a broad continuous dark gutter near the middle. It can fail
where the gutter is faint, discontinuous, occluded or far off-centre. Evaluate
its previews across different years before adopting it in the production OCR
pipeline. Saving two images does not itself validate that all text is readable.

## Inner gutter trim (selected checkpoint default: 0.001)

The default stage-1 output **no longer includes a shared overlap** across the
book binding. The previous `--overlap .012` doubled up portions of the
neighbouring printed page and could cause OCR to transcribe stray characters.
The final selected default `--inner-trim .001` removes **0.1% of the *full spread width*
from each side of the detected gutter**. The halves are approximately equal
and have a narrow excluded strip in between. It does not erase or rotate pixels;
it crops the original scans.

Run the spread splitter again before re-running stage 2:

```bash
.venv/bin/python code/split_spreads.py --year 1882 --pages 245 247 249
open derived/situation_financiere/split_pages/1882/review_contact_sheet.jpg
```

The shaded central strip in the preview is *removed* from both output images.
The CSV now records `inner_trim_fraction`, `left_end_px`, and `right_start_px`.
Check especially whether text is very close to the book binding. A trim helps
prevent unrelated text but **cannot recover text hidden by the crease**, and
an overly aggressive trim can clip legitimate data.

To experiment with a larger trim, use `--inner-trim .003`. To reproduce the old
intentional page overlap, use `--inner-trim 0 --overlap .012`. A positive overlap
and a positive trim cannot be combined.

Previously accepted PNGs are regenerated in place. If a scan is now flagged for
review, stale crops are removed; a configured absent printed side is likewise
removed so it cannot accidentally be passed to OCR.

For years marked `scan_layout: single`, the script saves an intact
`pXXXX_single_<dpi>dpi.png` and includes it in the contact sheet. A single-page
scan is never bisected. For configured spreads, `page_sides` controls which
printed halves are relevant. The year-wide review remains in `derived/`.
