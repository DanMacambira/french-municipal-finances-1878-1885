# Reference-page table templates (local, no Docker)

## Start with one year (1882)

Run from the repository root:

```bash
.venv/bin/python code/table_template_tool.py --year 1882 --page 245
```

A browser tab will open at `http://127.0.0.1:<random-port>/`. Draw **one rectangle per COMPLETE financial table**. For a two-page spread, there are generally four tables; for a single printed page, two. Include headings and the last commune rows. Click **Save template and preview the year**.

This creates:

- `schema/situation_financiere/table_templates.json` — one reusable reference template per year, with normalized coordinates; safe to commit after checking.
- `derived/situation_financiere/reference_templates/1882/overlays/*.jpg` — transferred bounding boxes on the other pages in that year; diagnostic only.
- `derived/situation_financiere/reference_templates/1882/review.csv` — one row per page; the accuracy field is intentionally blank.

Close the browser tab and press Ctrl+C in Terminal when done.

For a different year, run the same command with its year and a representative one-based PDF page. For example, 1878 page 446:

```bash
.venv/bin/python code/table_template_tool.py --year 1878 --page 446
```

To regenerate the previews without reopening the drawing UI:

```bash
.venv/bin/python code/table_template_tool.py --year 1882 --preview-only
```

**Important:** This is intentionally a first-stage manual template/preview test. It transfers box positions as proportions of the full PDF scan, not with image registration. This is a hypothesis to test, **not** guaranteed alignment. Verify several transferred overlays before using boxes to crop or OCR other pages. The tool does not modify PDFs, OCR caches or the production parser, and it makes no API requests.

`page_ranges.json` must identify the PDF and confirmed page range; the PDF must already exist in `raw_data/situation_financiere_communes/`. Use the existing `pymupdf` and `Pillow` environment.
