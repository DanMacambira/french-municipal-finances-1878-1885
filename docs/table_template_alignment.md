# Conservative table-template registration (experimental)

Keep the manually drawn reference boxes in `schema/situation_financiere/table_templates.json`.
The tool compares the printed column-header bands in each later page with the reference.
It proposes a translation **per table**, not one rigid transform for the spread.
It does not modify PDFs, source OCR JSONs, or the validated dataset.

From the repository root, with the existing Python environment and `opencv-python-headless`:

```shell
.venv/bin/python code/table_template_tool.py --year 1882 --preview-only
```

Outputs:

- `derived/situation_financiere/reference_templates/1882/overlays/`
- `derived/situation_financiere/reference_templates/1882/review.csv`

`status=provisional_aligned` means an apparently unique nearby image match,
not a verified crop. `uncertain_*` keeps the original rectangle and requires
review; the script will **never** use uncertain predictions for automated OCR.
Use `--no-align` to compare with the old normalized-position overlays.

Repeated headers can match the wrong table; source text near the fold may be
physically hidden. This is a diagnostic prototype, not production-ready OCR.
