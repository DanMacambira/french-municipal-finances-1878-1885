# OCR handoff — accepted Stage 2 panels only (v1.8)

Stage 1/2 own all PDF rendering and table splitting. `situation_financiere.py`
never renders, crops, or re-segments a PDF. Its only OCR provider is Mistral.
Source PDFs are opened locally solely to verify confirmed page ranges and
source filenames. No PDF is sent to the API.

Approved images come only from the Stage 2 per-year `table_split_review.csv`
and exist under `derived/situation_financiere/table_panels/<year>/accepted/`.
Review-only candidates are never OCR-ready.

## Zero-cost checks

```bash
.venv/bin/python code/situation_financiere.py plan --years 1882
.venv/bin/python code/situation_financiere.py ocr --years 1882 --pages 245 --dry-run
```

PDF page 245 of the validated 1882 range produces four approved images
(left/right, col1/col2). `--dry-run` never calls the API and does not require
`MISTRAL_API_KEY`.

## Small paid pilot

After checking the dry run, set `MISTRAL_API_KEY` and run:

```bash
.venv/bin/python code/situation_financiere.py ocr --years 1882 --pages 245
```

A paid OCR run over all confirmed scans requires `--yes-all`.
Rerunning on unchanged images/model uses the existing OCR JSON cache; use
`--force` only if intentionally replacing it. Cache entries contain the
original PDF scan number, side, panel, source PDF filename, PNG path and
SHA256, model, timestamp, and raw OCR response. The new build re-verifies
image hashes and manifest membership before parsing cached responses.

OCR cache location:

```
derived/situation_financiere/ocr/mistral/segmented/<year>/p<PAGE>_<side>_col<N>.json
```

Older cache files are ignored but not deleted.

## Parsing

```bash
.venv/bin/python code/situation_financiere.py build --years 1882 --no-match
```

This retains all existing financial mappings, commune parsing, canton
validation, and extraordinary-tax parsing. Extracted values remain OCR
candidates, not research-verified data. Tax attribution merits further review.

Pass `--dpi` (default 200) consistently to `plan`, `ocr`, and `build` if Stage 2
produced images at another resolution. These commands never rescale images.

## Tests

```bash
.venv/bin/python -m pytest -q tests/
git diff --check
```

Obsolete PDF cropping commands and their tests were retired in v1.8. Their
prior versions remain recoverable in Git history and installer backups.
