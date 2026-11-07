# Concurrent Mistral OCR update

This adds `--workers N` to the existing `code/situation_financiere.py ocr` command.

## Install on macOS

```bash
python3 ~/Downloads/parallel_ocr_update.pyz --cleanup
```

The installer backs up the existing OCR entrypoint before patching it. It leaves
all approved image panels, segmentation manifests, PDFs, JSON caches, and parser
results untouched. It runs offline tests using the project's `.venv`.

## Usage

```bash
cd ~/dev/french-municipal-finances-1878-1885

# No network calls: check how many accepted panels are already cached.
.venv/bin/python code/situation_financiere.py ocr --yes-all --workers 4 --dry-run

# Process *only missing or changed* accepted panels, four at a time.
.venv/bin/python code/situation_financiere.py ocr --yes-all --workers 4

# Restrict request starts if Mistral rate limits your account.
.venv/bin/python code/situation_financiere.py ocr --yes-all --workers 4 --rpm 30

# The original sequential method still works unchanged.
.venv/bin/python code/situation_financiere.py ocr --yes-all
```

`MISTRAL_API_KEY` must be set for paid OCR if there are uncached images.
The key is not saved in source files. Do **not** use `--force` unless you
intentionally want to re-run OCR and incur new charges.

## Design

- Uses `ThreadPoolExecutor` (OCR requests are network-bound, not CPU-bound).
- Default remains sequential unless explicitly given `--workers` > 1.
- Worker count is capped at 32; start with four and increase only if the
  account's Mistral rate limit permits.
- Existing image-SHA256 / source-PDF / panel / model cache rules are reused.
- The existing `api_ocr()` is reused for model parameters, HTTP errors, and retries.
- Completed JSON files are written atomically and can be reused immediately.
- One failed request does not cancel other requests. The exit code is nonzero
  if any requests fail; rerunning skips successes and retries only failures.
- Optional `--rpm` paces the start of new panel OCR jobs. The core client's
  internal HTTP retries are not strictly covered by this limit; Mistral's
  published/account-specific quotas always take precedence.
- No department/year/page-specific corrections are used. Future 1878–1900
  expansion still requires extending the underlying project's year / page
  configuration and accepted-image segmentation.

**Note:** If all 1878–1885 panels were already OCR'd, the dry run will report
zero new requests. Parallelization benefits the next department or any future
newly approved images, not the already completed OCR run.
