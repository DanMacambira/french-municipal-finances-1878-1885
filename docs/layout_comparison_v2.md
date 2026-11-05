# Layout model versus volume-template comparison (experimental)

The previous vertical-line heuristics are **not used** by this benchmark. The
script analyzes entire scanned PDF pages. It never guesses a gutter before
finding tables, never crops production images, and never calls Mistral.

## Methods and limitations

1. **Blue:** pretrained Microsoft Table Transformer (PubTables-1M) recognizes
   table bounding boxes on the full PDF-page image. The detector was not
   trained specifically for 19th-century French financial tables and may fail.
2. **Orange:** volume template built using **development pages only**. It
   robustly aggregates stable table rectangles detected by the *same*
   pretrained model, then transfers them after conservative visual registration.
   **It is not an independent OCR or detection model.** It tests whether
   pooling repeated volume geometry helps or hurts generalization.
3. **Purple:** potential gutter between detected tables, only if a wide scan
   visibly contains distinct tables on either side. If the model cannot find
   suitable boxes, the gutter is explicitly `unresolved`.

*Neither blue/orange agreement nor detector confidence establishes accuracy.*
The script deliberately does **not** analyze column/brace geometry yet. First
verify that whole-table regions and multi-page structure are correct.

## Installation

From the repository root (same virtualenv as the existing project):

```bash
.venv/bin/python -m pip install -r requirements-layout-model.txt
```

Model weights are downloaded from Hugging Face on first execution. This is **a
model download**, not a Mistral OCR call. The first run requires internet
access, and the model has licensing and dependency requirements. **On Intel
macOS (your earlier OpenCV installer output showed an x86_64 wheel), current
PyTorch may not provide a suitable Python 3.14 wheel. If that happens, do
not change the research project environment: use a separate Linux Python
environment for this optional benchmark.** See
https://huggingface.co/microsoft/table-transformer-detection (MIT license).

### Docker option for Intel macOS (no changes to your existing `.venv`)

The host's earlier OpenCV wheel was `macosx_14_0_x86_64`. Native PyTorch
support on that architecture may be unavailable. If Docker is installed, use
this **alternative** rather than installing PyTorch in `.venv`:

```bash
# The Dockerfile consumes no build context: never copy the source PDFs into an image.
docker build -t municipal-layout-detector - < Dockerfile.layout-model

# Plan-only: verify your frozen sample is available.
docker run --rm -v "$PWD":/work municipal-layout-detector \
  python code/layout_compare.py --plan-only

# Small smoke run. Cached model weights are stored outside your Git repository.
docker run --rm -v "$PWD":/work \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  municipal-layout-detector python code/layout_compare.py --max-pages 6 \
  --output derived/situation_financiere/layout_comparison_smoke
```

This Linux CPU option avoids depending on native Mac PyTorch availability.
It does require Docker and sufficient disk space for model packages and
weights. All PDF processing happens locally inside the mounted repository;
only public model weights are downloaded. Do not commit model caches.

## Run

Do the initial smoke test with a **separate output directory**, so it does
not create a frozen two-year review shortlist in the full benchmark folder.

```bash
# Use the existing frozen 40-PDF-page sample_plan.json (default).
.venv/bin/python code/layout_compare.py --plan-only

# Initial smoke run, deliberately not a valid eight-year evaluation:
.venv/bin/python code/layout_compare.py --max-pages 6 --output derived/situation_financiere/layout_comparison_smoke

# Full benchmark; remove --max-pages only after smoke test succeeds:
.venv/bin/python code/layout_compare.py
```

The comparison uses the existing sample from
`derived/situation_financiere/layout_benchmark/sample_plan.json` if present.
If not, it generates a deterministic plan using the existing
`schema/situation_financiere/page_ranges.json`. Sample roles are frozen.
`--max-pages` limits smoke testing only; because there may be fewer than two
development samples per year, the orange method may be unavailable.

Outputs under `derived/situation_financiere/layout_comparison_v2/`:

- `comparison_manifest.csv`: model counts, transferred counts, agreement,
  unresolved cases, and proposed gutter positions.
- `templates.json`: training pages and status for each year/layout family.
- `detections/`: exact table boxes from both methods.
- `model_cache/`: reusable model predictions tied to image hashes.
- `overlays/`: full-page comparisons.
- `review_contact_sheet.jpg`: **one page per year**, not 40 pages.
- `review_shortlist.csv`: enter `yes/no/na` only for whether *all* table
  regions are enclosed without clipping, for blue and orange independently.
- `comparison_summary.json`: machine counts explicitly marked **not accuracy**.

To score the human shortlist after reviewing the contact sheet and detailed
overlays:

```bash
.venv/bin/python code/layout_compare.py --score-only
```

For a meaningful evaluation, keep the holdout pages untouched while tuning.
A layout-template fit from model detections is **not** an independent baseline.
If both approaches fail, we should consider a second *independent* pretrained
layout model, or a small curated layout annotation benchmark, rather than
return to page-specific handcrafted coordinates.

## Compatibility and data safety

The benchmark is additive; no production modules or OCR outputs are modified.
The `raw_data` and `derived` directories are unchanged except for new
benchmark outputs in `derived`. Model confidence is not a substitute for
manually inspecting some pages. Original scan PDFs and API credentials must
never be committed to GitHub.
