# Commune matching: shared reference and reuse across projects

This integrates the **existing** `communematch` engine. It does not replace its algorithms and does not select a research sample. The `build` step:

- Preserves each `commune_raw` exactly as OCR extracted it.
- Applies your school-pipeline lookup cleaning to a **temporary** name: `strip`, remove enclosing straight apostrophes, and remove parenthetical suffixes.
- Calls `get_matcher().translate_frame(..., column="_commune_match_query")`.
- Retains `cassini_id` (from `cm_id`), `commune_cleaned` (from `cm_name`), and `cm_status` (the matching diagnostic); discards extra `cm_*` and temporary lookup columns.
- Never drops unmatched or duplicate candidates; the econometric exporter separates them for review. No `apply_sample()` is invoked.

## Where to put COMMUNES_ref.xlsx

For one master workbook shared across repositories, place it here:

    ~/dev/shared_data/COMMUNES_ref.xlsx

For example, with the source file currently in another project:

    mkdir -p ~/dev/shared_data
    cp /absolute/path/to/other/project/raw_data/COMMUNES_ref.xlsx ~/dev/shared_data/COMMUNES_ref.xlsx

You can instead leave your reference in a different location and specify its full path:

    export COMMUNES_REF_PATH="/absolute/path/to/COMMUNES_ref.xlsx"

The wrapper selects, in order: `COMMUNES_REF_PATH`, shared default, then legacy `raw_data/COMMUNES_ref.xlsx` in the finance repository. It does not move/delete your old workbook. Do not commit the workbook without checking permissions/licensing.

## Making the existing communematch engine importable

**`commune_matcher.py` is only a wrapper.** It imports the separate Python package `communematch`, which must be available to this repository's `.venv/bin/python`.

Preferred if you know the source repository of your package and it has a `pyproject.toml` or `setup.py`:

    cd ~/dev/french-municipal-finances-1878-1885
    .venv/bin/python -m pip install -e /absolute/path/to/communematch/repository

Alternatively, if the package source is not installable but contains `communematch/` or `src/communematch/`:

    export COMMUNEMATCH_SOURCE="/absolute/path/to/source/repository"

`COMMUNEMATCH_SOURCE` can also point to the package folder itself. Existing engine dependencies still need to be installed. To discover a local copy, try:

    find ~/dev -type d -name communematch -not -path '*/.venv/*'

Do not attempt to `pip install commune_matcher.py`: it is just your wrapper, not the package.

## Run (offline; no Mistral calls)

    cd ~/dev/french-municipal-finances-1878-1885
    .venv/bin/python code/situation_financiere.py build
    .venv/bin/python code/export_econometric_panel.py --repo .

The output is under `derived/situation_financiere/econometric_export/`. Check `year_quality_summary.csv` and `excluded_unmatched.csv` before interpreting the provisional panel.

For another department, when the rest of the extraction pipeline supports it, set e.g. `COMMUNEMATCH_DEPARTMENT=59` before building. The default remains `62` (Pas-de-Calais); the source OCR pipeline currently has separate 1878–1885 and Pas-de-Calais assumptions to generalize later.
