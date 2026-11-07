"""One reusable historical-commune matcher, shared by finance and school pipelines.

The external ``communematch`` engine and the reference workbook stay outside Git.
Configure paths with COMMUNES_REF_PATH and, when necessary, COMMUNEMATCH_SOURCE.
The matcher is created only once per Python process.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent
LOCAL_REFERENCE = PROJECT / "raw_data" / "COMMUNES_ref_v1.xlsx"
SHARED_REFERENCE = Path.home() / "dev" / "shared_data" / "COMMUNES_ref_v1.xlsx"


def reference_path() -> Path:
    """Explicit path, then shared workbook, then legacy project-local workbook."""
    explicit = os.environ.get("COMMUNES_REF_PATH", "").strip()
    if explicit:
        source = Path(explicit).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"COMMUNES_REF_PATH does not exist: {source}")
        return source
    for source in (SHARED_REFERENCE, LOCAL_REFERENCE):
        if source.is_file():
            return source.resolve()
    raise FileNotFoundError(
        "Could not find COMMUNES_ref.xlsx. Put it at "
        f"{SHARED_REFERENCE}, retain it at {LOCAL_REFERENCE}, "
        "or set COMMUNES_REF_PATH=/absolute/path/COMMUNES_ref.xlsx."
    )


def _load_engine():
    """Use installed communematch or a deliberately selected local source tree."""
    hint = os.environ.get("COMMUNEMATCH_SOURCE", "").strip()
    if hint:
        source = Path(hint).expanduser().resolve()
        # Accept source tree, the src/ layout, or the package directory itself.
        choices = (source, source / "src", source.parent)
        root = next((p for p in choices if (p / "communematch").is_dir() or
                     (p / "communematch.py").is_file()), None)
        if root is None:
            raise FileNotFoundError(
                f"COMMUNEMATCH_SOURCE={source} does not contain communematch/ "
                "or communematch.py (directly or under src/)."
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        from communematch import CommuneMatcher, MatcherConfig, ReferenceSpec
    except ModuleNotFoundError as exc:
        if exc.name == "communematch":
            raise RuntimeError(
                "Missing your external communematch package. Either install it "
                "with '.venv/bin/python -m pip install -e /path/to/its/project' "
                "or set COMMUNEMATCH_SOURCE to the folder containing communematch/."
            ) from exc
        raise
    return CommuneMatcher, MatcherConfig, ReferenceSpec


@lru_cache(maxsize=1)
def get_matcher():
    """Construct once per process; default to the Pas-de-Calais reference subset."""
    reference = reference_path()
    CommuneMatcher, MatcherConfig, ReferenceSpec = _load_engine()
    department = os.environ.get("COMMUNEMATCH_DEPARTMENT", "62").strip() or "62"
    return CommuneMatcher.from_file(
        reference,
        config=MatcherConfig(department=department),
        spec=ReferenceSpec(
            id_column="noacass",
            name_columns=["nom_1999", "nom_1801", "nom_an3", "nom_bull_des_lois"],
            multi_value_columns=["nom_bull_des_lois"],
        ),
    )


def translate_communes(df, column="commune"):
    """Compatibility wrapper for other projects using this same matcher."""
    return get_matcher().translate_frame(df, column=column)
