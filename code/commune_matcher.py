# Commune matcher engine : to be run only once before derived.py
from functools import lru_cache
from pathlib import Path

from communematch import CommuneMatcher, MatcherConfig, ReferenceSpec

REF_PATH = Path(__file__).resolve().parent.parent / "raw_data" / "COMMUNES_ref.xlsx"

@lru_cache(maxsize=1)
def get_matcher() -> CommuneMatcher:
    """Builds the matcher once per process. Every later call reuses it."""
    return CommuneMatcher.from_file(
        REF_PATH,
        config=MatcherConfig(department="62"),
        spec=ReferenceSpec(
            id_column="noacass",
            name_columns=["nom_1999", "nom_1801", "nom_an3", "nom_bull_des_lois"],
            multi_value_columns=["nom_bull_des_lois"],
        ),
    )

def translate_communes(df, column="commune"):
    """Thin wrapper every function calls — never builds the index itself."""
    return get_matcher().translate_frame(df, column=column)