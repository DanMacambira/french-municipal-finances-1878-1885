#!/usr/bin/env python3
"""Add one reviewed shared nine-column layout for selected years, without rewriting other mappings."""
from __future__ import annotations
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Printed headings in selected Pas-de-Calais 1878-1885 sample share this layout.
# Never assume that every year/department has the same columns.
NINE_COLUMNS = {
    "2": {"name": "population", "type": "integer"},
    "3": {"name": "superficie_hectares", "type": "integer"},
    "4": {"name": "revenus_annuels_francs", "type": "integer"},
    "5": {"name": "valeur_centime_francs", "type": "decimal_fr"},
    "6": {"name": "centimes_total", "type": "integer"},
    "7": {"name": "centimes_extraordinaires_raw", "type": "text"},
    "8": {"name": "duree_impositions_extraordinaires_raw", "type": "text"},
    "9": {"name": "revenus_bureau_bienfaisance_raw", "type": "text"},
}


def extend_mappings(config: dict, years: list[int]) -> tuple[dict, list[int], list[int]]:
    """Return additive config, added years, and years with an existing 9-col mapping.

    Existing entries are never overwritten, including 1882's reviewed 8-col exception.
    """
    config = json.loads(json.dumps(config))
    added, retained = [], []
    for year in years:
        bucket = config.setdefault(str(year), {})
        layouts = bucket.setdefault("layouts", [])
        existing = [layout for layout in layouts if layout.get("column_count") == 9]
        if existing:
            retained.append(year)
        else:
            layouts.append({
                "name": "pas_de_calais_reviewed_nine_columns",
                "column_count": 9,
                "columns": NINE_COLUMNS,
            })
            added.append(year)
    return config, added, retained


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--years", type=int, nargs="+", default=list(range(1878, 1886)))
    p.add_argument("--apply", action="store_true", help="Explicitly write config (default: dry-run)")
    args = p.parse_args()
    path = args.repo.expanduser().resolve() / "schema/situation_financiere/column_mappings.json"
    if not path.is_file():
        p.error(f"Cannot find mapping config: {path}")
    current = json.loads(path.read_text(encoding="utf-8"))
    updated, added, retained = extend_mappings(current, args.years)
    print("Adding shared 9-column layout to years:", added or "none")
    print("Already have a 9-column layout (left unchanged):", retained or "none")
    print("No 7- or 8-column corrections are inferred or applied.")
    if not args.apply:
        print("DRY RUN: add --apply when ready.")
        return
    if current == updated:
        print("Config unchanged")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"column_mappings.before_shared_layout.{stamp}.json")
    shutil.copy2(path, backup)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    print("Updated", path)
    print("Backup", backup)

if __name__ == "__main__":
    main()
