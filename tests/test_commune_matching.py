"""Offline tests: no external communematch package or reference file required."""
from __future__ import annotations

import ast
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "code" / "situation_financiere.py"
WRAPPER = ROOT / "code" / "commune_matcher.py"


def load_matching_function():
    # Extract a single function to avoid Mistral/PDF dependency during offline tests.
    text = MAIN.read_text(encoding="utf-8")
    tree = ast.parse(text)
    matches = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "apply_commune_match"]
    if len(matches) != 1:
        raise AssertionError("Expected precisely one apply_commune_match")
    namespace = {"pd": pd}
    exec(compile(ast.Module(body=matches, type_ignores=[]), str(MAIN), "exec"), namespace)
    return namespace["apply_commune_match"]


class FakeMatcher:
    def translate_frame(self, df, column):
        assert column == "_commune_match_query"
        assert list(df[column]) == ["Saint-Omer", "Calais", "Missing"]
        out = df.copy()
        out["cm_id"] = ["123", "456.0", None]
        out["cm_name"] = ["Saint-Omer", "Calais", pd.NA]
        out["cm_status"] = ["exact", "matched_fuzzy", "unmatched"]
        out["cm_score"] = [1.0, 0.96, 0.0]
        return out


class TestBuildMatch(unittest.TestCase):
    def test_preserve_all_rows_and_original_names(self):
        frame = pd.DataFrame({
            "row_id": ["r1", "r2", "r3"],
            "year": [1880, 1880, 1880],
            "commune_raw": [" 'Saint-Omer (suite)' ", "'Calais' ", "Missing"],
            "population": [20, 100, 30],
        })
        cm = types.ModuleType("commune_matcher")
        cm.get_matcher = lambda: FakeMatcher()
        with patch.dict(sys.modules, {"commune_matcher": cm}):
            result = load_matching_function()(frame)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["row_id"].tolist(), ["r1", "r2", "r3"])
        self.assertEqual(result["commune_raw"].tolist(), frame["commune_raw"].tolist())
        self.assertEqual(result["commune_cleaned"].iloc[:2].tolist(), ["Saint-Omer", "Calais"])
        self.assertEqual(result["cassini_id"].iloc[:2].tolist(), [123, 456])
        self.assertTrue(pd.isna(result["cassini_id"].iloc[2]))
        self.assertEqual(result["cm_status"].iloc[1], "matched_fuzzy")
        self.assertNotIn("cm_score", result.columns)
        self.assertNotIn("_commune_match_query", result.columns)

    def test_row_reordering_is_not_silent(self):
        class Reordered(FakeMatcher):
            def translate_frame(self, df, column):
                return super().translate_frame(df, column).iloc[::-1].reset_index(drop=True)
        cm = types.ModuleType("commune_matcher")
        cm.get_matcher = lambda: Reordered()
        frame = pd.DataFrame({"row_id": ["1", "2", "3"], "commune_raw": ["Saint-Omer", "Calais", "Missing"]})
        with patch.dict(sys.modules, {"commune_matcher": cm}):
            with self.assertRaisesRegex(RuntimeError, "row order"):
                load_matching_function()(frame)


class TestSharedReference(unittest.TestCase):
    def test_explicit_reference_override(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("finance_commune_wrapper_test", WRAPPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "COMMUNES_ref.xlsx"
            f.write_bytes(b"stub")
            with patch.dict(os.environ, {"COMMUNES_REF_PATH": str(f)}):
                self.assertEqual(module.reference_path(), f.resolve())

    def test_explicit_missing_workbook_is_error(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("finance_commune_wrapper_test2", WRAPPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"COMMUNES_REF_PATH": str(Path(tmp)/"missing.xlsx")}):
                with self.assertRaises(FileNotFoundError):
                    module.reference_path()

if __name__ == "__main__":
    unittest.main()
