import unittest
from pathlib import Path
import importlib.util
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

exporter = load_module("econometric_export", BASE / "code/export_econometric_panel.py")
layout = load_module("shared_layout", BASE / "tools/install_shared_layout.py")

class TestLayout(unittest.TestCase):
    def test_additive_only_no_pages(self):
        config = {"1882": {"layouts": [{"name": "existing", "column_count": 9, "columns": {"2": {"name": "custom", "type": "integer"}}}, {"name": "8-col", "column_count": 8, "columns": {}}]}}
        new, added, retained = layout.extend_mappings(config, [1878,1882,1900])
        self.assertEqual(added, [1878,1900])
        self.assertEqual(retained, [1882])
        self.assertEqual(len(new["1882"]["layouts"]), 2)
        self.assertEqual(new["1882"]["layouts"][0]["columns"]["2"]["name"], "custom")
        self.assertEqual(new["1900"]["layouts"][0]["columns"]["9"]["name"], "revenus_bureau_bienfaisance_raw")

class TestNumbers(unittest.TestCase):
    def test_readable_numeric_values(self):
        self.assertEqual(exporter.parse_number("12,573", "int"), (12573, "parsed"))
        self.assertEqual(exporter.parse_number("1 234", "int"), (1234, "parsed"))
        self.assertEqual(exporter.parse_number("2.345", "int"), (2345, "parsed"))
        self.assertEqual(exporter.parse_number("32,66", "decimal"), (32.66, "parsed"))
        self.assertEqual(exporter.parse_number("1.234,50", "decimal"), (1234.5, "parsed"))
        self.assertEqual(exporter.parse_number("3,14", "int")[1], "ambiguous")
        self.assertEqual(exporter.parse_number("2,777,08", "decimal")[1], "ambiguous")
        self.assertTrue(pd.isna(exporter.parse_number('"', "int")[0]))
        self.assertEqual(exporter.parse_number(0, "int"), (0, "parsed"))

class TestPanel(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame([
            {"row_id": "a", "year": 1880, "cassini_id": "100", "commune_raw": "A", "population": 100, "superficie_hectares": 200, "revenus_annuels_francs": 300, "valeur_centime_francs": 40.5, "centimes_total": 50, "revenus_bureau_bienfaisance_raw": "1,234", "review_flags": "tax:incomplete", "extraordinary_tax_pair_status": "incomplete"},
            {"row_id": "b", "year": 1880, "cassini_id": "200", "commune_raw": "B", "population": 1, "review_flags": ""},
            {"row_id": "c", "year": 1880, "cassini_id": "200", "commune_raw": "B-other", "population": 2, "review_flags": ""},
            {"row_id": "d", "year": 1880, "cassini_id": None, "commune_raw": "C", "population": 10, "review_flags": "unmatched_commune"},
            {"row_id": "e", "year": 1881, "cassini_id": "100", "commune_raw": "A", "population": 110, "superficie_hectares": 210, "revenus_annuels_francs": 310, "valeur_centime_francs": "41,55", "centimes_total": 51, "review_flags": "low_page_ocr_confidence"},
            {"row_id": "f", "year": 1881, "cassini_id": "201", "commune_raw": "F", "population": 55, "superficie_hectares": 50, "revenus_annuels_francs": 20, "valeur_centime_francs": "3,02", "centimes_total": 3, "review_flags": "unmapped_columns"},
        ])
    def test_unique_and_strict_differ(self):
        r = exporter.generate(self.rows)
        self.assertEqual(set(r["commune_year_panel_provisional"]["row_id"]), {"a", "e", "f"})
        self.assertEqual(set(r["commune_year_panel_strict"]["row_id"]), {"a"})
        self.assertEqual(set(r["excluded_duplicate_ids"]["row_id"]), {"b", "c"})
        self.assertEqual(set(r["excluded_unmatched"]["row_id"]), {"d"})
        a = r["commune_year_panel_provisional"].query("row_id == 'a'").iloc[0]
        self.assertEqual(a["revenus_bureau_bienfaisance_francs"], 1234)
        self.assertEqual(a["core_review_flags"], "")
        self.assertFalse(a["needs_core_review"])
        self.assertTrue(all(r["extraordinary_taxes_unverified"]["tax_ocr_unverified"]))
    def test_no_matched_does_not_fake_id(self):
        sample = self.rows.loc[self.rows["row_id"].eq("d")]
        result = exporter.generate(sample)
        self.assertTrue(result["commune_year_panel_provisional"].empty)
        self.assertEqual(len(result["excluded_unmatched"]), 1)
    def test_reject_duplicate_row_id(self):
        x = self.rows.copy()
        x.loc[1, "row_id"] = "a"
        with self.assertRaises(ValueError):
            exporter.generate(x)

if __name__ == "__main__":
    unittest.main()
