# SPDX-License-Identifier: AGPL-3.0-only
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("import_mozc", Path(__file__).with_name("import-mozc.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ImportTest(unittest.TestCase):
    def test_cost_and_context_ids_are_retained(self):
        self.assertEqual(module.convert("にほんご\t1\t2\t4000\t日本語"), "にほんご\t日本語\tmozc:1:2:4000\t3000\n")

    def test_optional_metadata_and_invalid_rows(self):
        self.assertIn(":SPELLING_CORRECTION", module.convert("にほんご\t1\t2\t4000\t日本語\tSPELLING_CORRECTION"))
        self.assertIsNone(module.convert("あ" * 129 + "\t1\t2\t4000\t長い"))
        self.assertIsNone(module.convert("にほんご\t1\t2\t4000\t日本\0語"))
        with self.assertRaises(ValueError):
            module.convert("incomplete")


if __name__ == "__main__":
    unittest.main()
