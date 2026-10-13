# SPDX-License-Identifier: AGPL-3.0-only
import bz2
from collections import Counter
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('audit', Path(__file__).with_name('audit-mixed-corpus.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CorpusTests(unittest.TestCase):
    def test_offline_selection_preserves_attribution_and_excludes_heldout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'corpus.bz2'
            rows = [f'{i}\tjpn\t語彙の異なるテスト文章ですProduct{chr(97+i//26)}{chr(97+i%26)}を使う。\twriter{i % 20}\tdate\tdate' for i in range(200)]
            rows += [rows[0], 'invalid', '201\tjpn\t私はEnglishを使う。\t\\N\tdate\tdate']
            path.write_bytes(bz2.compress('\n'.join(rows).encode()))
            excluded = {'語彙の異なるテスト文章ですProductaaを使う。'}
            selected, counts = audit.select(path, excluded)
            self.assertEqual(len(selected), 24)
            self.assertEqual(audit.select(path, excluded), (selected, counts))
            self.assertTrue(all(r['author'] and not r['eligible_for_training'] and not r['reading_verified'] for r in selected))
            self.assertTrue(all(r['text'] not in excluded for r in selected))
            self.assertLessEqual(max(Counter(r['author'] for r in selected).values()), 3)
            self.assertGreater(counts['legacy_heldout'], 0)
            self.assertEqual(counts['malformed'], 1)
            self.assertEqual(counts['unattributed_or_other_language'], 1)


if __name__ == '__main__':
    unittest.main()
