# SPDX-License-Identifier: AGPL-3.0-only
import bz2
import importlib.util
from pathlib import Path
import unittest
import hashlib
import json

spec = importlib.util.spec_from_file_location('trainer', Path(__file__).with_name('train-context.py'))
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class TrainingTests(unittest.TestCase):
    def test_bundled_artifact_matches_report_and_attribution(self):
        model_dir = Path(__file__).resolve().parents[1] / 'models'
        report = json.loads((model_dir / 'context-training.json').read_text(encoding='utf-8'))
        data = (model_dir / 'context-public.bin').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), report['model_sha256'])
        self.assertEqual(len(data), 8 + 16 * report['retained_features'])
        authors = (model_dir / 'TATOEBA-AUTHORS.tsv').read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(authors), sum(report['partitions'].values()))
        self.assertEqual(len({line.split('\t')[0] for line in authors}), len(authors))
        synthetic = (model_dir / 'context-synthetic.txt').read_text(encoding='utf-8')
        self.assertEqual(hashlib.sha256(synthetic.encode()).hexdigest(), report['synthetic']['sha256'])
        self.assertEqual(len([line for line in synthetic.splitlines() if line and not line.startswith('#')]),
                         report['synthetic']['sentences'])

    def test_filter_deduplicate_and_split(self):
        data = bz2.compress(('1\tjpn\t食事に箸を使う。\tAuthor\tdate\tdate\n'
                            '2\tjpn\t食事に箸を使う。\tOther\tdate\tdate\n'
                            '3\tjpn\t川に橋を架ける。\t\\N\tdate\tdate\n'
                            '4\teng\tExample text\tAuthor\tdate\tdate\n').encode())
        records = list(trainer.rows(data))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][2], 'Author')
        self.assertEqual(trainer.split(records[0][1]), trainer.split('食事に箸を使う。'))

    def test_deterministic_counts_encoding_and_pruning(self):
        records = [('1', '食事に箸を使う。', 'Author')] * 3
        _, retained = trainer.train(records, cap=10)
        self.assertEqual(len(retained), 10)
        self.assertTrue(all(count == 3 for _, count in retained))
        self.assertEqual(trainer.encode(retained), trainer.encode(trainer.train(records, 10)[1]))
        self.assertEqual(len(trainer.encode(retained)), 8 + 16 * 10)


if __name__ == '__main__':
    unittest.main()
