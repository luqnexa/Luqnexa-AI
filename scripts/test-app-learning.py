# SPDX-License-Identifier: AGPL-3.0-only
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('worker', Path(__file__).with_name('learn-while-app.py'))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class LearningTests(unittest.TestCase):
    def test_closed_application_never_starts_training(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            row = dict(context='',good='橋',bad='箸',group='seed',split='train')
            (cache/'seed.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
            with patch.object(worker,'CACHE',cache), patch.object(worker,'app_running',return_value=False), \
                    patch.object(worker,'cycle') as cycle, patch.object(worker.sys,'argv',['worker','--once']):
                worker.main()
                cycle.assert_not_called()
            self.assertEqual(json.loads((cache/'status.json').read_text())['state'],'waiting_for_codex')

    def test_stop_signal_prevents_next_teacher_request(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory)/'stop'
            stop.touch()
            with patch.object(worker.trainer,'STOP_FILE',stop):
                with self.assertRaises(KeyboardInterrupt):
                    worker.trainer.request(None,'','','/health')

    def test_generated_pair_requires_one_reviewed_substitution(self):
        row = dict(context='川を渡る。', good='橋を通る', bad='箸を通る')
        self.assertTrue(worker.trainer.valid_generated(row))
        for invalid in (dict(row, bad='箸を通らない'), dict(row, bad='橋を通る'),
                        dict(row, good='橋\nを通る'), dict(row, context=''),
                        dict(row, good='橋'*97), {'context': 'test'}):
            self.assertFalse(worker.trainer.valid_generated(invalid))

    def test_rolling_training_preserves_all_fixed_evaluation_rows(self):
        seed = [dict(group='fixed-test', split='test'), dict(group='seed', split='train')]
        added = [dict(group=str(i), split='train') for i in range(200)]
        result = worker.retain(seed, seed+added+[dict(group='untrusted-test', split='test')])
        self.assertEqual(result[:2], seed)
        self.assertEqual(len(result), 98)
        self.assertEqual(result[2]['group'], '104')
        self.assertNotIn('untrusted-test', [r['group'] for r in result])

    def test_worker_lock_prevents_duplicate_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'worker.lock'
            with worker.single_worker(path):
                with self.assertRaises(OSError):
                    with worker.single_worker(path):
                        pass
            with worker.single_worker(path):
                pass

    def test_generation_filters_invalid_duplicates_and_keeps_test_fixed(self):
        row = dict(context='川を渡る。', good='橋を通る', bad='箸を通る')
        response = dict(choices=[dict(message=dict(content=json.dumps(dict(pairs=[row,row]))))])
        with patch.object(worker.trainer, 'request', return_value=response):
            result = worker.trainer.generate(None, '', '', [], 4)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['split'], 'train')
            self.assertEqual(worker.trainer.generate(None, '', '', result, 4), [])


if __name__ == '__main__':
    unittest.main()
