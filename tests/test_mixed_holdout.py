import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('training_holdout',ROOT/'models/chatgpt-training/run-training.py')
training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)

class HoldoutTests(unittest.TestCase):
    def test_input_leak_is_rejected(self):
        row=training.ranker.read_records(training.HERE/'mixed-heldout-v1.jsonl')[0]
        with self.assertRaises(ValueError):training.check_mixed_holdout([dict(row,split='train',good='異なる文')])

    def test_reference_leak_is_rejected(self):
        row=training.ranker.read_records(training.HERE/'mixed-heldout-v1.jsonl')[0]
        with self.assertRaises(ValueError):training.check_mixed_holdout([dict(row,split='train',input='different')])

    def test_test_rows_do_not_train(self):
        rows=training.ranker.read_records(training.HERE/'mixed-heldout-v1.jsonl')
        training.check_mixed_holdout(rows)

    def test_selection_reference_is_not_a_training_partition(self):
        rows,digest=training.mixed_selection([])
        self.assertEqual(len(rows),12)
        self.assertEqual(len(digest),64)
        self.assertTrue(all(r['split']=='validation' for r in rows))

    def test_selection_leak_is_rejected(self):
        rows,_=training.mixed_selection([])
        with self.assertRaises(ValueError):training.mixed_selection([dict(rows[0],split='train')])

if __name__=='__main__':unittest.main()
