import importlib.util
import unittest
from collections import Counter
from pathlib import Path

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('mixed_builder',root/'models/chatgpt-training/build-mixed-1000.py')
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

class MixedThousandTests(unittest.TestCase):
    def test_count_balance_and_reproducible_unique_rows(self):
        rows=builder.build([])
        self.assertEqual(rows,builder.build([]))
        self.assertEqual(Counter(r['topic'] for r in rows),{k:250 for k in ['proper_names','identifiers','english_literals','acronyms']})
        self.assertEqual(len({r['input'] for r in rows}),1000)
        self.assertEqual(len({r['good'] for r in rows}),1000)
        self.assertTrue(all(r['split']=='train' and r['language_pair']=='ja-en' and r['good']!=r['bad'] for r in rows))

    def test_existing_and_frozen_examples_are_excluded(self):
        old=builder.build([])[:2]
        known=[dict(r,split='test') for r in old]
        new=builder.build(known)
        self.assertEqual(len(new),1000)
        self.assertFalse({r['input'] for r in known}&{r['input'] for r in new})
        self.assertFalse({r['good'] for r in known}&{r['good'] for r in new})

if __name__=='__main__':unittest.main()
