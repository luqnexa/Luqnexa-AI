import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('assistant_training',Path(__file__).resolve().parents[1]/'models/chatgpt-training/run-training.py')
training=importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)

class TrainingFocusTests(unittest.TestCase):
    def test_scope_uses_surface_not_romaji_or_hint(self):
        rows=[{'input':'honwoyomu','context':'Pythonの話','good':'本を読む','split':'train'},
              {'good':'Pythonで書く','split':'train'}, {'good':'hello','split':'train'},
              {'good':'日本語と한국어','split':'train'}]
        self.assertEqual(training.select_training(rows,'mixed-ja-en'),[rows[1]])

    def test_other_language_lessons_are_not_english(self):
        for text in ['bonjourと挨拶する','caféに行く','holaと書く','Pythonと한국어を使う']:
            self.assertFalse(training.is_mixed_japanese_english({'good':text}))
        self.assertFalse(training.is_mixed_japanese_english({'good':'helloと書く','language_pair':'ja-fr'}))

    def test_fixed_partitions_remain_identical(self):
        rows=[{'good':'雨が降る','split':s} for s in ['train','validation','test','regression']]
        focused=training.select_training(rows,'mixed-ja-en')
        self.assertEqual(focused,rows[1:])
        self.assertTrue(all(a is b for a,b in zip(focused,rows[1:])))
        self.assertEqual(training.select_training(rows,'all'),rows)

    def test_previous_weights_are_copied_and_reproducible(self):
        rows=[{'good':'Pythonで書く','bad':'pythonで書く','context':'言語の正式名','split':s} for s in ('train','validation')]
        initial=[0.01]*training.ranker.DIM
        before=initial[:]
        first=training.ranker.fit(rows,epochs=2,initial_weights=initial)
        self.assertEqual(first,training.ranker.fit(rows,epochs=2,initial_weights=initial))
        self.assertEqual(initial,before)
        with self.assertRaises(ValueError):training.ranker.fit(rows,initial_weights=[float('nan')]*training.ranker.DIM)
        self.assertEqual(training.ranker.fit(rows,epochs=2),training.ranker.fit(rows,epochs=2,initial_weights=[0.0]*training.ranker.DIM))

if __name__=='__main__':unittest.main()
