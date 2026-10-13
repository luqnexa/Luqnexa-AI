import importlib.util
from pathlib import Path
import unittest
import math

spec=importlib.util.spec_from_file_location('checkpoint_ranker',Path(__file__).resolve().parents[1]/'scripts/train-contrast.py')
ranker=importlib.util.module_from_spec(spec);spec.loader.exec_module(ranker)

class CheckpointTests(unittest.TestCase):
    def test_warm_start_survives_opposing_updates(self):
        good,bad='正解','誤答'
        a,b=ranker.features('',good),ranker.features('',bad)
        initial=[a.get(i,0)-b.get(i,0) for i in range(ranker.DIM)]
        rows=[dict(context='',good=bad,bad=good,split='train'),dict(context='',good=good,bad=bad,split='validation')]
        result,epoch=ranker.fit(rows,epochs=4,initial_weights=initial)
        self.assertEqual(epoch,0)
        self.assertEqual(result,initial)
        self.assertIsNot(result,initial)

    def test_prior_penalty_limits_drift_while_learning_new_signal(self):
        rows=[dict(context='新しい話題',good='正しい回答',bad='誤った回答',split=split) for split in ('train','validation')]
        initial=[0.0]*ranker.DIM
        free,_=ranker.fit(rows,epochs=30,initial_weights=initial)
        retained,_=ranker.fit(rows,epochs=30,initial_weights=initial,anchor_strength=0.1)
        norm=lambda w:math.sqrt(sum(x*x for x in w))
        self.assertLess(norm(retained),norm(free))
        self.assertGreater(ranker.score(retained,rows[0]['context'],rows[0]['good'])-ranker.score(retained,rows[0]['context'],rows[0]['bad']),0)
        self.assertEqual(initial,[0.0]*ranker.DIM)

    def test_invalid_prior_penalty_is_rejected(self):
        for value in [-1,3,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):ranker.fit([],anchor_strength=value)

    def test_adagrad_learns_and_is_reproducible(self):
        rows=[dict(context='話題',good='正解',bad='誤答',split=split) for split in ('train','validation')]
        first=ranker.fit(rows,epochs=10,optimizer='adagrad')
        self.assertEqual(first,ranker.fit(rows,epochs=10,optimizer='adagrad'))
        self.assertEqual(ranker.evaluate(first[0],rows)['wins'],2)
        with self.assertRaises(ValueError):ranker.fit(rows,optimizer='unsupported')

    def test_runtime_gap_increases_pressure_without_changing_validation(self):
        rows=[dict(context='話題',good='正解',bad='誤答',split=split) for split in ('train','validation')]
        base,_=ranker.fit(rows,epochs=1)
        rows[0]['score_margin']=0.5
        targeted,_=ranker.fit(rows,epochs=1)
        delta=lambda w:ranker.score(w,'話題','正解')-ranker.score(w,'話題','誤答')
        self.assertGreater(delta(targeted),delta(base))
        self.assertNotIn('score_margin',rows[1])

if __name__=='__main__':unittest.main()
