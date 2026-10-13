import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('diagnose',Path(__file__).resolve().parents[1]/'models/chatgpt-training/diagnose-runtime.py')
diagnose=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnose)

class RuntimeDiagnosisTests(unittest.TestCase):
    def test_score_components_are_complete_and_bounded(self):
        result=diagnose.score_components('score: 1 -500 2000\nscore: 2 -600 1000',['甲','乙'])
        self.assertEqual(result[0],dict(surface='甲',total=-500,native=2000,other=-2500))
        for text in ['','score: 1 0 5000','score: 1 0 0\nscore: 1 0 0','score: 3 0 0']:
            with self.assertRaises(ValueError):diagnose.score_components(text,['甲'])

    def test_missing_reference_is_not_ranker_error(self):
        self.assertEqual(diagnose.classify('正解',['誤答']),'missing_candidate')
        self.assertEqual(diagnose.classify('正解',[]),'missing_candidate')

    def test_present_reference_distinguishes_ranking(self):
        self.assertEqual(diagnose.classify('正解',['誤答','正解']),'ranking_error')
        self.assertEqual(diagnose.classify('正解',['正解','誤答']),'correct')

    def test_session_commands_cannot_be_injected_from_input(self):
        row=dict(diagnose.training.ranker.read_records(diagnose.HERE/'round-20261013-18.jsonl')[0])
        for input_text in ['text\ntraining','',None,'x'*513]:
            with self.assertRaises(ValueError):diagnose.validate([dict(row,input=input_text)])

    def test_fixed_evaluation_cannot_become_training_negatives(self):
        row=dict(diagnose.training.ranker.read_records(diagnose.HERE/'mixed-heldout-v1.jsonl')[0])
        with self.assertRaises(ValueError):diagnose.validate([row])
        row['split']='train'
        with self.assertRaises(ValueError):diagnose.validate([row])
        row=dict(diagnose.training.ranker.read_records(diagnose.HERE/'mixed-validation-v1.jsonl')[0],split='train')
        with self.assertRaises(ValueError):diagnose.validate([row])

if __name__=='__main__':unittest.main()
