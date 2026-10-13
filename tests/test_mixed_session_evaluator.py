import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('session_evaluator',Path(__file__).resolve().parents[1]/'models/chatgpt-training/evaluate-mixed.py')
evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)

class SessionEvaluatorTests(unittest.TestCase):
    def test_uses_final_candidates_not_context_or_intermediate_answers(self):
        output='preedit: 文脈\n> 1. 文脈\nlearning: off features=0\npreedit: 最終候補\n> 1. 最終候補\n  2. 別候補\nlearning: off features=0\n'
        self.assertEqual(evaluator.session_candidates(output),['最終候補','別候補'])

    def test_fail_closed_if_learning_was_enabled(self):
        with self.assertRaises(ValueError):evaluator.session_candidates('preedit: 候補\n> 1. 候補\nlearning: on features=12\n')

if __name__=='__main__':unittest.main()
