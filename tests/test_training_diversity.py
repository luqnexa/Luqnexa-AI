import importlib.util
from pathlib import Path
import unittest
import json
import subprocess
import sys
import tempfile
spec=importlib.util.spec_from_file_location('diversity',Path(__file__).resolve().parents[1]/'models/chatgpt-training/audit-diversity.py')
diversity=importlib.util.module_from_spec(spec);spec.loader.exec_module(diversity)

class DiversityTests(unittest.TestCase):
    def test_word_substitutions_are_not_new_frames(self):
        result=diversity.audit([{'good':text} for text in ['Pythonで書く','Rustで書く','Javaで書く','mailを送った']])
        self.assertEqual(result['surface_frames'],2)
        self.assertEqual(result['substitution_variants'],2)

    def test_case_and_variable_names_do_not_inflate_diversity(self):
        self.assertEqual(diversity.frame('request_idの値を読む'),diversity.frame('USER_IDの値を読む'))

    def test_similar_japanese_phrasing_is_flagged_for_review(self):
        result=diversity.audit([{'good':text} for text in
            ['Pythonの利用方法を説明してください','Rustの使用方法を説明してください','mailが届かなければ電話する']])
        self.assertEqual(len(result['near_frames_for_review']),1)

    def test_exposure_counts_each_word_once_per_example(self):
        result=diversity.audit([{'good':'Pythonとpythonを比較する'},{'good':'Pythonを起動する'}])
        self.assertEqual(result['english_word_exposure'],[{'word':'python','examples':2}])

    def test_strict_mode_rejects_reused_prior_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            old=Path(folder)/'old.jsonl';new=Path(folder)/'new.jsonl'
            old.write_text(json.dumps({'good':'Pythonで書く'})+'\n',encoding='utf-8')
            new.write_text(json.dumps({'good':'Rustで書く'})+'\n',encoding='utf-8')
            result=subprocess.run([sys.executable,'-X','utf8',spec.origin,str(new),'--against',str(old),
                '--require-distinct-frames'],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            report=json.loads(result.stdout)
            self.assertEqual(len(report['reused_previous_frames']),1)

if __name__=='__main__':unittest.main()
