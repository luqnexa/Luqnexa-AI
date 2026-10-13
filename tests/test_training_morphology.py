import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('morphology',Path(__file__).resolve().parents[1]/'models/chatgpt-training/build-morphology-journal.py')
morphology=importlib.util.module_from_spec(spec);spec.loader.exec_module(morphology)

class MorphologyTests(unittest.TestCase):
    def test_godan_sound_changes_and_negative(self):
        self.assertIn(('およいだ','泳いだ','verb'),morphology.forms('およぐ','泳ぐ','godan'))
        self.assertIn(('およがない','泳がない','verb'),morphology.forms('およぐ','泳ぐ','godan'))
        self.assertIn(('たった','立った','verb'),morphology.forms('たつ','立つ','godan'))
        self.assertIn(('しんだ','死んだ','verb'),morphology.forms('しぬ','死ぬ','godan'))

    def test_ichidan_and_iku_are_not_godan_ru(self):
        self.assertIn(('ねていた','寝ていた','verb'),morphology.forms('ねる','寝る','ichidan'))
        self.assertIn(('いってください','行ってください','verb'),morphology.forms('いく','行く','iku'))
        self.assertNotIn(('いいて','行いて','verb'),morphology.forms('いく','行く','iku'))

    def test_adjective_and_fail_closed_annotations(self):
        self.assertIn(('さむくなかった','寒くなかった','adjective'),morphology.forms('さむい','寒い','adjective'))
        with self.assertRaises(ValueError):morphology.forms('ねる','寝る','unknown')
        with self.assertRaises(ValueError):morphology.forms('ゆく','行く','iku')

    def test_conditionals_and_passive_use_the_inflection_class(self):
        godan=morphology.forms('およぐ','泳ぐ','godan')
        self.assertIn(('およげば','泳げば','verb'),godan)
        self.assertIn(('およがれていた','泳がれていた','verb'),godan)
        ichidan=morphology.forms('ねる','寝る','ichidan')
        self.assertIn(('ねれば','寝れば','verb'),ichidan)
        self.assertIn(('ねられていた','寝られていた','verb'),ichidan)
        self.assertNotIn(('ねれれば','寝れれば','verb'),ichidan)

if __name__=='__main__':unittest.main()
