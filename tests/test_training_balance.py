import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('training',Path(__file__).resolve().parents[1]/'models/chatgpt-training/run-training.py')
training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)

class BalanceTests(unittest.TestCase):
    def row(self,skill,good):return dict(skill=skill,good=good,bad=good+'誤',split='train')
    def test_case_flood_cannot_dominate(self):
        rows=[self.row('case',f'Word{i}を使う') for i in range(900)]
        rows += [self.row(s,'Englishを使う') for s in training.SKILLS if s!='case']
        report=training.assign_balanced_weights(rows)
        for value in report['skill_weight_totals'].values():self.assertAlmostEqual(value,1/6)
        self.assertEqual(report['missing_skills'],[])

    def test_repeated_frames_share_budget_and_evaluation_is_untouched(self):
        rows=[self.row('case','Pythonを使う') for _ in range(100)] + [self.row('case','Rustで書く')]
        fixed=dict(good='固定評価',bad='別候補',split='test')
        rows.append(fixed)
        report=training.assign_balanced_weights(rows)
        self.assertAlmostEqual(sum(r['training_weight'] for r in rows[:100]),rows[100]['training_weight'])
        self.assertNotIn('training_weight',fixed)
        self.assertEqual(report['missing_skills'],list(training.SKILLS[1:]))

    def test_inference_groups_case_errors_together(self):
        for good,bad in [('Pythonで書く','pythonで書く'),('idの値','IDの値')]:
            self.assertEqual(training.skill_for(dict(good=good,bad=bad)),'case')
        with self.assertRaises(ValueError):training.skill_for(dict(skill='unknown'))

if __name__=='__main__':unittest.main()
