# SPDX-License-Identifier: AGPL-3.0-only
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

practice=module('practice','romaji-practice.py')
trainer=module('trainer','distill-qwen.py')

class PracticeTests(unittest.TestCase):
    def test_generated_kana_fallback_is_not_a_learning_reference(self):
        row=dict(input='suzukimi',expected='すずきみ',reading='すずきみ',generated=True,reading_matches=True,candidates=['すずきみ'])
        self.assertIsNotNone(practice.generated_rejection(row))
        memory={'x':dict(input='suzukimi',answer='すずきみ',generated=True,topic='weather')}
        self.assertEqual(practice.retain_memory(memory,strict_generated=True),{})
        self.assertEqual(practice.retain_memory(memory),memory)

    def test_translation_and_mismatched_reading_are_excluded_before_publication(self):
        examples=[dict(input='hatsuyoku',reading='はつよく',expected='喜び',generated=True,reading_matches=False,candidates=['はつよく']),
                  dict(input='supper',reading='すっぱー',expected='夕食',generated=True,reading_matches=False,candidates=['すっぱー']),
                  dict(input='delta',reading='でるた',expected='デルタ',generated=True,reading_matches=False,candidates=['でた'])]
        for row in examples:self.assertIsNotNone(practice.generated_rejection(row))
        valid=dict(input='yorokobi',expected='喜び',generated=True,reading_matches=True,candidates=['喜び'])
        self.assertIsNone(practice.generated_rejection(valid))
        self.assertIsNotNone(practice.generated_rejection(dict(valid,candidates=['よろこび'])))

    def test_generated_reference_requires_matching_reading_and_native_candidate(self):
        row=dict(id='new',split='train',generated=True,actual='誤答',expected='正答',reading_matches=True,
                 teacher_selected_answer='正答',choices=[0,1],candidates=['誤答','正答'])
        self.assertEqual(practice.approved_mistakes([row]),[row])
        for change in [dict(row,reading_matches=False),dict(row,candidates=['誤答'])]:
            self.assertEqual(practice.approved_mistakes([change]),[])

    def test_generated_questions_exclude_duplicates_and_evaluation_inputs(self):
        valid=dict(topic='everyday',input='neko ga neru',reading='ねこ が ねる',expected='猫が寝る',context='猫が眠る')
        held=dict(valid,input=practice.HELD_OUT[0][0])
        duplicate=dict(valid,context='different')
        response=dict(choices=[dict(message=dict(content=json.dumps(dict(questions=[valid,held,duplicate]))))])
        with tempfile.TemporaryDirectory() as directory,patch.object(trainer,'request',return_value=response):
            seen={practice.HELD_OUT[0][0]:True}
            result=practice.generate_questions(trainer,None,'','',Path(directory),seen)
            self.assertEqual(len(result),1)
            self.assertFalse(result[0]['reference_verified'])
            self.assertTrue(result[0]['generated'])

    def test_only_verified_training_mistakes_are_learned(self):
        row=dict(id='x',split='train',actual='箸',expected='橋',teacher_answer='橋',choices=[0,1])
        self.assertEqual(practice.approved_mistakes([row]),[row])
        for changed in [dict(row,split='test'),dict(row,actual='橋'),dict(row,teacher_answer='箸'),
                        dict(row,choices=[0,0]),dict(row,choices=[False,True])]:
            self.assertEqual(practice.approved_mistakes([changed]),[])

    def test_topic_counts_are_equal_and_rotation_covers_bank(self):
        seen=set()
        for n in range(3):
            rows=practice.problems(n)
            for topic in practice.TOPICS:
                self.assertEqual(sum(r.get('topic')==topic and r['split']=='train' for r in rows),4)
            seen.update(r['input'] for r in rows if r['split']=='train')
        self.assertEqual(seen,{r[0] for r in practice.BANK})

    def test_topic_weights_have_equal_total_and_validation_is_unchanged(self):
        originals=[dict(group=str(i),split='train',context='',good='AI',bad='愛') for i in range(12)]
        originals+=[dict(group='other',split='train',context='',good='橋',bad='箸'),
                    dict(group='validation',split='validation',context='',good='AI',bad='愛')]
        rows=practice.balanced_dataset(originals,{})
        for topic in {r['topic'] for r in rows if r['split']=='train'}:
            self.assertAlmostEqual(sum(r['training_weight'] for r in rows if r.get('topic')==topic),1)
        self.assertEqual(rows[-1],originals[-1])

    def test_duplicate_contexts_cannot_dominate_memory(self):
        memory={}
        for i in range(40):
            text=practice.BANK[0][0];ctx=str(i)
            memory[practice.identity(text,ctx)]=dict(input=text,context=ctx,answer='橋',bad='箸',choices=[0,1])
        kept=practice.retain_memory(memory)
        self.assertEqual(len(kept),2)
        self.assertEqual([r['context'] for r in kept.values()],['38','39'])

    def test_same_reading_different_context_never_shares_correction(self):
        key=practice.identity('ai','人工知能')
        memory={key:dict(answer='AI')}
        row=dict(id=practice.identity('ai','恋人'),actual='愛')
        self.assertEqual(practice.learned_answer(row,memory),'愛')
        self.assertEqual(practice.learned_answer(dict(row,id=key),memory),'AI')

    def test_trained_scores_must_prefer_correction_before_it_is_used(self):
        row=dict(id='example',context='',actual='誤答')
        memory={'example':dict(answer='正答')}
        class Ranker:
            @staticmethod
            def score(weights,context,text):
                return weights[0] if text=='正答' else 0
        self.assertEqual(practice.ranked_answer(row,memory,[-1],Ranker),'誤答')
        self.assertEqual(practice.ranked_answer(row,memory,[1],Ranker),'正答')
        self.assertEqual(practice.ranked_answer(row,memory,None,Ranker),'誤答')

    def test_fixed_tests_and_romaji_inputs_are_separate(self):
        rows=practice.problems()
        self.assertTrue(all(r['input'].isascii() for r in rows))
        train={r['input'] for r in rows if r['split']=='train'}
        test={r['input'] for r in rows if r['split']=='test'}
        self.assertFalse(train & test)
        self.assertEqual(len({r['id'] for r in rows}),len(rows))

    def test_cannot_learn_held_out_input_even_under_another_context(self):
        text=practice.HELD_OUT[0][0];context='changed'
        record=dict(input=text,context=context,answer='橋を渡る',bad='箸を渡る',choices=[0,1])
        with self.assertRaises(ValueError):
            practice.training_records({practice.identity(text,context):record})

    def test_end_to_end_learning_reproduces_and_never_teaches_test_answers(self):
        class FakeConverter:
            def __init__(self,*_): pass
            def convert(self,row):
                return dict(actual='誤答',candidates=['誤答'],elapsed_ms=1)
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);output=cache/'runs'/'round';output.mkdir(parents=True)
            (cache/'romaji-config.json').write_text('{}')
            with patch.object(practice,'CACHE',cache),patch.object(practice,'Converter',FakeConverter), \
                 patch.object(practice,'translated',side_effect=AssertionError('Unused free translation must be skipped')), \
                 patch.object(practice,'generate_questions',return_value=[]), \
                 patch.object(trainer,'choose',side_effect=lambda o,b,k,r,reverse=False:int(reverse)):
                practice.run(trainer,None,'','',output)
                practice.verify(trainer,output)
                trainer.verify(output,output/'unused.jsonl')
            result=json.loads((output/'practice-results.json').read_text(encoding='utf-8'))
            self.assertEqual(result['learned'],sum(r['split']=='train' for r in practice.problems()))
            report=json.loads((output/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['free_translation_requests'],0)
            self.assertEqual(report['teacher_candidate_requests'],2*sum(r['split']=='train' for r in practice.problems()))
            for row in result['results']:
                self.assertEqual(row['after'],row['expected'] if row['split']=='train' else '誤答')
            (output/'native-model.bin').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                practice.verify(trainer,output)

if __name__=='__main__':
    unittest.main()
