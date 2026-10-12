# SPDX-License-Identifier: AGPL-3.0-only
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('distill', Path(__file__).with_name('distill-qwen.py'))
distill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(distill)


class DistillationTests(unittest.TestCase):
    def test_proper_names_and_explicit_identifiers_use_exact_case(self):
        for correct,wrong in [('Python','python'),('Rust','rust'),('GitHub','Github'),('AI','Ai'),('CPU','Cpu')]:
            self.assertTrue(distill.valid_candidate_casing(correct+'を使う','正式な製品名を使う'))
            self.assertFalse(distill.valid_candidate_casing(wrong+'を使う','正式な製品名を使う'))
        self.assertTrue(distill.valid_candidate_casing('pythonを使う','変数名「python」をそのまま表示'))
        self.assertFalse(distill.valid_candidate_casing('Pythonを使う','変数名「python」をそのまま表示'))
        self.assertTrue(distill.valid_candidate_casing('Pythonでaiを使う','変数名「ai」を扱う'))

    def test_teacher_wrong_case_is_rejected_in_both_orders(self):
        row=dict(context='言語の正式名を使う',good='Pythonで書く',bad='pythonで書く')
        for reverse,index in [(False,1),(True,0)]:
            reply=dict(choices=[dict(message=dict(content=json.dumps(dict(index=index))))])
            with patch.object(distill,'request',return_value=reply):
                with self.assertRaises(ValueError):distill.choose(None,'','',row,reverse)

    def test_teacher_cache_is_bound_to_pair_context_prompt_and_model(self):
        row = dict(context='川を渡る。', good='橋を渡る', bad='箸を渡る')
        key = distill.cache_key(row, 'model-a')
        self.assertNotEqual(key, distill.cache_key(row, 'model-b'))
        self.assertNotEqual(key, distill.cache_key(dict(row, context='料理を食べる。'), 'model-a'))
        self.assertNotEqual(key, distill.cache_key(dict(row, good=row['bad'], bad=row['good']), 'model-a'))
        with patch.object(distill, 'PROMPT', 'different prompt'):
            self.assertNotEqual(key, distill.cache_key(row, 'model-a'))
    def test_display_file_lock_does_not_abort_training(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, 'replace', side_effect=PermissionError), patch.object(distill.time, 'sleep'):
                distill.publish_question(Path(directory), 'generation')
            distill.publish_question(Path(directory), 'generation')
            self.assertEqual(json.loads((Path(directory)/'current-question.json').read_text())['phase'],'generation')

    def test_rejects_order_bias_and_invalid_labels(self):
        self.assertTrue(distill.accepted([0, 1]))
        for choices in ([0, 0], [1, 1], [1, 0], [False, True], [0], []):
            self.assertFalse(distill.accepted(choices))

    def test_held_out_labels_do_not_update_weights(self):
        base = distill.pairwise.read_records(distill.ROOT / 'models/native-training.jsonl')
        extra = distill.pairwise.read_records(distill.ROOT / 'models/qwen-teacher-pairs.jsonl')
        first, epoch = distill.pairwise.fit(base + extra, epochs=3)
        changed = [dict(r, good=r['bad'], bad=r['good']) if r['split'] == 'test' else r for r in extra]
        second, next_epoch = distill.pairwise.fit(base + changed, epochs=3)
        self.assertEqual((first, epoch), (second, next_epoch))

    def test_model_serialization_and_checksum(self):
        binary, weights = distill.quantize([100.0, -100.0] + [0.0] * 4094)
        self.assertEqual(len(binary), 8200)
        self.assertEqual(binary[:8], b'NXCR0001')
        self.assertEqual(weights[:2], [32767 / 1024, -32768 / 1024])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.bin'
            path.write_bytes(binary)
            self.assertEqual(distill.sha(path), distill.hashlib.sha256(binary).hexdigest())

    def test_probes_are_separate_from_previous_examples(self):
        probes = distill.pairwise.read_records(distill.ROOT / 'models/qwen-teacher-pairs.jsonl')
        base = distill.pairwise.read_records(distill.ROOT / 'models/native-training.jsonl')
        identity = lambda r: (r['context'], r['good'], r['bad'])
        self.assertFalse(set(map(identity, probes)) & set(map(identity, base)))
        self.assertEqual(sum(r['split'] == 'test' for r in probes), 8)

    def test_teacher_cannot_return_invalid_or_boolean_index(self):
        row = dict(context='川を渡る。', good='橋を渡る', bad='箸を渡る')
        for index in (True, -1, 2, '0'):
            reply = {'choices': [{'message': {'content': json.dumps({'index': index})}}]}
            with patch.object(distill, 'request', return_value=reply):
                with self.assertRaises(ValueError):
                    distill.choose(None, '', '', row)

    def test_runtime_scope_preserves_unseen_candidates(self):
        rows = [dict(context='', good='橋を渡る', bad='箸を渡る')]
        result = distill.runtime_evaluate([1.0] * 4096, [('AIを使う', '愛を使う')], rows)
        self.assertEqual(result, dict(pairs=1, wins=0, ties=1, losses=0))


if __name__ == '__main__':
    unittest.main()
