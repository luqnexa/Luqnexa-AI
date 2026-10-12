# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('contrast', Path(__file__).with_name('train-contrast.py'))
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class ContrastTests(unittest.TestCase):
    def test_artifact_and_held_out_provenance(self):
        models = trainer.ROOT/'models'
        data = models/'context-contrast.jsonl'
        report = json.loads((models/'contrast-training.json').read_text(encoding='utf-8'))
        model = (models/'context-contrast.bin').read_bytes()
        self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(), report['data_sha256'])
        self.assertEqual(hashlib.sha256(model).hexdigest(), report['model_sha256'])
        self.assertEqual(len(model), 8200)
        self.assertEqual(model[:8], b'NXCR0001')
        weights = [v/1024 for v in struct.unpack('<4096h', model[8:])]
        rows = trainer.read_records(data)
        expected_anchors = ''.join(a+'\t'+b+'\n' for a,b in trainer.anchors(rows))
        self.assertEqual((models/'contrast-anchors.tsv').read_text(encoding='utf-8'), expected_anchors)
        for part in ('train', 'validation', 'test', 'regression'):
            self.assertEqual(trainer.evaluate(weights, [r for r in rows if r['split']==part]),
                             report['evaluation'][part])
        expected = ''.join('\t'.join([r['split'],r['group'],r['context'],r['good'],r['bad']])+'\n'
                           for r in rows if r['split'] in ('test','regression'))
        self.assertEqual((models/'contrast-evaluation.tsv').read_text(encoding='utf-8'), expected)

    def test_test_partition_never_updates_weights(self):
        rows = [dict(group=part,split=part,context='評価します。',good='精度がいい',bad='精度害意')
                for part in ('train','validation','test')]
        first, epoch = trainer.fit(rows, epochs=8)
        rows[2]['good'], rows[2]['bad'] = rows[2]['bad'], rows[2]['good']
        second, second_epoch = trainer.fit(rows, epochs=8)
        self.assertEqual(first, second)
        self.assertEqual(epoch, second_epoch)

    def test_rejects_duplicate_and_group_leakage(self):
        row = dict(group='group',split='train',context='',good='正しい',bad='誤変換')
        for rows in ([row,row], [row,dict(row,split='test',good='別の例')]):
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'examples.jsonl'
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
                with self.assertRaises(ValueError):
                    trainer.read_records(path)

    def test_unicode_feature_hash_reference(self):
        self.assertEqual(trainer.slot('s:がいい'),1852)
        self.assertEqual(trainer.features('古い'+'あ'*24,'文章'),trainer.features('あ'*24,'文章'))


if __name__=='__main__':
    unittest.main()
