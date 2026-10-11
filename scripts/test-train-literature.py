# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('literature', Path(__file__).with_name('train-literature.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LiteratureTests(unittest.TestCase):
    def test_rights_gate_checks_every_contributor(self):
        clear = dict(作品著作権フラグ='なし', 人物著作権フラグ='なし', 役割フラグ='著者')
        self.assertTrue(module.allowed_aozora([clear]))
        self.assertFalse(module.allowed_aozora([]))
        self.assertFalse(module.allowed_aozora([clear, dict(clear, 人物著作権フラグ='あり')]))
        self.assertFalse(module.allowed_aozora([dict(clear, 作品著作権フラグ='あり')]))
        self.assertFalse(module.allowed_aozora([dict(clear, 役割フラグ='翻訳者')]))

    def test_body_and_fragment_extraction(self):
        text = '題名\n'+'-'*30+'\n記法案内\n'+'-'*30+'\n｜漢字《かんじ》を読む。［＃注記］\n底本：書誌\n'
        body = module.clean_aozora(text)
        self.assertIn('漢字を読む。', body)
        self.assertNotIn('かんじ', body)
        self.assertNotIn('書誌', body)
        with self.assertRaises(ValueError): module.clean_aozora('不明な形式')
        html = b'<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Heading</h1><p>Hello <em>world</em>.</p></body></html>'
        self.assertEqual(module.clean_xhtml(html), 'Hello world.')
        chunks = list(module.sentences('あ'*320+'。', 'ja'))
        self.assertEqual([len(chunk) for chunk in chunks], [160,160])

    def test_bundled_artifact_provenance(self):
        models = Path(__file__).resolve().parents[1]/'models'
        report = json.loads((models/'literature-training.json').read_text(encoding='utf-8'))
        binary = (models/'context-literature.bin').read_bytes()
        self.assertEqual(hashlib.sha256(binary).hexdigest(), report['model_sha256'])
        self.assertEqual(len(binary), 8+16*report['retained_features'])
        self.assertEqual(len(report['works']), report['total_books'])
        self.assertEqual(len({work['id'] for work in report['works']}), report['total_books'])
        for work in report['works']:
            self.assertIn(work['partition'], ('train','validation','test'))
            self.assertEqual(len(work['sha256']), 64)
            if work['source']=='Aozora Bunko':
                self.assertEqual(work['copyright_work'], 'なし')
                self.assertEqual(work['copyright_person'], 'なし')
            else:
                self.assertIn('CC0 1.0',work['original_license'])
                self.assertLessEqual(work['author_death'],1900)


if __name__ == '__main__':
    unittest.main()
