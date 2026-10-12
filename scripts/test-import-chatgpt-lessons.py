# SPDX-License-Identifier: AGPL-3.0-only
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('lesson_import',Path(__file__).with_name('import-chatgpt-lessons.py'))
lesson=importlib.util.module_from_spec(spec);spec.loader.exec_module(lesson)

class ImportTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.cache_patch=patch.object(lesson,'CACHE',Path(self.directory.name));self.cache_patch.start();self.addCleanup(self.cache_patch.stop)
        self.items=json.loads((lesson.ROOT/'benchmarks/manual-paste-example.json').read_text(encoding='utf-8'))
        by_input={r['input']:r['reading'] for r in self.items}
        class Converter:
            def convert(self,row):return dict(reading=by_input.get(row['input'],'incorrect'),actual='検証用候補')
        self.converter=Converter()
    def parse(self,items):return lesson.parse_lessons(json.dumps(items,ensure_ascii=False),self.converter)
    def test_data_only_fenced_json_and_balanced_partitions(self):
        rows=lesson.parse_lessons('```json\n'+json.dumps(self.items,ensure_ascii=False)+'\n```',self.converter)
        self.assertEqual(len(rows),6)
        self.assertTrue(all(r['split']=='train' for r in rows))
        self.assertEqual(len({r['topic'] for r in rows}),6)
    def test_malformed_executable_extra_fields_and_unbalanced_data_are_rejected(self):
        with self.assertRaises(ValueError):lesson.parse_lessons('print("execute")',self.converter)
        for changed in [dict(self.items[0],command='execute'),dict(self.items[0],topic='technology')]:
            with self.assertRaises(ValueError):self.parse([changed]+self.items[1:])
    def test_wrong_reading_case_and_identical_candidates_are_rejected(self):
        for index,change in [(0,dict(reading='しをよむ')),(1,dict(expected='pythonで情報を整理する')),(0,dict(incorrect=self.items[0]['expected']))]:
            items=[dict(r) for r in self.items];items[index].update(change)
            with self.assertRaises(ValueError):self.parse(items)
    def test_fixed_evaluation_cannot_enter_training(self):
        items=[dict(r) for r in self.items];items[0]['input']='hashiwowataru'
        with self.assertRaises(ValueError):self.parse(items)
    def test_duplicate_and_control_characters_are_rejected(self):
        for change in [dict(input=self.items[1]['input']),dict(context='bad\ntext')]:
            items=[dict(r) for r in self.items];items[0].update(change)
            with self.assertRaises(ValueError):self.parse(items)
    def test_overlapping_import_jobs_are_rejected(self):
        with lesson.IMPORT_LOCK:
            with self.assertRaises(RuntimeError):lesson.start_import('[]')
    def test_failed_validation_releases_import_lock(self):
        with self.assertRaises(ValueError):lesson.start_import('[]')
        self.assertTrue(lesson.IMPORT_LOCK.acquire(blocking=False));lesson.IMPORT_LOCK.release()

if __name__=='__main__':unittest.main()
