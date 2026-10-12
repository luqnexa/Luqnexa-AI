# SPDX-License-Identifier: AGPL-3.0-only
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import hashlib
import http.client
import threading
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('dashboard',Path(__file__).with_name('learning-dashboard.py'))
dashboard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


class DashboardTests(unittest.TestCase):
    def test_answer_endpoint_saves_locally_and_rejects_cross_origin_and_stale_quiz(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'practice-results.json').write_text(json.dumps(dict(quiz_id='quiz',results=[dict(id='case')])) )
            server=dashboard.ThreadingHTTPServer(('127.0.0.1',0),dashboard.Handler)
            server.daemon_threads=True
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            origin=f'http://127.0.0.1:{server.server_port}'
            def post(value,source=origin,route='/api/answer'):
                client=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
                client.request('POST',route,json.dumps(value),headers={'Origin':source,'Content-Type':'application/json'})
                response=client.getresponse();code=response.status;response.read();client.close();return code
            try:
                with patch.object(dashboard,'CACHE',p):
                    valid=dict(quiz_id='quiz',id='case',answer='橋を渡る')
                    self.assertEqual(post(valid),200)
                    self.assertEqual(json.loads((p/'practice-answers.json').read_text(encoding='utf-8'))['case']['answer'],'橋を渡る')
                    self.assertEqual(post(valid,'http://evil.example'),403)
                    self.assertEqual(post(dict(text='[]'),'http://evil.example','/api/lessons'),403)
                    with patch.object(dashboard.LESSON_IMPORT,'start_import',return_value=dict(ok=True,questions=6,run_id='test')) as importer:
                        self.assertEqual(post(dict(text='[]'),origin,'/api/lessons'),200)
                        importer.assert_called_once_with('[]')
                    self.assertEqual(post(dict(valid,quiz_id='old')),409)
                    self.assertEqual(post(dict(valid,answer='x\n')),400)
                    self.assertEqual(post(dict(valid,id='unknown')),409)
            finally:
                server.shutdown();server.server_close();thread.join()

    def test_missing_state_is_not_reported_as_running(self):
        with tempfile.TemporaryDirectory() as directory:
            s=dashboard.snapshot(Path(directory),alive=False)
            self.assertEqual(s['state'],'not_started')
            self.assertIsNone(s['integrity'])

    def test_romaji_results_are_separate_from_teacher_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            report=dict(cases_total=1,cases_passed=1,results=[dict(input='maruichi',expected='①',actual='①',passed=True)])
            (p/'romaji-results.json').write_text(json.dumps(report),encoding='utf-8')
            s=dashboard.snapshot(p,alive=False)
            self.assertEqual(s['romaji']['results'][0]['input'],'maruichi')
            self.assertEqual(s['romaji']['cases_passed'],1)
            self.assertIsNone(s['report']['evaluation'])

    def test_stale_pid_does_not_appear_as_training(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'running.json').write_text('{"pid":123}')
            (p/'status.json').write_text('{"state":"training"}')
            self.assertEqual(dashboard.snapshot(p,alive=False)['state'],'stopped')

    def test_progress_hash_and_evaluation_are_read_from_real_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);run=p/'runs/20261012T090000Z';run.mkdir(parents=True)
            (p/'status.json').write_text(json.dumps(dict(state='training',run=str(run))))
            (run/'training.log').write_text('not-json\n'+json.dumps(dict(processed=3,total=8,accepted=2))+'\n')
            data=b'checked model';(run/'native-model.bin').write_bytes(data)
            report=dict(model_sha256=hashlib.sha256(data).hexdigest(),accepted=2,generated_problems=4,
                        evaluation={'new-held-out-test':{'after':{'wins':5,'pairs':8}}})
            (run/'report.json').write_text(json.dumps(report))
            (p/'latest.json').write_text(json.dumps(dict(run=str(run),report=report,verified=True,installed=False)))
            s=dashboard.snapshot(p,alive=True)
            self.assertEqual(s['progress']['processed'],3)
            self.assertTrue(s['integrity'])
            self.assertEqual(s['history'][0]['wins'],5)
            self.assertFalse(s['installed'])
            (run/'native-model.bin').write_bytes(b'tampered')
            self.assertFalse(dashboard.snapshot(p,alive=True)['integrity'])

    def test_external_paths_and_corrupt_json_are_not_read(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'status.json').write_text('{broken')
            self.assertEqual(dashboard.snapshot(p,alive=True)['state'],'not_started')
            self.assertIsNone(dashboard.safe_run(str(p.parent),p))

    def test_question_text_and_teacher_selections_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);run=p/'runs/round';run.mkdir(parents=True)
            row=dict(group='example',split='train',context='川を渡る。',good='橋を渡る',bad='箸を渡る')
            (run/'probes.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
            (run/'teacher-audit.json').write_text(json.dumps([dict(group='example',accepted=True,choices=[0,1])]))
            (run/'current-question.json').write_text(json.dumps(dict(row,phase='reverse')),encoding='utf-8')
            (p/'status.json').write_text(json.dumps(dict(state='training',run=str(run))))
            s=dashboard.snapshot(p,alive=True)
            self.assertEqual(s['questions'][0]['good'],'橋を渡る')
            self.assertEqual(s['questions'][0]['status'],'accepted')
            self.assertEqual(s['current_question']['phase'],'reverse')
            self.assertEqual(dashboard.snapshot(p,alive=False)['current_question'],{})


if __name__=='__main__':
    unittest.main()
