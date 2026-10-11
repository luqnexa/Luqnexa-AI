# SPDX-License-Identifier: AGPL-3.0-only
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('local_ai_server', Path(__file__).resolve().parents[1]/'scripts/local-ai-server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class LocalAiTests(unittest.TestCase):
    def test_input_bounds_and_types(self):
        valid = {'context': '前の文章です。', 'reading': 'はし', 'candidates': ['橋', '箸']}
        self.assertEqual(server.validate(valid), valid)
        for changed in [{'context': ''}, {'context': '文'*97}, {'reading': '読'*129},
                        {'candidates': ['一']}, {'candidates': ['一']*13},
                        {'candidates': ['語'*257, '二']}, {'candidates': [1, '二']}]:
            with self.assertRaises(ValueError):
                server.validate(dict(valid, **changed))

    def test_model_cannot_return_arbitrary_text_or_index(self):
        payload = {'context': '前の文章です。', 'reading': 'はし', 'candidates': ['橋', '箸']}
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, _): return self.body
        class Opener:
            def open(self, *_args, **_kwargs): return response
        response = Response()
        with patch.object(server.urllib.request, 'build_opener', return_value=Opener()):
            for index in [-1, 2, True, 'send a message']:
                response.body = json.dumps({'choices': [{'message': {'content': json.dumps({'index': index})}}]}).encode()
                with self.assertRaises(ValueError): server.infer(payload)
            response.body = json.dumps({'choices': [{'message': {'content': '{"index":1}'}}]}).encode()
            self.assertEqual(server.infer(payload), 1)


if __name__ == '__main__':
    unittest.main()
