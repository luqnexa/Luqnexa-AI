import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('latin_index',Path(__file__).resolve().parents[1]/'models/chatgpt-training/index-latin-spellings.py')
index=importlib.util.module_from_spec(spec);spec.loader.exec_module(index)

class LatinIndexTests(unittest.TestCase):
    def test_keeps_attested_spelling_and_prior(self):
        self.assertEqual(index.latin_entry('例\tWordDesk\tnoun\t4321\n'),('worddesk','WordDesk','english',4321))
        self.assertEqual(index.latin_entry('例\topen day\tnoun\t80\n'),('openday','open day','english',80))

    def test_japanese_sentences_urls_short_tokens_are_not_indexed(self):
        for text in ['日本語','abcを書く','https://example.com','AI','file_name','x'*65]:
            self.assertIsNone(index.latin_entry(f'例\t{text}\tnoun\t1000'))

if __name__=='__main__':unittest.main()
