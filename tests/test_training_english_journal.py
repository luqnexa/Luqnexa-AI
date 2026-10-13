import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('english_journal',Path(__file__).resolve().parents[1]/'models/chatgpt-training/build-english-journal.py')
journal=importlib.util.module_from_spec(spec);spec.loader.exec_module(journal)

class JournalTests(unittest.TestCase):
    def test_learns_english_tokens_not_complete_answers(self):
        rows=[dict(split='train',good='NewsRoomでlost and foundを確認する',language_pair='ja-en')]
        self.assertEqual(journal.spellings(rows),[('lostandfound','lost and found'),('newsroom','NewsRoom')])

    def test_refuses_evaluation_examples(self):
        with self.assertRaises(ValueError):journal.spellings([dict(split='test',good='Toolで書く')])

    def test_does_not_force_ai_or_foreign_words(self):
        self.assertEqual(journal.spellings([dict(split='train',good='AIを読む'),dict(split='train',good='merciと書く')]),[])

if __name__=='__main__':unittest.main()
