from pathlib import Path
import subprocess
import tempfile
import unittest

CLI=Path('C:/opencode-rust/target/release/nextime.exe')

@unittest.skipUnless(CLI.exists(),'Build the Rust CLI before this integration check')
class BulkDictionaryTests(unittest.TestCase):
    def run_cli(self,*args):
        return subprocess.run([str(CLI),*map(str,args)],capture_output=True,text=True,encoding='utf-8',timeout=30)

    def test_invalid_tsv_does_not_append_and_valid_bulk_is_searchable(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/'base.nud';journal=root/'words.nxl'
            source=root/'base.tsv';source.write_text('で\tで\tparticle\t9000\n',encoding='utf-8')
            self.assertEqual(self.run_cli('dict','build',source,base).returncode,0)
            self.assertEqual(self.run_cli('dict','add',base,journal,'sampleword','SampleWord','english','9000').returncode,0)
            before=journal.read_bytes()
            source.write_text('otherword\tOtherWord\tenglish\t9000\ninvalid\n',encoding='utf-8')
            self.assertNotEqual(self.run_cli('dict','add-tsv',base,journal,source).returncode,0)
            self.assertEqual(journal.read_bytes(),before)
            source.write_text('otherword\tOtherWord\tenglish\t9000\n',encoding='utf-8')
            self.assertEqual(self.run_cli('dict','add-tsv',base,journal,source).returncode,0)
            result=self.run_cli('--nud',base,'--journal',journal,'--','otherwordde')
            self.assertEqual(result.returncode,0)
            self.assertIn('reading: otherwordで',result.stdout)
            self.assertIn('OtherWordで',result.stdout)

if __name__=='__main__':unittest.main()
