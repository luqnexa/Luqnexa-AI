# SPDX-License-Identifier: AGPL-3.0-only
"""Learn reviewed English spellings from training only into a separate NEXC journal."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('english_journal_training',HERE/'run-training.py')
training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)

def spellings(rows):
    words=set()
    for row in rows:
        if row['split']!='train':raise ValueError('Only training lessons may supply learned spellings')
        if not training.is_mixed_japanese_english(row):continue
        for surface in re.findall(r'[A-Za-z]+(?: [A-Za-z]+)*',row['good']):
            # Multiword English phrases retain spaces in the surface but accept
            # the continuous key sequence used by romanized sentence input.
            reading=surface.replace(' ','').lower()
            if 3<=len(reading)<=64:words.add((reading,surface))
    return sorted(words)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--cli',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Refusing to append to an existing journal')
    rows=training.ranker.read_records(args.training)
    training.check_mixed_holdout(rows)
    training.mixed_selection(rows)
    words=spellings(rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def sha(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    before=sha(args.base)
    for reading,surface in words:
        subprocess.run([str(args.cli),'dict','add',str(args.base),str(args.output),reading,surface,'english','10000'],check=True,stdout=subprocess.DEVNULL,timeout=30)
    if sha(args.base)!=before:raise ValueError('Base dictionary was modified')
    report=dict(entries=len(words),spellings=words,source_sha256=sha(args.training),base_sha256=before,journal_sha256=sha(args.output),cli_sha256=sha(args.cli),frequency='Uniform illustrative prior, not measured corpus frequency',test_data_used=False,installed=False,limits='English lexical entries only; no full-sentence lookup, Japanese grammar, or candidate-ranking weights learned here.')
    args.output.with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='spellings'},ensure_ascii=True))

if __name__=='__main__':main()
