# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare a general English spelling lexicon from pinned, locally built ESDB."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Refusing to replace source-derived words')
    source=args.source.resolve()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    words=subprocess.check_output([sys.executable,'-X','utf8','-m','libscowl','word-list','60','A','1'],cwd=source,text=True,encoding='utf-8').splitlines()
    # Restrict this first index to bare Latin words. Punctuation, possessives,
    # identifiers and short romaji particles are deliberately not imported.
    accepted=sorted({word for word in words if re.fullmatch('[A-Za-z]{3,48}',word)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        for word in accepted:stream.write(f'{word.lower()}\t{word}\tenglish\t2000\n')
    notices=args.output.with_suffix('.Copyright.txt')
    shutil.copyfile(source/'Copyright',notices)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(source='https://github.com/en-wl/wordlist',commit=commit,selection='ESDB size 60; American English; variant level 1',upstream_words=len(words),indexed_words=len(accepted),excluded_words=len(words)-len(accepted),tsv_sha256=sha(args.output),copyright_sha256=sha(notices),database_sha256=sha(source/'scowl.db'),test_or_validation_data_used=False,teacher_inference_used=False,
        limits='General spelling dictionary, not neural training or independent sentence evaluation. Uniform frequency 2000 is an illustrative prior. Retain original Copyright; dictionary is not relicensed as AGPL.')
    args.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
