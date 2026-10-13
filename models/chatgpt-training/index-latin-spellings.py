# SPDX-License-Identifier: AGPL-3.0-only
"""Index existing public dictionary's attested Latin surfaces, without test data."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

def latin_entry(line):
    fields=line.rstrip('\n').split('\t')
    if len(fields)!=4:raise ValueError('Invalid unpacked entry')
    _,surface,_,frequency=fields
    if not re.fullmatch(r'[A-Za-z]{3,}(?: [A-Za-z]+)*',surface):return None
    reading=surface.replace(' ','').lower()
    if len(reading)>64:return None
    return reading,surface,'english',int(frequency)

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cli',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Refusing to replace a spelling index')
    before=sha(args.base)
    entries={};examined=0
    with subprocess.Popen([str(args.cli),'dict','unpack',str(args.base)],stdout=subprocess.PIPE,text=True,encoding='utf-8') as process:
        for line in process.stdout:
            examined+=1
            entry=latin_entry(line)
            if entry:
                key=entry[:3]
                entries[key]=max(entries.get(key,0),entry[3])
        if process.wait()!=0:raise ValueError('Public base dictionary export failed')
    if sha(args.base)!=before:raise ValueError('Base changed during export')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        for key,frequency in sorted(entries.items()):stream.write('\t'.join((*key,str(frequency)))+'\n')
    report=dict(base_sha256=before,entries_examined=examined,latin_spelling_entries=len(entries),index_sha256=sha(args.output),test_or_validation_data_read=False,network_used=False,
        provenance='Attested surfaces from existing public base. Retain upstream dictionaries/MOZC-NOTICES.txt. Data is not relicensed as AGPL.',
        limits='Latin spellings may include names or non-English words; this is not an English language classifier or neural training. Original dictionary frequency priors are retained.')
    args.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
