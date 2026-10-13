# SPDX-License-Identifier: AGPL-3.0-only
"""Read-only frozen mixed-language evaluation; no teacher, memory or deployment."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def session_candidates(stdout):
    candidates=[]
    for line in stdout.splitlines():
        if line.startswith('preedit: '):candidates=[]
        match=re.match(r'^[ >]+[0-9]+\. (.*)$',line)
        if match:candidates.append(match.group(1))
    if not stdout.rstrip().endswith('learning: off features=0'):
        raise ValueError('Session learning was not disabled for evaluation')
    return candidates

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cli',type=Path,help='Optional isolated CLI; this binary may bundle a different model')
    parser.add_argument('--native-config',type=Path,help='Optional isolated FFI using a full dictionary and preceding context; bundled model')
    parser.add_argument('--nud',type=Path,help='Use this read-only base dictionary with the isolated CLI')
    parser.add_argument('--journal',type=Path,help='Separate learned dictionary journal; requires --nud')
    parser.add_argument('--session-cli',action='store_true',help='Evaluate actual character-by-character session with bundled local AI and supplied context')
    parser.add_argument('--split',choices=('test','validation'),default='test',help='Use frozen validation for diagnosis; do not tune to test answers')
    args=parser.parse_args()
    if args.journal and not args.nud:raise ValueError('Journal needs a base dictionary')
    if (args.nud or args.journal) and not args.cli:raise ValueError('Dictionary options require --cli')
    if args.session_cli and not args.cli:raise ValueError('Session evaluation needs a CLI')
    stem='mixed-heldout-v1' if args.split=='test' else 'mixed-validation-v1'
    data=HERE/(stem+'.jsonl')
    manifest=json.loads((HERE/(stem+'-manifest.json')).read_text(encoding='utf-8'))
    if sha(data)!=manifest['sha256']:raise ValueError('Frozen mixed evaluation changed')
    ranker=module('mixed_eval_ranker',ROOT/'scripts/train-contrast.py')
    rows=ranker.read_records(data)
    raw=args.model.read_bytes()
    if len(raw)!=8200 or raw[:8]!=b'NXCR0001':raise ValueError('Unsupported model')
    weights=[x/1024 for x in struct.unpack('<4096h',raw[8:])]
    pair=ranker.evaluate(weights,rows)
    compiled_identity=None
    if args.session_cli:
        info=subprocess.run([str(args.cli),'--model-info'],capture_output=True,text=True,encoding='utf-8',check=True,timeout=30)
        compiled_identity=json.loads(info.stdout)
        if compiled_identity['research_model'] and compiled_identity['model_sha256']!=sha(args.model):
            raise ValueError('Session research model differs from the reported candidate model')
    converter=None
    config=None
    if args.native_config:
        config=json.loads(args.native_config.read_text(encoding='utf-8'))
        practice=module('mixed_native_practice',ROOT/'scripts/romaji-practice.py')
        converter=practice.Converter(config)
    details=[]
    for row in rows:
        item=dict(row,candidate_pair_correct=ranker.score(weights,row['context'],row['good'])>ranker.score(weights,row['context'],row['bad'])+1e-9)
        if converter:
            native=converter.convert(row)
            item.update(native_ffi=native,native_ffi_top_one_correct=native['actual']==row['good'],native_ffi_top_eight_contains_reference=row['good'] in native['candidates'])
        if args.cli:
            command=[str(args.cli)]
            if args.session_cli:command+=['session']
            if args.nud:command+=['--nud',str(args.nud)]
            if args.journal:command+=['--journal',str(args.journal)]
            if args.session_cli:
                # Turn off session learning before the context is committed.
                # Each row gets a fresh process; nothing persists or transmits.
                commands=['training','text '+row['context'],'literal','enter']
                commands+=['text '+char for char in row['input']]
                commands+=['space','quit']
                env=dict(os.environ,LUQNEXA_AI_BACKEND='original')
                result=subprocess.run(command+['--limit','8'],input='\n'.join(commands)+'\n',env=env,capture_output=True,text=True,encoding='utf-8',check=True,timeout=30)
                candidates=session_candidates(result.stdout)
            else:
                result=subprocess.run(command+['--limit','8','--',row['input']],capture_output=True,text=True,encoding='utf-8',check=True,timeout=30)
                candidates=[line.split('. ',1)[1].split('\t',1)[0] for line in result.stdout.splitlines() if '. ' in line and line.split('. ',1)[0].isdigit()]
            item.update(native_candidates=candidates,native_top_one_correct=bool(candidates and candidates[0]==row['good']),native_top_eight_contains_reference=row['good'] in candidates)
        details.append(item)
    report=dict(dataset_sha256=sha(data),model_sha256=sha(args.model),examples=len(rows),candidate_pair=pair,candidate_pair_percent=100*pair['wins']/len(rows),details=details,trained_on_evaluation=False,independently_reviewed=False,
        limits='Correct candidate supplied for pair ranking. Plain CLI uses dictionary/frequency scoring without AI or context. Session CLI uses the compiled model identified by --model-info and synthetic preceding context; --model alone does not select runtime weights. Neither is installed desktop IME performance.')
    if args.cli:
        wins=sum(r['native_top_one_correct'] for r in details)
        report['native_cli']=dict(binary_sha256=sha(args.cli),top_one_wins=wins,examples=len(rows),top_one_percent=100*wins/len(rows),top_eight_hits=sum(r['native_top_eight_contains_reference'] for r in details))
        model_source='No AI: dictionary and frequency only'
        if args.session_cli:
            model_source='Explicit research snapshot' if compiled_identity['research_model'] else 'Bundled local AI'
        report['native_cli'].update(session_ai=args.session_cli,character_by_character=args.session_cli,preceding_context_supplied=args.session_cli,model_source=model_source,learning_during_evaluation=False)
        if compiled_identity:report['native_cli']['compiled_model_identity']=compiled_identity
        if args.nud:report['native_cli']['dictionary_sha256']=sha(args.nud)
        if args.journal:report['native_cli']['journal_sha256']=sha(args.journal)
    if converter:
        wins=sum(r['native_ffi_top_one_correct'] for r in details)
        report['native_ffi']=dict(binary_sha256=sha(Path(config['dll'])),dictionary_sha256=sha(Path(config['nud'])),top_one_wins=wins,examples=len(rows),top_one_percent=100*wins/len(rows),top_eight_hits=sum(r['native_ffi_top_eight_contains_reference'] for r in details),preceding_context_supplied=True,model_source='Bundled in this isolated DLL; NOT the candidate model passed via --model',typing_history_used=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report['partition']=args.split
    report['selection_bias']=args.split=='validation'
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('details','limits')},ensure_ascii=False))

if __name__=='__main__':main()
