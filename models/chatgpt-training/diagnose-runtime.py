# SPDX-License-Identifier: AGPL-3.0-only
"""Mine actual competitors from training-only sessions; never adopt automatically."""
import argparse
from collections import Counter
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

training=load('runtime_training',HERE/'run-training.py')
evaluation=load('runtime_evaluation',HERE/'evaluate-mixed.py')

def validate(rows):
    if not rows or len(rows)>24 or any(r['split']!='train' for r in rows):
        raise ValueError('Use at most 24 training-only rows')
    if any(not isinstance(r.get('input'),str) or not r['input'] or len(r['input'])>512
           or any(ord(c)<32 or ord(c)==127 for c in r['input']) for r in rows):
        raise ValueError('Invalid conversion input; session commands cannot be embedded')
    training.check_mixed_holdout(rows)
    training.mixed_selection(rows)
    originals=training.ranker.read_records(HERE/'examples.jsonl')+training.ranker.read_records(HERE.parent/'native-training.jsonl')
    frozen=[r for r in originals if r['split']!='train']
    inputs={r.get('input') for r in frozen}; surfaces={r['good'] for r in frozen}
    if any(r.get('input') in inputs or r['good'] in surfaces for r in rows):
        raise ValueError('Frozen evaluation cannot be mined for training')

def classify(reference,candidates):
    if not candidates or reference not in candidates:return 'missing_candidate'
    return 'correct' if candidates[0]==reference else 'ranking_error'

def score_components(stdout,candidates):
    scores={}
    for line in stdout.splitlines():
        match=re.fullmatch(r'score: ([0-9]+) (-?[0-9]+) (-?[0-9]+)',line)
        if match:
            index,total,native=map(int,match.groups())
            if not 1<=index<=len(candidates) or index in scores or not 0<=native<=4096:
                raise ValueError('Invalid or duplicate score diagnostic')
            scores[index]=dict(surface=candidates[index-1],total=total,native=native,other=total-native)
    if len(scores)!=len(candidates):raise ValueError('Incomplete score diagnostic; use a CLI with scores support')
    return [scores[i] for i in range(1,len(candidates)+1)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lessons',type=Path,required=True)
    p.add_argument('--cli',type=Path,required=True)
    p.add_argument('--nud',type=Path,required=True)
    p.add_argument('--journal',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=training.ranker.read_records(args.lessons);validate(rows)
    identity=json.loads(subprocess.check_output([str(args.cli),'--model-info'],encoding='utf-8'))
    details=[]
    for row in rows:
        commands=['training','text '+row['context'],'literal','enter']
        commands+=['text '+char for char in row['input']]
        commands+=['space','scores','quit']
        result=subprocess.run([str(args.cli),'session','--nud',str(args.nud),'--journal',str(args.journal),'--limit','32'],
            input='\n'.join(commands)+'\n',env=dict(os.environ,LUQNEXA_AI_BACKEND='original'),
            capture_output=True,text=True,encoding='utf-8',timeout=45)
        if result.returncode:raise RuntimeError('Isolated conversion failed: '+result.stderr[:500])
        candidates=evaluation.session_candidates(result.stdout)
        components=score_components(result.stdout,candidates)
        kind=classify(row['good'],candidates)
        # Proposed pairs are only diagnostics. Preserve the authored reference;
        # absent candidates need generation fixes, not a fabricated ranking win.
        pair=dict(row,bad=candidates[0],review_required=True) if kind=='ranking_error' else None
        details.append(dict(input=row['input'],context=row['context'],reference=row['good'],
            skill=training.skill_for(row),classification=kind,candidates=candidates,score_components=components,proposed_pair=pair))
    report=dict(partition='train',examples=len(rows),counts=dict(Counter(r['classification'] for r in details)),
        hashes={key:evaluation.sha(getattr(args,key)) for key in ('lessons','cli','nud','journal')},
        compiled_model_identity=identity,learning_enabled=False,installed=False,details=details,
        limits='Training-only diagnostic with synthetic contexts. Proposed negatives need review; no model training or adoption. Not held-out accuracy.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('details','limits')},ensure_ascii=False))

if __name__=='__main__':main()
