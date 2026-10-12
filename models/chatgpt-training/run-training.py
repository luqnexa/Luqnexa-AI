# SPDX-License-Identifier: AGPL-3.0-only
"""Reproduce this ChatGPT-authored local training session; never installs a model."""
import hashlib
import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import struct
import time
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

ranker=module('ranker','train-contrast.py')
practice=module('practice','romaji-practice.py')
trainer=module('trainer','distill-qwen.py')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--extra',type=Path,help='New curated training-only examples')
    parser.add_argument('--output',type=Path,help='Separate run artifact directory')
    args=parser.parse_args()
    started=time.monotonic()
    authored=ranker.read_records(HERE/'examples.jsonl')
    extra=ranker.read_records(args.extra) if args.extra else []
    if any(r['split']!='train' for r in extra):raise ValueError('Extra lessons must not modify validation or test sets')
    original=ranker.read_records(ROOT/'models/native-training.jsonl')
    frozen_inputs={r['input'] for r in authored+original if r['split'] in ('validation','test') and r.get('input')}
    assert not ({r['input'] for r in authored+extra if r['split']=='train'} & frozen_inputs)
    previous=None
    status_path=ROOT/'.cache/chatgpt-training/status.json'
    if status_path.exists():
        status=json.loads(status_path.read_text(encoding='utf-8'))
        previous_dir=Path(status['latest_run'])
        previous_report=json.loads((previous_dir/'report.json').read_text(encoding='utf-8'))
        if previous_report['references_sha256']!=sha(HERE/'examples.jsonl'):
            raise ValueError('Fixed authored references changed')
        if previous_report.get('original_references_sha256',sha(ROOT/'models/native-training.jsonl'))!=sha(ROOT/'models/native-training.jsonl'):
            raise ValueError('Fixed original references changed')
        previous_model=previous_dir/'native-model.bin'
        if sha(previous_model)!=previous_report['model_sha256']:
            raise ValueError('Previous model hash mismatch')
        previous=[x/1024 for x in struct.unpack('<4096h',previous_model.read_bytes()[8:])]
    dataset=original+authored+extra
    counts={}
    for r in dataset:
        if r['split']=='train':
            r['topic']=r.get('topic',practice.record_topic(r))
            counts[r['topic']]=counts.get(r['topic'],0)+1
    for r in dataset:
        if r['split']=='train':r['training_weight']=1/counts[r['topic']]
    output=args.output or ROOT/'.cache/chatgpt-training'/datetime.datetime.now(datetime.timezone.utc).strftime('session-%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True,exist_ok=True)
    path=output/'training.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in dataset),encoding='utf-8')
    dataset=ranker.read_records(path)
    weights,epoch=ranker.fit(dataset)
    binary,quantized=trainer.quantize(weights)
    model=output/'native-model.bin';model.write_bytes(binary)
    assert trainer.quantize(ranker.fit(dataset)[0])[0]==binary
    baseline=[x/1024 for x in struct.unpack('<4096h',(ROOT/'models/native-model.bin').read_bytes()[8:])]
    anchors=ranker.anchors(dataset)
    (output/'native-anchors.tsv').write_text(''.join(a+'\t'+b+'\n' for a,b in anchors),encoding='utf-8')
    evaluation={}
    for name,rows in [('authored-training',[r for r in authored if r['split']=='train']),
                      ('authored-validation',[r for r in authored if r['split']=='validation']),
                      ('authored-held-out',[r for r in authored if r['split']=='test']),
                      ('original-test',[r for r in original if r['split']=='test'])]:
        evaluation[name]=dict(before=ranker.evaluate(baseline,rows),after=ranker.evaluate(quantized,rows))
        if previous is not None:evaluation[name]['previous']=ranker.evaluate(previous,rows)
    details=[dict(r,before_prefers_reference=ranker.score(baseline,r['context'],r['good'])>ranker.score(baseline,r['context'],r['bad']),
                    after_prefers_reference=ranker.score(quantized,r['context'],r['good'])>ranker.score(quantized,r['context'],r['bad']))
             for r in authored if r['split']=='test']
    report=dict(teacher='ChatGPT in this conversation; authored references',method='Local pairwise ranking SGD',
        authored_examples=len(authored)+len(extra),extra_training_examples=len(extra),extra_training_evaluation=ranker.evaluate(quantized,extra),extra_sha256=sha(args.extra) if args.extra else None,teacher_weights_updated=False,installed=False,network_used=False,
        independently_reviewed=False,training_topic_counts=counts,selected_epoch=epoch,
        evaluation=evaluation,held_out_details=details,model_sha256=sha(model),model_bytes=len(binary),
        training_sha256=sha(path),references_sha256=sha(HERE/'examples.jsonl'),reproducible=True,
        runtime_seconds=round(time.monotonic()-started,2),
        limits='Candidate-pair scores with the reference supplied. Held-out sentences share lexical families. Not end-to-end IME accuracy, not training ChatGPT weights, not installed in current IME.')
    original_test=evaluation['original-test']
    report['eligible_for_ime']=False
    report['original_references_sha256']=sha(ROOT/'models/native-training.jsonl')
    report['previous_model_sha256']=previous_report['model_sha256'] if previous is not None else None
    report['passes_candidate_regression_gate']=original_test['after']['wins']>=original_test['before']['wins'] and all(
        result['after']['wins']>=result.get('previous',result['before'])['wins']
        and result['after']['losses']<=result.get('previous',result['before'])['losses']
        for name,result in evaluation.items() if name in ('authored-validation','authored-held-out','original-test'))
    report['deployment_requires']='Independent and end-to-end IME evaluation; candidate-pair gate alone is insufficient'
    report['rejection_reason']=None if report['passes_candidate_regression_gate'] else 'Fixed candidate evaluation regressed against previous accepted model or original baseline; preserve experiment without adopting.'
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if report['passes_candidate_regression_gate']:
        (ROOT/'benchmarks/results/chatgpt-training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),evaluation=evaluation,seconds=report['runtime_seconds'],model_sha256=report['model_sha256']),ensure_ascii=False))

if __name__=='__main__':
    lock_path=ROOT/'.cache/chatgpt-training/training.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+b') as stream:
        stream.seek(0)
        if sys.platform=='win32':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
