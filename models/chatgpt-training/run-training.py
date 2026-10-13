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
import re
from collections import Counter, defaultdict

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

ranker=module('ranker','train-contrast.py')
practice=module('practice','romaji-practice.py')
trainer=module('trainer','distill-qwen.py')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def check_mixed_holdout(training):
    """Keep the expanded fixed check out of all future training batches."""
    path=HERE/'mixed-heldout-v1.jsonl'
    if not path.exists():return
    manifest=json.loads((HERE/'mixed-heldout-v1-manifest.json').read_text(encoding='utf-8'))
    if sha(path)!=manifest['sha256']:raise ValueError('Frozen mixed evaluation changed')
    frozen=ranker.read_records(path)
    inputs={r['input'] for r in frozen}
    surfaces={r['good'] for r in frozen}
    if any(r['split']=='train' and (r.get('input') in inputs or r['good'] in surfaces) for r in training):
        raise ValueError('Expanded mixed held-out example leaked into training')

def mixed_selection(training):
    path=HERE/'mixed-validation-v1.jsonl'
    if not path.exists():return [],None
    manifest=json.loads((HERE/'mixed-validation-v1-manifest.json').read_text(encoding='utf-8'))
    if sha(path)!=manifest['sha256']:raise ValueError('Frozen mixed validation changed')
    rows=ranker.read_records(path)
    if any(r['split']!='validation' for r in rows):raise ValueError('Mixed selection must be validation only')
    held=ranker.read_records(HERE/'mixed-heldout-v1.jsonl')
    keys=lambda rs:({r.get('input') for r in rs},{r['good'] for r in rs})
    inputs,surfaces=keys(rows)
    other_inputs,other_surfaces=keys(training+held)
    if inputs&other_inputs or surfaces&other_surfaces:raise ValueError('Mixed validation overlaps training or test')
    return rows,sha(path)

def is_mixed_japanese_english(row):
    """Classify the expected surface, never romanized input or a Japanese hint."""
    surface=row['good']
    # Curated historical lessons include French/Spanish and Korean. ASCII letters
    # alone do not establish English; exclude those known foreign spans.
    if row.get('language_pair') not in (None,'ja-en'):return False
    if re.search(r'[\uac00-\ud7af]|(?<![A-Za-zÀ-ÿ])(?:bonjour|merci|gracias|hola|café|mañana)(?![A-Za-zÀ-ÿ])',surface,re.IGNORECASE):return False
    return bool(re.search(r'[A-Za-z]',surface) and re.search(r'[ぁ-んァ-ヶ一-龯]',surface))

def select_training(rows,focus):
    if focus=='all':return list(rows)
    return [r for r in rows if r['split']!='train' or is_mixed_japanese_english(r)]

SKILLS=('case','english_preservation','japanese_context','japanese_inflection','boundaries','orthography')

def skill_for(row):
    if 'skill' in row:
        if row['skill'] not in SKILLS:raise ValueError('Unknown learning skill')
        return row['skill']
    good,bad=row['good'],row['bad']
    if good.casefold()==bad.casefold():return 'case'
    words=lambda s:re.findall(r'[A-Za-z_]+',s)
    if words(good)!=words(bad):return 'english_preservation'
    if any(token+particle in good and token+wrong in bad for token in words(good)
           for particle,wrong in [('で','出'),('を','尾'),('に','荷'),('から','殻')]):return 'boundaries'
    return 'japanese_context'

def assign_balanced_weights(rows):
    """Equal skill budgets and equal surface-frame budgets within each skill."""
    training=[r for r in rows if r['split']=='train']
    families=Counter();by_skill=defaultdict(set)
    for row in training:
        skill=skill_for(row)
        frame=row.get('balance_family') or re.sub(r'[A-Za-z_]+','<EN>',row['good'])
        row['balance_skill']=skill;row['balance_frame']=frame
        families[skill,frame]+=1;by_skill[skill].add(frame)
    for row in training:
        skill,frame=row['balance_skill'],row['balance_frame']
        row['training_weight']=1/(len(SKILLS)*len(by_skill[skill])*families[skill,frame])
    counts=Counter(r['balance_skill'] for r in training)
    totals={skill:sum(r['training_weight'] for r in training if r['balance_skill']==skill) for skill in SKILLS}
    return dict(policy='Equal six-skill budgets; equal frames within skill; equal rows within frame',
        skill_counts=dict(counts),skill_weight_totals=totals,missing_skills=[s for s in SKILLS if not counts[s]],
        frame_counts={s:len(by_skill[s]) for s in SKILLS},
        limits='Equal exposure is not proof of semantic diversity. Input text is not a feature of the current candidate ranker.')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--extra',type=Path,help='New curated training-only examples')
    parser.add_argument('--output',type=Path,help='Separate run artifact directory')
    parser.add_argument('--training-focus',choices=('all','mixed-ja-en'),help='Filter training only; keep every fixed evaluation unchanged')
    parser.add_argument('--anchor-strength',type=float,default=0.1,help='Warm-start L2 retention per epoch; 0 reproduces the previous update rule')
    parser.add_argument('--optimizer',choices=('sgd','adagrad'),default='sgd',help='Explicit offline optimizer experiment; fixed evaluation stays unchanged')
    args=parser.parse_args()
    started=time.monotonic()
    authored=ranker.read_records(HERE/'examples.jsonl')
    extra=ranker.read_records(args.extra) if args.extra else []
    if any(r['split']!='train' for r in extra):raise ValueError('Extra lessons must not modify validation or test sets')
    original=ranker.read_records(ROOT/'models/native-training.jsonl')
    frozen_inputs={r['input'] for r in authored+original if r['split'] in ('validation','test') and r.get('input')}
    assert not ({r['input'] for r in authored+extra if r['split']=='train'} & frozen_inputs)
    previous=None
    focus=args.training_focus or 'all'
    status_path=ROOT/'.cache/chatgpt-training/status.json'
    if status_path.exists():
        status=json.loads(status_path.read_text(encoding='utf-8'))
        focus=args.training_focus or status.get('training_focus','all')
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
    all_rows=original+authored+extra
    check_mixed_holdout(all_rows)
    selection,selection_sha=mixed_selection(all_rows) if focus=='mixed-ja-en' else ([],None)
    dataset=select_training(all_rows,focus)
    active_extra=[r for r in extra if focus=='all' or is_mixed_japanese_english(r)]
    counts={}
    for r in dataset:
        if r['split']=='train':
            r['topic']=r.get('topic',practice.record_topic(r))
            counts[r['topic']]=counts.get(r['topic'],0)+1
    balance=assign_balanced_weights(dataset) if focus=='mixed-ja-en' else None
    if balance is None:
        for r in dataset:
            if r['split']=='train':r['training_weight']=1/counts[r['topic']]
    output=args.output or ROOT/'.cache/chatgpt-training'/datetime.datetime.now(datetime.timezone.utc).strftime('session-%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True,exist_ok=True)
    path=output/'training.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in dataset),encoding='utf-8')
    dataset=ranker.read_records(path)
    initial_weights=previous if focus=='mixed-ja-en' else None
    fit_rows=dataset+selection
    weights,epoch=ranker.fit(fit_rows,initial_weights=initial_weights,anchor_strength=args.anchor_strength,optimizer=args.optimizer)
    binary,quantized=trainer.quantize(weights)
    model=output/'native-model.bin';model.write_bytes(binary)
    assert trainer.quantize(ranker.fit(fit_rows,initial_weights=initial_weights,anchor_strength=args.anchor_strength,optimizer=args.optimizer)[0])[0]==binary
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
        authored_examples=len(authored)+len(extra),extra_training_examples=len(active_extra),extra_training_evaluation=ranker.evaluate(quantized,active_extra),extra_sha256=sha(args.extra) if args.extra else None,teacher_weights_updated=False,installed=False,network_used=False,
        independently_reviewed=False,training_topic_counts=counts,selected_epoch=epoch,
        evaluation=evaluation,held_out_details=details,model_sha256=sha(model),model_bytes=len(binary),
        training_sha256=sha(path),references_sha256=sha(HERE/'examples.jsonl'),reproducible=True,
        runtime_seconds=round(time.monotonic()-started,2),
        limits='Candidate-pair scores with the reference supplied. Held-out sentences share lexical families. Not end-to-end IME accuracy, not training ChatGPT weights, not installed in current IME.')
    report['training_focus']=focus
    report['optimizer']=dict(name=args.optimizer,learning_rate=0.25,adagrad_epsilon=0.1 if args.optimizer=='adagrad' else None)
    report['runtime_margin_training_examples']=sum(bool(r.get('score_margin',0)) for r in dataset if r['split']=='train')
    report['warm_start_regularization']=dict(method='L2 to initial accepted weights, full vector per epoch',
        anchor_strength=args.anchor_strength if initial_weights is not None else 0.0,
        tuning_partition='validation only; no test gradients or hyperparameter selection',
        limits='Prior penalty reduces drift but does not guarantee retained accuracy; regression gates still required')
    report['mixed_selection_sha256']=selection_sha
    report['mixed_selection_examples']=len(selection)
    report['selection_in_training_weights']=False
    if selection:
        evaluation['mixed-validation']=dict(before=ranker.evaluate(baseline,selection),after=ranker.evaluate(quantized,selection))
        if previous is not None:evaluation['mixed-validation']['previous']=ranker.evaluate(previous,selection)
    report['training_balance']=balance
    report['initial_model_sha256']=previous_report['model_sha256'] if initial_weights is not None else None
    report['active_training_examples']=sum(r['split']=='train' for r in dataset)
    report['excluded_training_examples']=sum(r['split']=='train' for r in all_rows)-sum(r['split']=='train' for r in dataset)
    report['archived_extra_examples']=len(extra)
    mixed_test=[r for r in authored+original if r['split']=='test' and is_mixed_japanese_english(r)]
    report['mixed_fixed_test']=dict(before=ranker.evaluate(baseline,mixed_test),after=ranker.evaluate(quantized,mixed_test))
    if previous is not None:report['mixed_fixed_test']['previous']=ranker.evaluate(previous,mixed_test)
    original_test=evaluation['original-test']
    report['eligible_for_ime']=False
    report['original_references_sha256']=sha(ROOT/'models/native-training.jsonl')
    report['previous_model_sha256']=previous_report['model_sha256'] if previous is not None else None
    report['passes_candidate_regression_gate']=original_test['after']['wins']>=original_test['before']['wins'] and all(
        result['after']['wins']>=result.get('previous',result['before'])['wins']
        and result['after']['losses']<=result.get('previous',result['before'])['losses']
        for name,result in evaluation.items() if name in ('authored-validation','authored-held-out','original-test','mixed-validation'))
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
