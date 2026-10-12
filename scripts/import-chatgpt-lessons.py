# SPDX-License-Identifier: AGPL-3.0-only
"""Import bounded data-only ChatGPT lessons and train a staged local model."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.cache/manual-training'
IMPORT_LOCK=threading.Lock()

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

practice=module('import_practice','romaji-practice.py')
trainer=module('import_trainer','distill-qwen.py')

def atomic(path,value):practice.atomic(path,value)

def parse_lessons(text,converter=None):
    if not isinstance(text,str) or not 1<=len(text.encode('utf-8'))<=32768:raise ValueError('教材は32KB以内にしてください')
    text=text.strip()
    if text.startswith('```'):
        lines=text.splitlines()
        if len(lines)<3 or lines[-1].strip()!='```':raise ValueError('JSONコードブロックが閉じていません')
        text='\n'.join(lines[1:-1])
    try:items=json.loads(text)
    except ValueError as error:raise ValueError('JSON形式で貼り付けてください') from error
    if isinstance(items,dict) and set(items)=={'lessons'}:items=items['lessons']
    if not isinstance(items,list) or not 6<=len(items)<=24:raise ValueError('6〜24問の配列にしてください（6分野を同数）')
    fixed=trainer.pairwise.read_records(ROOT/'models/chatgpt-training/examples.jsonl')
    original=trainer.pairwise.read_records(ROOT/'models/native-training.jsonl')
    protected={r['input'] for r in fixed if r['split']!='train'}|{r[0] for r in practice.HELD_OUT}
    fixed_cases=ROOT/'tests/romaji-symbols.tsv'
    if fixed_cases.exists():protected.update(line.split('\t')[0] for line in fixed_cases.read_text(encoding='utf-8').splitlines() if line and not line.startswith('#'))
    seen_pairs={(r['context'],r['good'],r['bad']) for r in fixed+original}
    prior_inputs=set()
    for path in (CACHE/'runs').glob('*/lessons.jsonl'):
        for prior in trainer.pairwise.read_records(path):
            seen_pairs.add((prior['context'],prior['good'],prior['bad']))
            prior_inputs.add(prior['input'])
    seen_inputs=set();rows=[];counts={key:0 for key in practice.TOPICS}
    if converter is None:
        config=json.loads((ROOT/'.cache/app-learning/romaji-config.json').read_text(encoding='utf-8-sig'))
        converter=practice.Converter(config)
    for n,item in enumerate(items,1):
        if not isinstance(item,dict) or set(item)!={'topic','input','reading','context','expected','incorrect'}:
            raise ValueError(f'{n}問目: 指定された6項目だけを使用してください')
        if item['topic'] not in counts:raise ValueError(f'{n}問目: 分野が不正です')
        for key in ('input','reading','context','expected','incorrect'):
            value=item[key]
            if not isinstance(value,str) or not 1<=len(value)<=96 or any(ord(c)<32 for c in value):raise ValueError(f'{n}問目: {key}は1〜96文字の文字列にしてください')
        if not re.fullmatch('[A-Za-z ]+',item['input']):raise ValueError(f'{n}問目: inputはローマ字にしてください')
        if item['input'] in protected:raise ValueError(f'{n}問目: 評価専用問題は学習に使えません')
        identity=(item['context'],item['expected'],item['incorrect'])
        if item['input'] in seen_inputs or item['input'] in prior_inputs or identity in seen_pairs:raise ValueError(f'{n}問目: 重複教材です')
        if item['expected']==item['incorrect']:raise ValueError(f'{n}問目: 正答と誤答が同じです')
        if item['expected']==item['reading'] or re.fullmatch('[ぁ-んー ]+',item['expected']):raise ValueError(f'{n}問目: 読みのコピーだけの教材は採用しません')
        if not trainer.valid_candidate_casing(item['expected'],item['context']):raise ValueError(f'{n}問目: 正答の大文字小文字を確認してください')
        converted=converter.convert(dict(input=item['input'],context=item['context']))
        actual_reading=converted.get('reading','')
        # A trailing roman n is pending in preedit; conversion finalizes it to ん.
        if actual_reading.endswith('n') and re.search('[ぁ-ん]',actual_reading):actual_reading=actual_reading[:-1]+'ん'
        if actual_reading!=item['reading']:raise ValueError(f'{n}問目: ローマ字の実際の読み「{actual_reading}」とreadingが不一致です')
        seen_inputs.add(item['input']);seen_pairs.add(identity);counts[item['topic']]+=1
        rows.append(dict(group='paste-'+hashlib.sha256(json.dumps(item,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),split='train',
            input=item['input'],reading=item['reading'],context=item['context'],good=item['expected'],bad=item['incorrect'],topic=item['topic'],
            provenance='User-pasted ChatGPT synthetic lessons; meaning not independently verified',native_actual=converted['actual']))
    if len(set(counts.values()))!=1:raise ValueError('6分野を同じ問題数にしてください')
    return rows

def start_import(text):
    if not IMPORT_LOCK.acquire(blocking=False):raise RuntimeError('別の貼り付け教材を処理中です')
    try:
        rows=parse_lessons(text)
        CACHE.mkdir(parents=True,exist_ok=True)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run=CACHE/'runs'/stamp;run.mkdir(parents=True)
        data=run/'lessons.jsonl'
        data.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        previous=[]
        for p in sorted((CACHE/'runs').glob('*/lessons.jsonl')):
            if p==data:continue
            previous+=trainer.pairwise.read_records(p)
        previous+=trainer.pairwise.read_records(ROOT/'models/chatgpt-training/round-20261012-01.jsonl')
        identities=set();combined=[]
        for r in previous+rows:
            key=(r['context'],r['good'],r['bad'])
            if key not in identities:identities.add(key);combined.append(r)
        if len(combined)>240:raise ValueError('累積教材が240問を超えました。点検が必要です')
        cumulative=run/'cumulative.jsonl';cumulative.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in combined),encoding='utf-8')
        state=dict(state='training',run_id=stamp,questions=len(rows),total_lessons=len(combined),rows=rows,installed=False,
            message='端末内で学習・評価中。貼り付けた意味の正しさは独立検証されていません。')
        atomic(CACHE/'status.json',state)
        threading.Thread(target=train_job,args=(run,cumulative,state),daemon=True).start()
        return dict(ok=True,run_id=stamp,questions=len(rows))
    except BaseException:
        IMPORT_LOCK.release();raise

def train_job(run,data,state):
    try:
        output=run/'model'
        with (run/'training.log').open('wb') as log:
            result=subprocess.run([sys.executable,str(ROOT/'models/chatgpt-training/run-training.py'),'--extra',str(data),'--output',str(output)],
                stdout=log,stderr=log,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        if result.returncode:raise ValueError('学習に失敗しました。教材・重複・評価データの隔離を確認してください')
        report=json.loads((output/'report.json').read_text(encoding='utf-8'))
        if hashlib.sha256((output/'native-model.bin').read_bytes()).hexdigest()!=report['model_sha256'] or not report['reproducible']:raise ValueError('モデル検証に失敗しました')
        state.update(state='complete',report=report,verified=True,message='学習・候補比較・再現性検証が完了しました。IMEには未適用です。')
        atomic(CACHE/'status.json',state)
    except (OSError,ValueError,KeyError,subprocess.TimeoutExpired) as error:
        state.update(state='error',message=str(error));atomic(CACHE/'status.json',state)
    finally:IMPORT_LOCK.release()
