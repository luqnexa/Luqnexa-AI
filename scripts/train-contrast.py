# SPDX-License-Identifier: AGPL-3.0-only
"""Local pairwise ranking training on authored examples, never input history."""
import argparse
from collections import Counter
import hashlib
import json
import math
import re
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
DIM = 4096


def slot(text):
    value = 2166136261
    for byte in text.encode('utf-8'):
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return value % DIM


def features(context, text):
    context = context[-24:]
    text = text[:96]
    counts = Counter()
    for size in (1, 2, 3):
        for start in range(len(text) - size + 1):
            counts[slot('s:' + text[start:start + size])] += 1
    # Cross features allow the same homophone to win in different contexts.
    for left in sorted({context[i:i+2] for i in range(len(context)-1)}):
        for right in sorted({text[i:i+2] for i in range(len(text)-1)}):
            counts[slot('c:' + left + ':' + right)] += 1
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1
    return {key: value / norm for key, value in counts.items()}


def score(weights, context, text):
    return sum(weights[key] * value for key, value in features(context, text).items())


def evaluate(weights, records):
    wins = ties = 0
    for row in records:
        delta = score(weights, row['context'], row['good']) - score(weights, row['context'], row['bad'])
        wins += delta > 1e-9
        ties += abs(delta) <= 1e-9
    return dict(pairs=len(records), wins=wins, ties=ties, losses=len(records)-wins-ties)


def anchors(rows):
    pairs = set()
    for row in rows:
        if row['split'] != 'train':
            continue
        good, bad = row['good'], row['bad']
        good_words, bad_words = re.findall('[A-Za-z]+', good), re.findall('[A-Za-z]+', bad)
        if len(good_words) == len(bad_words) and any(a != b for a,b in zip(good_words,bad_words)):
            pairs.update((a,b) for a,b in zip(good_words,bad_words) if a != b)
            continue
        while good and bad and good[0] == bad[0]:
            good, bad = good[1:], bad[1:]
        while good and bad and good[-1] == bad[-1]:
            good, bad = good[:-1], bad[:-1]
        if good and bad:
            pairs.add((good, bad))
    return sorted(pairs)


def read_records(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    seen = set()
    groups = {}
    for row in rows:
        if row['split'] not in ('train', 'validation', 'test', 'regression'):
            raise ValueError('Unknown partition')
        if any(not isinstance(row[field], str) or len(row[field]) > 160 or
               any(ord(c) < 32 for c in row[field]) for field in ('context', 'good', 'bad')):
            raise ValueError('Invalid or oversized example')
        identity = (row['context'], row['good'], row['bad'])
        if identity in seen or row['good'] == row['bad']:
            raise ValueError('Duplicate or non-contrastive example')
        seen.add(identity)
        if row['group'] in groups and groups[row['group']] != row['split']:
            raise ValueError('Group leakage across partitions')
        groups[row['group']] = row['split']
    return rows


def fit(rows, epochs=180):
    training = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    if not training or not validation:
        raise ValueError('Training and validation partitions are required')
    differences = []
    for row in training:
        sample_weight=row.get('training_weight',1.0)
        if not isinstance(sample_weight,(int,float)) or not math.isfinite(sample_weight) or sample_weight<=0:
            raise ValueError('Invalid training weight')
        good = features(row['context'], row['good'])
        bad = features(row['context'], row['bad'])
        differences.append((sample_weight,{key: good.get(key, 0)-bad.get(key, 0) for key in good.keys() | bad.keys()}))
    weights = [0.0] * DIM
    best = weights[:]
    best_key = (-1, float('-inf'))
    selected_epoch = 0
    for epoch in range(1, epochs + 1):
        for sample_weight,difference in differences:
            margin = sum(weights[key] * value for key, value in difference.items())
            gradient = sample_weight * 0.25 / (1 + math.exp(min(margin, 40)))
            for key, value in difference.items():
                weights[key] += gradient * value - sample_weight * 0.0001 * weights[key]
        result = evaluate(weights, validation)
        loss = sum(math.log1p(math.exp(-max(-40, min(40,
                   score(weights, r['context'], r['good'])-score(weights, r['context'], r['bad'])))))
                   for r in validation)
        key = (result['wins'], -loss)
        if key > best_key:
            best_key, best, selected_epoch = key, weights[:], epoch
    return best, selected_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'models/context-contrast.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/contrast-training')
    args = parser.parse_args()
    rows = read_records(args.data)
    weights, epoch = fit(rows)
    # Explicit little-endian fixed point; exactly 8 KiB plus header.
    quantized = [max(-32768, min(32767, round(value * 1024))) for value in weights]
    model = b'NXCR0001' + struct.pack('<4096h', *quantized)
    weights = [value/1024 for value in quantized]
    report = dict(source='Authored synthetic contrast examples, 2026-10-12',
                  license='AGPL-3.0-only', data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(model).hexdigest(), model_bytes=len(model),
                  dimension=DIM, epochs=180, selected_epoch=epoch,
                  algorithm='Pairwise logistic SGD; FNV-1a hashed character and context-cross features',
                  learning_rate=0.25, l2_per_update=0.0001, shuffle=False,
                  normalization='L2 norm of hashed feature counts; Unicode code-point bounds: context last 24, candidate first 96',
                  runtime='Requires preceding context and both trained-domain alternatives; only matching candidates gain 0..3072 points; no candidate generation',
                  selection='Validation only; test and reported regression examples never update weights',
                  input_history_used=False, independent_real_world_evaluation=False,
                  evaluation={part: evaluate(weights, [r for r in rows if r['split'] == part])
                              for part in ('train', 'validation', 'test', 'regression')})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'context-contrast.bin').write_bytes(model)
    (args.output/'contrast-anchors.tsv').write_text(''.join(a+'\t'+b+'\n' for a,b in anchors(rows)), encoding='utf-8')
    (args.output/'contrast-evaluation.tsv').write_text(''.join(
        '\t'.join([r['split'], r['group'], r['context'], r['good'], r['bad']])+'\n'
        for r in rows if r['split'] in ('test', 'regression')), encoding='utf-8')
    (args.output/'contrast-training.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True))


if __name__ == '__main__':
    main()
