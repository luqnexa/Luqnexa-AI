# SPDX-License-Identifier: AGPL-3.0-only
"""Reproducible public-corpus n-gram training. Never reads user input history."""
import argparse
import bz2
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences_detailed.tsv.bz2'
LICENSE = 'https://creativecommons.org/licenses/by/2.0/fr/'
BI = 0xFFFFFFFF


def split(text):
    # Identical texts always belong to the same partition, regardless of ID.
    bucket = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little') % 10
    return 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'


def rows(data):
    seen = set()
    for line in bz2.decompress(data).decode('utf-8').splitlines():
        fields = line.split('\t')
        if len(fields) != 6:
            # Export occasionally contains embedded line breaks; do not guess
            # attribution or merge malformed rows into training text.
            continue
        sid, lang, text, author, _, _ = fields
        # Only attributed Japanese sentences, bounded length, no control chars.
        if (lang != 'jpn' or author == '\\N' or not author or
                not 4 <= len(text) <= 160 or any(ord(c) < 32 for c in text)):
            continue
        if text in seen:
            continue
        seen.add(text)
        yield sid, text, author


def features(text):
    chars = [0, 0] + list(map(ord, text))
    for i in range(2, len(chars)):
        yield BI, chars[i-1], chars[i]
        yield chars[i-2], chars[i-1], chars[i]


def train(records, cap=200000):
    counts = Counter()
    for _, text, _ in records:
        counts.update(features(text))
    retained = sorted((key, count) for key, count in sorted(
        counts.items(), key=lambda item: (-item[1], item[0]))[:cap] if count >= 3)
    return counts, retained


def encode(retained):
    return b'NXAI0001' + b''.join(struct.pack('<IIII', *key, count) for key, count in retained)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Explicitly refresh official export')
    args = parser.parse_args()
    cache = ROOT / '.cache/tatoeba/jpn_sentences_detailed.tsv.bz2'
    if args.download:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            cache.write_bytes(response.read())
    data = cache.read_bytes()
    partitions = {name: [] for name in ('train', 'validation', 'test')}
    for record in rows(data):
        partitions[split(record[1])].append(record)
    counts, retained = train(partitions['train'])
    output = ROOT / 'models'
    model = encode(retained)
    (output / 'context-public.bin').write_bytes(model)
    # IDs and usernames attribute every accepted sentence, including held-out data.
    attribution = '\n'.join('\t'.join((sid, author, part)) for part, records in partitions.items()
                            for sid, _, author in records) + '\n'
    (output / 'TATOEBA-AUTHORS.tsv').write_text(attribution, encoding='utf-8')
    import math
    table = dict(retained)
    def score(text):
        fs = list(features(text))
        return sum(math.log1p(table.get(key, 0)) * (128 if key[0] == BI else 768)
                   for key in fs) / len(text)
    curated = Counter()
    for text in (ROOT / 'models/context-corpus.txt').read_text(encoding='utf-8').splitlines():
        if text and not text.startswith('#'):
            curated.update(features(text))
    def baseline(text):
        return sum(math.log1p(curated.get(key, 0)) * (128 if key[0] == BI else 768)
                   for key in features(text)) / len(text)
    synthetic_text = (ROOT / 'models/context-synthetic.txt').read_text(encoding='utf-8')
    synthetic_lines = [text for text in synthetic_text.splitlines() if text and not text.startswith('#')]
    augmented = curated.copy()
    for text in synthetic_lines:
        augmented.update(features(text))
    def augmented_score(text):
        return sum(math.log1p(augmented.get(key, 0)) * (128 if key[0] == BI else 768)
                   for key in features(text)) / len(text)
    # Held-out proxy only: intact sentence vs adjacent swap. Not IME accuracy.
    wins = ties = total = 0
    covered = opportunities = 0
    old_wins = old_ties = combined_wins = combined_ties = 0
    for _, text, _ in partitions['test']:
        fs = list(features(text))
        covered += sum(key in table for key in fs)
        opportunities += len(fs)
        pos = len(text) // 2
        if text[pos] == text[pos-1]:
            continue
        corrupt = text[:pos-1] + text[pos] + text[pos-1] + text[pos+1:]
        difference = score(text) - score(corrupt)
        wins += difference > 0
        ties += difference == 0
        total += 1
        old_difference = baseline(text) - baseline(corrupt)
        old_wins += old_difference > 0
        old_ties += old_difference == 0
        combined_difference = (int(augmented_score(text)) + int(min(score(text)/4, 1024)) -
                               int(augmented_score(corrupt)) - int(min(score(corrupt)/4, 1024)))
        combined_wins += combined_difference > 0
        combined_ties += combined_difference == 0
    report = dict(source=URL, license=LICENSE, source_sha256=hashlib.sha256(data).hexdigest(),
                  model_sha256=hashlib.sha256(model).hexdigest(), model_bytes=len(model),
                  partitions={key: len(value) for key, value in partitions.items()},
                  split='SHA256(text UTF-8), first four bytes little endian modulo 10: 0 test, 1 validation, else train',
                  raw_features=len(counts), retained_features=len(retained), min_count=3, cap=200000,
                  evaluation=dict(task='held-out intact vs middle adjacent-character swap; NOT IME conversion accuracy',
                                  pairs=total, wins=wins, ties=ties, losses=total-wins-ties,
                                  intact_win_rate=wins/total, feature_coverage=covered/opportunities),
                  previous_curated_model=dict(wins=old_wins, ties=old_ties, losses=total-old_wins-old_ties),
                  bundled_combined_model=dict(wins=combined_wins, ties=combined_ties,
                                              losses=total-combined_wins-combined_ties),
                  validation_used_for_tuning=False)
    report['synthetic'] = dict(source='ChatGPT-generated examples for NextIME, 2026-10-11',
                               sentences=len(synthetic_lines), sha256=hashlib.sha256(synthetic_text.encode()).hexdigest(),
                               license='AGPL-3.0-only', independent_evaluation=False)
    (output / 'context-training.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
