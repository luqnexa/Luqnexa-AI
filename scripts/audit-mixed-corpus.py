# SPDX-License-Identifier: AGPL-3.0-only
"""Audit an existing attributed public corpus offline; never train or download."""
import argparse
import bz2
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def select(source, excluded, limit=24):
    counts = Counter()
    used_words = Counter()
    used_authors = Counter()
    seen = set()
    pool = []
    with bz2.open(source, 'rt', encoding='utf-8') as corpus:
        for line in corpus:
            counts['lines'] += 1
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 6:
                counts['malformed'] += 1
                continue
            sid, language, text, author, _, _ = fields
            if language != 'jpn' or not author or author == '\\N':
                counts['unattributed_or_other_language'] += 1
                continue
            if not 8 <= len(text) <= 96 or any(ord(c) < 32 for c in text):
                counts['text_bounds'] += 1
                continue
            if text in seen:
                counts['duplicate'] += 1
                continue
            seen.add(text)
            words = re.findall(r'[A-Za-z]{2,}', text)
            japanese = sum('\u3040' <= c <= '\u9fff' for c in text)
            if not words or japanese < len(text) / 2:
                continue
            counts['mixed_candidates'] += 1
            bucket = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little') % 10
            if bucket in (0, 1):
                counts['legacy_heldout'] += 1
                continue
            if text in excluded:
                counts['existing_reference'] += 1
                continue
            pool.append(dict(source_id=sid, author=author, text=text,
                    source_url=f'https://tatoeba.org/en/sentences/show/{sid}',
                    license='CC BY 2.0 FR', license_url='https://creativecommons.org/licenses/by/2.0/fr/',
                    latin_words=words, source_partition='train', reviewed=False,
                    reading_verified=False, eligible_for_training=False))
    # Stable hash ordering avoids taking only the oldest IDs/first author.
    pool.sort(key=lambda row: hashlib.sha256(('mixed-review-v1\0' + row['text']).encode()).digest())
    selected = []
    for row in pool:
        key = row['latin_words'][0].lower()
        if used_words[key] >= 2 or used_authors[row['author']] >= 3:
            continue
        selected.append(row)
        used_words[key] += 1
        used_authors[row['author']] += 1
        if len(selected) == limit:
            break
    counts['eligible_pool'] = len(pool)
    counts['selected_authors'] = len(used_authors)
    counts['selected_latin_groups'] = len(used_words)
    return selected, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--exclude', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.exclude.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    excluded = {r[key] for r in rows for key in ('good', 'bad')}
    selected, counts = select(args.source, excluded)
    result = dict(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        selected=selected, counts=counts, training=False, network=False,
        limits='Script/length/diversity filters only. Volunteer sentences require meaning and reading review; not validated IME references.')
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps(dict(counts=counts, selected=len(selected))))


if __name__ == '__main__':
    main()
