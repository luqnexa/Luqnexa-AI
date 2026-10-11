# SPDX-License-Identifier: AGPL-3.0-only
"""Rights-allowlisted literary corpus training; no user input and no bulk crawl."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/literature'
INDEX = 'https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip'
AOZORA = ['000789','000773','000752','000128','000042','000170','000092',
          '000301','001567','000456','001927','000623','000682','000691','000389']
STANDARD = [
    ('jane-austen_pride-and-prejudice', 'Jane Austen', 1817, 1813, 'train'),
    ('jane-austen_sense-and-sensibility', 'Jane Austen', 1817, 1811, 'train'),
    ('jane-austen_emma', 'Jane Austen', 1817, 1815, 'train'),
    ('charles-dickens_a-christmas-carol', 'Charles Dickens', 1870, 1843, 'train'),
    ('charles-dickens_great-expectations', 'Charles Dickens', 1870, 1861, 'validation'),
    ('lewis-carroll_alices-adventures-in-wonderland', 'Lewis Carroll', 1898, 1865, 'test'),
]
spec = importlib.util.spec_from_file_location('trainer', ROOT/'scripts/train-context.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, name, download):
    path = CACHE/name
    if not path.exists():
        if not download:
            raise ValueError(f'Missing cache {path}; use --download')
        request = urllib.request.Request(url, headers={'User-Agent': 'NextIME-corpus-research/0.1'})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(32*1024*1024+1)
        if len(data) > 32*1024*1024:
            raise ValueError('Oversized source')
        path.write_bytes(data)
    return path.read_bytes()


def allowed_aozora(records):
    # Require every contributor row, not only the author's row, to be clear.
    return bool(records) and all(r['作品著作権フラグ'] == 'なし'
                                and r['人物著作権フラグ'] == 'なし'
                                and r['役割フラグ'] == '著者' for r in records)


def clean_aozora(text):
    # Separate the title/notation guide and colophon; retain originals in cache.
    blocks = re.split(r'(?m)^-{20,}\s*$', text, maxsplit=2)
    if len(blocks) != 3:
        raise ValueError('Missing Aozora notation guide delimiter')
    body = blocks[2].split('底本：', 1)[0]
    body = re.sub(r'［＃[^］]*］', '', body)
    body = re.sub(r'《[^》]*》', '', body).replace('｜', '')
    return body


def clean_xhtml(data):
    root = ET.fromstring(data)
    body = root.find('{http://www.w3.org/1999/xhtml}body')
    if body is None:
        raise ValueError('No XHTML body')
    # Paragraphs only: no colophon, headings, artwork, navigation or footnotes.
    return '\n'.join(''.join(p.itertext()) for p in body.iter('{http://www.w3.org/1999/xhtml}p'))


def standard_source(repo, download):
    path = CACHE/f'standard-{repo}.zip'
    if path.exists():
        return path.read_bytes()
    # Fetch only text and licensing when no archive is cached: some full
    # repositories contain tens of MB of artwork that training does not use.
    base = f'https://raw.githubusercontent.com/standardebooks/{repo}/master/'
    files = {}
    for name in ('LICENSE.md','src/epub/content.opf'):
        files[name] = fetch(base+name, f'raw-{repo}-{name.replace("/","-")}', download)
    root = ET.fromstring(files['src/epub/content.opf'])
    for item in root.iter('{http://www.idpf.org/2007/opf}item'):
        href = item.get('href','')
        if re.fullmatch(r'text/[a-zA-Z0-9_-]+\.xhtml', href):
            name = 'src/epub/'+href
            files[name] = fetch(base+name, f'raw-{repo}-{href.replace("/","-")}', download)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            info = zipfile.ZipInfo(repo+'-master/'+name)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    path.write_bytes(buffer.getvalue())
    return buffer.getvalue()


def sentences(body, language):
    body = re.sub(r'[\t\r\n\u3000 ]+', ' ', body).strip()
    # English abbreviations can split: documented heuristic, not a linguistic parser.
    parts = re.split(r'(?<=[。！？])' if language == 'ja' else r'(?<=[.!?])\s+', body)
    for text in parts:
        text = text.strip()
        if not any(ord(c) < 32 for c in text):
            for start in range(0,len(text),160):
                chunk = text[start:start+160].strip()
                if len(chunk)>=4:
                    yield chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Fetch missing allowlisted sources only')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    index_data = fetch(INDEX, 'aozora-index.zip', args.download)
    z = zipfile.ZipFile(io.BytesIO(index_data))
    catalog = list(csv.DictReader(io.StringIO(z.read(z.namelist()[0]).decode('utf-8-sig'))))
    works, texts = [], []
    for sid in AOZORA:
        records = [r for r in catalog if r['作品ID'] == sid]
        if not allowed_aozora(records):
            raise ValueError(f'Rights check failed for Aozora {sid}')
        row = records[0]
        url = row['テキストファイルURL']
        if not url.startswith('https://www.aozora.gr.jp/cards/') or not url.endswith('.zip'):
            raise ValueError('Unexpected Aozora source URL')
        data = fetch(url, f'aozora-{sid}.zip', args.download)
        archive = zipfile.ZipFile(io.BytesIO(data))
        names = [n for n in archive.namelist() if n.lower().endswith('.txt')]
        if len(names) != 1 or archive.getinfo(names[0]).file_size > 16*1024*1024:
            raise ValueError('Ambiguous or oversized Aozora text')
        original = archive.read(names[0]).decode('cp932')
        body = clean_aozora(original)
        part = 'test' if sid in ('000389','000773') else 'validation' if sid in ('000691','000301') else 'train'
        extracted = list(sentences(body, 'ja'))
        texts.append((part, 'ja', extracted))
        works.append(dict(source='Aozora Bunko', id=sid, language='ja', partition=part,
                          title=row['作品名'], author=row['姓']+row['名'],
                          url=row['図書カードURL'], download=url, sha256=sha(data),
                          copyright_work=row['作品著作権フラグ'], copyright_person=row['人物著作権フラグ'],
                          author_death=row['没年月日'], input_by=row['入力者'], proofread_by=row['校正者'],
                          original_header=original.split('-'*20,1)[0],
                          original_colophon=original[original.rfind('底本：'):] if '底本：' in original else '',
                          rules='https://www.aozora.gr.jp/guide/kijyunn.html', extracted_sentences=len(extracted)))
        print(f'Aozora {sid}: {len(extracted)} sentences', flush=True)
    for repo, author, death, published, part in STANDARD:
        # Original English works only; all selected authors died before 1900,
        # and the underlying editions were published before 1900. No translations.
        if death > 1900 or published > 1900:
            raise ValueError('Outside reviewed rights boundary')
        url = f'https://codeload.github.com/standardebooks/{repo}/zip/refs/heads/master'
        data = standard_source(repo, args.download)
        archive = zipfile.ZipFile(io.BytesIO(data))
        license_name = next(n for n in archive.namelist() if n.endswith('/LICENSE.md'))
        license_text = archive.read(license_name).decode('utf-8')
        if 'CC0 1.0 Universal Public Domain Dedication' not in license_text:
            raise ValueError('Missing CC0 contributor dedication')
        opf = ET.fromstring(archive.read(next(n for n in archive.namelist() if n.endswith('/content.opf'))))
        creators = [e.text for e in opf.iter('{http://purl.org/dc/elements/1.1/}creator')]
        if creators != [author] or any(e.text == 'trl' for e in opf.iter('{http://www.idpf.org/2007/opf}meta')):
            raise ValueError('Unexpected author or translation')
        title = next(opf.iter('{http://purl.org/dc/elements/1.1/}title')).text
        names = sorted(n for n in archive.namelist() if '/src/epub/text/' in n and n.endswith('.xhtml')
                       and n.rsplit('/',1)[-1] not in ('colophon.xhtml','titlepage.xhtml','imprint.xhtml','endnotes.xhtml','halftitle.xhtml'))
        extracted = [text for n in names for text in sentences(clean_xhtml(archive.read(n)), 'en')]
        texts.append((part, 'en', extracted))
        works.append(dict(source='Standard Ebooks', id=repo, title=title, author=author,
                          author_death=death, original_publication=published, language='en', partition=part,
                          url=f'https://github.com/standardebooks/{repo}', download=url, sha256=sha(data),
                          retrieval='Cached repository ZIP or deterministic bundle of official raw OPF, license and text manifest files; artwork is not used.',
                          original_license=license_text, extracted_sentences=len(extracted),
                          rights_review='Original English work; author died before 1900; original publication before 1900; editorial contributions CC0. Reviewed for Japan, not a worldwide legal guarantee.'))
        print(f'Standard Ebooks {repo}: {len(extracted)} sentences', flush=True)
    # Book-level split. Remove repeated sentences globally, including cross-partition duplicates.
    unique = set()
    partitions = {p: [] for p in ('train','validation','test')}
    languages = Counter()
    for part, lang, records in sorted(texts, key=lambda item: ('test','validation','train').index(item[0])):
        for text in records:
            if text not in unique:
                unique.add(text)
                partitions[part].append(text)
                languages[(part,lang)] += 1
    counts = Counter(key for text in partitions['train'] for key in trainer.features(text))
    retained = sorted((key,count) for key,count in sorted(counts.items(), key=lambda item:(-item[1],item[0]))[:200000] if count>=3)
    model = trainer.encode(retained)
    # Limited held-out intact-vs-swapped proxy. Do not call this IME accuracy.
    import math
    table = dict(retained)
    def score(text):
        return sum(math.log1p(table.get(k,0))*(128 if k[0]==trainer.BI else 768) for k in trainer.features(text))/len(text)
    wins = ties = total = 0
    for text in partitions['test']:
        pos = len(text)//2
        if text[pos] == text[pos-1]: continue
        bad = text[:pos-1]+text[pos]+text[pos-1]+text[pos+1:]
        difference=score(text)-score(bad)
        wins += difference>0; ties += difference==0; total += 1
    output = ROOT/'models'
    (output/'context-literature.bin').write_bytes(model)
    report = dict(index_sha256=sha(index_data), works=works, total_books=len(works),
                  partitions={p:len(v) for p,v in partitions.items()},
                  languages={f'{p}/{lang}':count for (p,lang),count in languages.items()},
                  model_bytes=len(model), model_sha256=sha(model), retained_features=len(retained),
                  raw_features=len(counts), min_count=3, cap=200000,
                  preprocessing='Remove notation, ruby, headings, colophons; normalize whitespace; sentence heuristic; split long sentences into <=160-character fragments; reject fragments shorter than 4; deduplicate; book-level split. No modernization or factual correction.',
                  evaluation=dict(task='Held-out literary intact vs adjacent swap proxy, NOT IME conversion accuracy',
                                  pairs=total,wins=wins,ties=ties,losses=total-wins-ties),
                  validation_used_for_tuning=False)
    (output/'literature-training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='works'},ensure_ascii=True,indent=2))


if __name__ == '__main__':
    main()
