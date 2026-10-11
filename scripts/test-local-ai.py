# SPDX-License-Identifier: AGPL-3.0-only
"""Real local LLM evaluation on synthetic examples. No real conversation data."""
import json
import argparse
import os
from pathlib import Path
import statistics
import time
import urllib.request

CASES = [
    ('人工知能のチャットサービスを利用します。', 'chatgpt', ['chatgpt', 'ChatGPT', 'CHATGPT'], 1),
    ('ソースコードを書くエディターの話です。', 'vscode', ['vscode', 'VSCode', 'VSCODE'], 1),
    ('ソースコードを共有するサービスです。', 'github', ['github', 'GITHUB', 'GitHub'], 2),
    ('Appleが販売するスマートフォンの製品名です。', 'iphone', ['iphone', 'IPHONE', 'iPhone'], 2),
    ('メモリ安全性を重視するプログラミング言語です。', 'rust', ['rust', 'Rust', 'RUST'], 1),
    ('Continue the English sentence: I ate an', 'apple', ['Apple', 'APPLE', 'apple'], 2),
    ('She began a new sentence with a greeting: “', 'hello', ['hello', 'HELLO', 'Hello'], 2),
    ('The exact variable name is chatgpt. Preserve its lowercase spelling.', 'chatgpt', ['ChatGPT', 'CHATGPT', 'chatgpt'], 2),
    ('夕食に麺を食べます。手に持つ道具です。', 'はし', ['橋', '箸', '端'], 1),
    ('対岸へ渡るための建造物です。', 'はし', ['箸', '橋', '端'], 1),
    ('テーブルの一番外側です。', 'はし', ['橋', '箸', '端'], 2),
    ('空から水滴が降ってきました。', 'あめ', ['飴', '雨'], 1),
    ('口の中で甘いお菓子をなめます。', 'あめ', ['雨', '飴'], 1),
    ('美容院で短く切ってもらいます。', 'かみ', ['紙', '神', '髪'], 2),
    ('印刷機に入れる白いシートです。', 'かみ', ['神', '髪', '紙'], 2),
    ('人工知能を使って変換候補を選びます。', 'あい', ['愛', '藍', 'AI'], 2),
    ('川の話は終わりです。これから麺を食べる道具の話です。', 'はし', ['橋', '箸', '端'], 1),
    ('食事の話は終わりです。これから対岸に渡る建造物の話です。', 'はし', ['箸', '橋', '端'], 1),
    ('ChatGPTや機械学習など人工知能の話をしています。', 'あいによる', ['愛による', 'AIによる'], 1),
    ('ChatGPTなどの人工知能を活用して文章を書きたいです。', 'あいをつかい', ['愛を使い', 'AIを使い'], 1),
    ('VSCodeで文章を書いています。ファイルの内容を記すことです。', 'VSCodeでかく', ['VSCodeで描く', 'VSCodeで書く'], 1),
    ('ChatGPTにテキストの下書きを作ってもらいました。文章を記すことです。', 'ChatGPTでかく', ['ChatGPTで書く', 'ChatGPTで描く'], 0),
    ('GitHubでソースコードを共有しています。修正した版を届けます。', 'GitHubでこうしん', ['GitHubで行進', 'GitHubで更新'], 1),
    ('Pythonでプログラムを書いています。動かして結果を確認します。', 'Pythonをじっこう', ['Pythonを実行', 'Pythonを実効'], 0),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Report path; defaults to benchmarks/results/local-llm.json')
    parser.add_argument('--only-casing', action='store_true', help='Evaluate case alternatives only')
    args = parser.parse_args()
    config = json.loads((Path(os.environ['LOCALAPPDATA'])/'Luqnexa/local-ai.json').read_text())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(path, payload):
        request = urllib.request.Request('http://127.0.0.1:43821'+path, json.dumps(payload).encode(),
            {'Content-Type': 'application/json', 'Authorization': 'Bearer '+config['token']})
        with opener.open(request, timeout=3) as response:
            return json.load(response)

    results = []
    for context, reading, candidates, expected in CASES:
        if args.only_casing and len({c.lower() for c in candidates}) != 1:
            continue
        started = time.monotonic()
        job = call('/submit', {'context': context, 'reading': reading, 'candidates': candidates})['job']
        while True:
            result = call('/poll', {'job': job})
            if result['ready']:
                break
            if time.monotonic()-started > 30:
                raise TimeoutError('Local inference timed out')
            time.sleep(.1)
        row = {'context': context, 'reading': reading, 'candidates': candidates,
               'expected_index': expected, 'actual_index': result.get('index'),
               'passed': result.get('index') == expected,
               'latency_ms': round((time.monotonic()-started)*1000, 2)}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'.cache/local-ai/manifest.json').read_text())
    report = {'scope': f'{len(results)} synthetic cases, not general accuracy; first inference may be cold',
              'model_sha256': manifest['model_sha256'], 'cases': results,
              'passed': sum(x['passed'] for x in results), 'total': len(results),
              'median_ms': statistics.median(x['latency_ms'] for x in results)}
    output = args.output or root/'benchmarks/results/local-llm.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Local LLM evaluation: {report['passed']}/{report['total']}", flush=True)
    return 0 if all(row['passed'] for row in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
