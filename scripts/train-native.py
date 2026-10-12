# SPDX-License-Identifier: AGPL-3.0-only
"""Train the original Luqnexa candidate classifier, without pretrained weights."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pairwise', Path(__file__).with_name('train-contrast.py'))
pairwise = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pairwise)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify reproducible committed artifacts without writing')
    args = parser.parse_args()
    data = ROOT / 'models/native-training.jsonl'
    rows = pairwise.read_records(data)
    weights, epoch = pairwise.fit(rows)
    quantized = [max(-32768, min(32767, round(w * 1024))) for w in weights]
    model = b'NXCR0001' + struct.pack('<4096h', *quantized)
    weights = [w / 1024 for w in quantized]
    # Whole collocations, rather than ambiguous single kanji such as 愛/AI.
    # Unknown uses of a homophone must retain the existing context rules.
    anchors = ''.join(a + '\t' + b + '\n' for a, b in sorted({
        (r['good'], r['bad']) for r in rows if r['split'] == 'train'}))
    if args.check:
        assert (ROOT / 'models/native-model.bin').read_bytes() == model
        assert (ROOT / 'models/native-anchors.tsv').read_text(encoding='utf-8') == anchors
    else:
        (ROOT / 'models/native-model.bin').write_bytes(model)
        (ROOT / 'models/native-anchors.tsv').write_text(anchors, encoding='utf-8')
    report = dict(algorithm='Pairwise logistic SGD, hashed character/context features, i16 weights',
                  pretrained_weights=False, source='Authored synthetic sentence pairs',
                  license='AGPL-3.0-only', selected_epoch=epoch, model_bytes=len(model),
                  data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(model).hexdigest(),
                  selection='Validation only; held-out test examples never update weights',
                  runtime='Bounded lexical domains; current sentence and optional previous context; no generation or network',
                  evaluation={part: pairwise.evaluate(weights, [r for r in rows if r['split'] == part])
                              for part in ('train', 'validation', 'test')})
    if args.check:
        assert json.loads((ROOT / 'models/native-training.json').read_text(encoding='utf-8')) == report
    else:
        (ROOT / 'models/native-training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
