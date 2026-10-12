# SPDX-License-Identifier: AGPL-3.0-only
"""Native FFI sentence regressions and warm conversion benchmark, no Qwen."""
import argparse
import ctypes as c
import json
import hashlib
import math
import os
from pathlib import Path
import time

CASES = [
    ('あいをあいようしている', 'AIを愛用している'),
    ('あいをたいせつにする', '愛を大切にする'),
    ('きかいをつかう', '機械を使う'),
    ('きかいにめぐまれた', '機会に恵まれた'),
    ('はしでごはんをたべる', '箸でご飯を食べる'),
    ('はしをわたる', '橋を渡る'),
    ('pythonでかくほうがいい', 'Pythonで書くほうがいい'),
    ('きょうはくもりです', '今日は曇りです'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', required=True, type=Path)
    parser.add_argument('--nud', required=True, type=Path)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results/native-ai.json'))
    parser.add_argument('--cases', type=Path, help='UTF-8 input/expected/category TSV')
    parser.add_argument('--allow-fail', action='store_true', help='Record baseline mismatches without aborting')
    args = parser.parse_args()
    cases = CASES if not args.cases else [tuple(line.split('\t')[:2]) for line in
        args.cases.read_text(encoding='utf-8').splitlines() if line and not line.startswith('#')]
    if not cases or any(len(row) != 2 for row in cases):
        parser.error('Cases must contain at least one input/expected pair')
    if not 20 <= args.iterations <= 10000:
        parser.error('iterations must be 20..10000')
    os.environ['LUQNEXA_AI_BACKEND'] = 'original'  # This test process only.
    ffi = c.CDLL(str(args.dll.resolve()))
    ffi.nextime_session_create.argtypes = [c.c_void_p, c.c_size_t, c.c_uint32, c.POINTER(c.c_uint64)]
    ffi.nextime_session_event.argtypes = [c.c_uint64, c.c_uint32, c.c_void_p, c.c_size_t, c.POINTER(c.c_uint32)]
    ffi.nextime_session_text.argtypes = [c.c_uint64, c.c_uint32, c.c_uint32, c.c_void_p, c.c_size_t, c.POINTER(c.c_size_t)]
    ffi.nextime_session_destroy.argtypes = [c.c_uint64]
    ffi.nextime_session_candidates.argtypes = [c.c_uint64, c.POINTER(c.c_uint32), c.POINTER(c.c_uint32)]
    handle = c.c_uint64()
    path = str(args.nud.resolve()).encode('utf-8')
    assert ffi.nextime_session_create(path, len(path), 8, c.byref(handle)) == 0

    def event(key, text=''):
        data = text.encode('utf-8')
        flags = c.c_uint32()
        assert ffi.nextime_session_event(handle, key, data or None, len(data), c.byref(flags)) == 0
        assert not flags.value & 2, 'must not auto commit'

    def text(kind, index=0):
        buffer = c.create_string_buffer(4096)
        size = c.c_size_t()
        assert ffi.nextime_session_text(handle, kind, index, buffer, len(buffer), c.byref(size)) == 0
        return buffer.value.decode('utf-8')

    try:
        assert text(9) == 'local:original'
        results = []
        for reading, expected in cases:
            for char in reading:
                event(1, char)
            event(10)
            actual = text(3)
            count, selected = c.c_uint32(), c.c_uint32()
            assert ffi.nextime_session_candidates(handle, c.byref(count), c.byref(selected)) == 0
            candidates = [text(4, index) for index in range(count.value)]
            results.append(dict(input=reading, expected=expected, actual=actual, passed=actual == expected,
                                candidates=candidates, expected_in_candidates=expected in candidates))
            if not args.allow_fail:
                assert actual == expected, (reading, actual, expected)
            event(7)
        samples = []
        for i in range(args.iterations + 20):
            event(7)
            reading = cases[i % len(cases)][0]
            start = time.perf_counter()
            event(1, reading)
            event(10)
            elapsed = (time.perf_counter() - start) * 1000
            if i >= 20:
                samples.append(elapsed)
        samples.sort()
        report = dict(cases_passed=sum(r['passed'] for r in results), cases_total=len(cases), results=results,
                      backend='original', dictionary=str(args.nud.resolve()),
                      dll_sha256=hashlib.sha256(args.dll.read_bytes()).hexdigest(),
                      measured_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                      model_scope='Built-in native model, not the staged Qwen-trained model',
                      iterations=args.iterations, warmup=20,
                      scope='Warm FFI Text+Space: parsing, lookup, predictions, candidate generation and ranking; no UI, Qwen or network; developer PC, not iPhone SE or 100M entries',
                      p50_ms=samples[math.ceil(len(samples)*.50)-1],
                      p95_ms=samples[math.ceil(len(samples)*.95)-1], max_ms=max(samples))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(args.output)
        print(json.dumps(report, ensure_ascii=True))
    finally:
        assert ffi.nextime_session_destroy(handle) == 0


if __name__ == '__main__':
    main()
