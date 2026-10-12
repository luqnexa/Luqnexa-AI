# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded local learning batches while the current user's Codex process runs."""
import argparse
from contextlib import contextmanager
import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/app-learning'
spec = importlib.util.spec_from_file_location('distill', ROOT / 'scripts/distill-qwen.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding='utf-8')
    temporary.replace(path)


def app_running():
    if os.name != 'nt':
        return False
    shell = shutil.which('pwsh') or shutil.which('powershell')
    command = "$base=Join-Path $env:LOCALAPPDATA 'OpenAI/Codex'; " + \
        "@(Get-Process -Name Codex -ErrorAction SilentlyContinue | Where-Object { " + \
        "$_.Path -and $_.Path.StartsWith($base+'\\',[StringComparison]::OrdinalIgnoreCase) }).Count"
    try:
        result = subprocess.run([shell, '-NoProfile', '-Command', command], capture_output=True,
                                timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
        return result.returncode == 0 and int(result.stdout.strip()) > 0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


@contextmanager
def single_worker(path):
    with path.open('a+b') as stream:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def retain(seed, rows, maximum=96):
    groups = {r['group'] for r in seed}
    extra = [r for r in rows if r['group'] not in groups and r['split'] == 'train']
    return seed + extra[-maximum:]


def cycle(seed, rows):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = CACHE / 'runs' / stamp
    run.mkdir(parents=True)
    data = run / 'probes.jsonl'
    data.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    stop = run / 'stop'
    command = [sys.executable, str(ROOT / 'scripts/distill-qwen.py'), '--data', str(data),
               '--output', str(run), '--teacher-cache', str(ROOT / '.cache/qwen-continuous/teacher-labels.json'),
               '--romaji-practice', '--threads', '4', '--stop-file', str(stop)]
    with (run / 'training.log').open('wb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        atomic(CACHE / 'status.json', dict(state='training', pid=os.getpid(), child_pid=process.pid, run=str(run)))
        closing = False
        while process.poll() is None:
            time.sleep(5)
            if (CACHE / 'disabled').exists() or not app_running():
                closing = True
                stop.touch(exist_ok=True)
        if closing:
            return rows, dict(state='paused', run=str(run))
        if process.returncode:
            raise RuntimeError('Training failed; inspect ' + str(run / 'training.log'))
    trainer.verify(run, data)
    report = json.loads((run / 'report.json').read_text(encoding='utf-8'))
    rows = retain(seed, trainer.pairwise.read_records(data))
    atomic(CACHE / 'latest.json', dict(run=str(run), verified=True, report=report, installed=False))
    (CACHE / 'pairs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    return rows, dict(state='resting', run=str(run), generated=report['generated_problems'],
                     accepted=report['accepted'], rejected=report['rejected'], installed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=10, help='Rest between batches, seconds (minimum 10)')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    with single_worker(CACHE / 'worker.lock'):
        atomic(CACHE / 'running.json', dict(pid=os.getpid(), script=str(Path(__file__).resolve())))
        seed_path = CACHE / 'seed.jsonl'
        if not seed_path.exists():
            shutil.copyfile(ROOT / '.cache/qwen-continuous/pairs.jsonl', seed_path)
        seed = trainer.pairwise.read_records(seed_path)
        data = CACHE / 'pairs.jsonl'
        rows = trainer.pairwise.read_records(data) if data.exists() else seed
        while True:
            if (CACHE / 'disabled').exists():
                atomic(CACHE / 'status.json', dict(state='disabled', pid=os.getpid()))
                if args.once:
                    return
                time.sleep(15)
                continue
            if not app_running():
                atomic(CACHE / 'status.json', dict(state='waiting_for_codex', pid=os.getpid()))
                if args.once:
                    return
                time.sleep(15)
                continue
            try:
                rows, status = cycle(seed, rows)
                atomic(CACHE / 'status.json', dict(status, pid=os.getpid()))
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                atomic(CACHE / 'status.json', dict(state='error', pid=os.getpid(), error=str(error)))
                if args.once:
                    raise
                time.sleep(300)
                continue
            if args.once:
                return
            time.sleep(max(10, args.interval))


if __name__ == '__main__':
    main()
