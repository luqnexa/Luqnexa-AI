# SPDX-License-Identifier: AGPL-3.0-only
"""Loopback learning dashboard and local practice answers. No external assets."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/app-learning'
PAGE = ROOT / 'platforms/learning-dashboard/index.html'
LOCK = threading.Lock()
ANSWER_LOCK = threading.Lock()
ALIVE = (0, None, False)
IMPORT_SPEC=importlib.util.spec_from_file_location('lesson_import',ROOT/'scripts/import-chatgpt-lessons.py')
LESSON_IMPORT=importlib.util.module_from_spec(IMPORT_SPEC);IMPORT_SPEC.loader.exec_module(LESSON_IMPORT)


def read_json(path, default=None):
    try:
        if path.stat().st_size > 1024 * 1024:
            return default
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def safe_run(value, cache=CACHE):
    try:
        path = Path(value).resolve()
        path.relative_to((cache / 'runs').resolve())
        return path
    except (TypeError, ValueError, OSError):
        return None


def worker_alive(record):
    global ALIVE
    pid = record.get('pid')
    if type(pid) is not int or pid <= 0 or os.name != 'nt':
        return False
    with LOCK:
        stamp, previous_pid, result = ALIVE
        if previous_pid == pid and time.monotonic()-stamp < 10:
            return result
        # Verify PID, executable and script; never expose a command line to the UI.
        expected = str((ROOT / 'scripts/learn-while-app.py').resolve()).replace("'", "''")
        command = f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; " + \
            f"[bool]($p -and $p.ExecutablePath -and ([IO.Path]::GetFileName($p.ExecutablePath) -eq 'python.exe') " + \
            f"-and $p.CommandLine -and $p.CommandLine.Contains('{expected}'))"
        try:
            shell = shutil.which('pwsh') or shutil.which('powershell')
            reply = subprocess.run([shell, '-NoProfile', '-Command', command], capture_output=True,
                                   timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
            result = reply.returncode == 0 and reply.stdout.strip() == b'True'
        except (OSError, TypeError, subprocess.TimeoutExpired):
            result = False
        ALIVE = (time.monotonic(), pid, result)
        return result


def snapshot(cache=CACHE, alive=None):
    status = read_json(cache / 'status.json', {})
    record = read_json(cache / 'running.json', {})
    latest = read_json(cache / 'latest.json', {})
    status = status if isinstance(status, dict) else {}
    record = record if isinstance(record, dict) else {}
    latest = latest if isinstance(latest, dict) else {}
    running = worker_alive(record) if alive is None else alive
    state = status.get('state', 'not_started')
    if not running:
        state = 'stopped' if record else 'not_started'
    if (cache / 'disabled').exists():
        state = 'disabled'
    run = safe_run(status.get('run'), cache)
    audit = read_json(run / 'teacher-audit.json', []) if run else []
    audit = audit if isinstance(audit, list) else []
    progress = None
    if run and state == 'training':
        try:
            with (run / 'training.log').open('rb') as stream:
                stream.seek(max(0, (run / 'training.log').stat().st_size-16384))
                lines = stream.read(16384).decode('utf-8', errors='replace').splitlines()
            for line in reversed(lines):
                try:
                    value = json.loads(line)
                    if isinstance(value, dict) and all(type(value.get(k)) is int for k in ('processed','total','accepted')):
                        progress = {k:value[k] for k in ('processed','total','accepted')}
                        break
                except ValueError:
                    pass
        except OSError:
            pass
    report = latest.get('report', {})
    report = report if isinstance(report, dict) else {}
    current_report = read_json(run / 'report.json', {}) if run else {}
    if state == 'training':
        phase = 'validation' if current_report or (progress and progress['processed'] >= progress['total']) else \
            ('ranking' if progress or audit else 'generation')
    else:
        phase = state
    active_question = read_json(run / 'current-question.json', {}) if run else {}
    if state == 'training' and isinstance(active_question, dict) and active_question.get('phase') == 'romaji':
        phase = 'ranking'
    examples = []
    questions = []
    current = read_json(run / 'current-question.json', {}) if run else {}
    if not isinstance(current, dict):
        current = {}
    if state != 'training':
        current = {}
    if run:
        try:
            path = run / 'probes.jsonl'
            if path.stat().st_size <= 1024 * 1024:
                results = {a['group']:a for a in audit if isinstance(a,dict) and isinstance(a.get('group'),str)}
                for line in path.read_text(encoding='utf-8').splitlines():
                    row = json.loads(line)
                    if row.get('split') != 'train' or not all(isinstance(row.get(k),str) for k in ('group','context','good','bad')):
                        continue
                    result = results.get(row['group'], {})
                    questions.append(dict(**{k:row[k][:160] for k in ('group','context','good','bad')},
                        status='error' if result.get('error_type') else ('accepted' if result.get('accepted') else
                            ('rejected' if result else 'pending')), choices=result.get('choices'),
                        reused=result.get('reused_teacher_labels') is True))
        except (OSError,ValueError,TypeError,AttributeError):
            pass
    proposals = read_json(run / 'generated-proposals.json', {}) if run else {}
    if isinstance(proposals, dict) and isinstance(proposals.get('proposed'), list):
        for item in proposals['proposed'][:4]:
            if isinstance(item, dict) and all(isinstance(item.get(k),str) for k in ('context','good','bad')):
                examples.append({k:item[k][:160] for k in ('context','good','bad')})
    history = []
    for folder in sorted((cache / 'runs').glob('*'), reverse=True)[:24]:
        item = read_json(folder / 'report.json', {})
        if not isinstance(item, dict) or not item:
            continue
        score = item.get('evaluation', {}).get('new-held-out-test', {}).get('after', {})
        history.append(dict(id=folder.name, accepted=item.get('accepted'), generated=item.get('generated_problems'),
                            seconds=item.get('runtime_seconds'), wins=score.get('wins'), pairs=score.get('pairs')))
        if len(history) == 10:
            break
    model_run = safe_run(latest.get('run'), cache)
    integrity = None
    if model_run and isinstance(report.get('model_sha256'), str):
        try:
            model = model_run / 'native-model.bin'
            if model.stat().st_size <= 1024*1024:
                integrity = hashlib.sha256(model.read_bytes()).hexdigest() == report['model_sha256']
        except OSError:
            integrity = False
    practice = read_json(cache / 'practice-results.json', {})
    if isinstance(practice, dict) and practice.get('results'):
        questions = [dict(group=r['id'],context=r['context'],good=r['expected'],bad=r['actual'],
                          status='accepted' if r.get('learned') else 'correct' if r.get('status')=='correct' else 'error' if r.get('status')=='teacher_error' else 'rejected' if practice.get('complete') else 'pending',choices=r.get('choices'),reused=False)
                     for r in practice['results']]
    return dict(manual_training=read_json(ROOT/'.cache/manual-training/status.json',{}),state=state, phase=phase, running=running, progress=progress, examples=examples,
                questions=questions[:160], current_question=current,
                romaji=read_json(cache / 'romaji-results.json', {}),
                practice=practice,
                user_answers=read_json(cache / 'practice-answers.json', {}),
                report={k:report.get(k) for k in ('method','teacher','accepted','rejected','proposed','generated_problems',
                        'reused_teacher_labels','runtime_seconds','model_bytes','evaluation')},
                verified=latest.get('verified') is True, integrity=integrity, installed=latest.get('installed') is True,
                history=history, error='学習ログにエラーがあります。' if status.get('error') else None,
                updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        host = self.headers.get('Host')
        if host not in allowed or self.headers.get('Origin') != 'http://' + host:
            self.send_error(403)
            return
        if self.path == '/api/lessons':
            self.import_lessons()
            return
        if self.path != '/api/answer':
            self.send_error(404)
            return
        if self.headers.get('Content-Type') != 'application/json':
            self.send_error(415)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 1 <= length <= 2048:
                raise ValueError('Invalid size')
            value = json.loads(self.rfile.read(length))
            answer = value.get('answer')
            if not isinstance(answer, str) or not 1 <= len(answer) <= 96 or any(ord(c) < 32 for c in answer):
                raise ValueError('Invalid answer')
            with ANSWER_LOCK:
                practice = read_json(CACHE / 'practice-results.json', {})
                if value.get('quiz_id') != practice.get('quiz_id') or value.get('id') not in {
                        r.get('id') for r in practice.get('results', [])}:
                    self.send_error(409)
                    return
                answers = read_json(CACHE / 'practice-answers.json', {})
                answers[value['id']] = dict(answer=answer, quiz_id=value['quiz_id'])
                answers = dict(list(answers.items())[-96:])
                temporary = CACHE / 'practice-answers.tmp'
                temporary.write_text(json.dumps(answers, ensure_ascii=False), encoding='utf-8')
                for attempt in range(10):
                    try:
                        temporary.replace(CACHE / 'practice-answers.json')
                        break
                    except PermissionError:
                        time.sleep(.02 * (attempt + 1))
                else:
                    raise OSError('Cannot save answer')
        except (ValueError, TypeError, AttributeError):
            self.send_error(400)
            return
        except OSError:
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', '11')
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def import_lessons(self):
        code=200
        try:
            if self.headers.get('Content-Type')!='application/json':raise ValueError('JSONを送信してください')
            length=int(self.headers.get('Content-Length','0'))
            if not 1<=length<=65536:raise ValueError('リクエストが大きすぎます')
            value=json.loads(self.rfile.read(length))
            if not isinstance(value,dict) or set(value)!={'text'}:raise ValueError('教材テキストだけを送信してください')
            body=LESSON_IMPORT.start_import(value['text'])
        except (ValueError,TypeError,KeyError) as error:code=400;body=dict(ok=False,error=str(error))
        except RuntimeError as error:code=409;body=dict(ok=False,error=str(error))
        except OSError:code=503;body=dict(ok=False,error='教材を保存できませんでした')
        data=json.dumps(body,ensure_ascii=False).encode()
        self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
        self.end_headers();self.wfile.write(data)

    def do_GET(self):
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in allowed:
            self.send_error(403)
            return
        if self.path == '/':
            data, mime = PAGE.read_bytes(), 'text/html; charset=utf-8'
        elif self.path == '/api/status':
            data, mime = json.dumps(snapshot(), ensure_ascii=False).encode(), 'application/json; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'none'; frame-ancestors 'none'")
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=0)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.daemon_threads=True
    CACHE.mkdir(parents=True,exist_ok=True)
    (CACHE/'dashboard.json').write_text(json.dumps(dict(pid=os.getpid(),url=f'http://127.0.0.1:{server.server_port}/')),encoding='utf-8')
    print(f'http://127.0.0.1:{server.server_port}/',flush=True)
    server.serve_forever()


if __name__=='__main__':
    main()
