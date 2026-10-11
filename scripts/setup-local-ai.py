# SPDX-License-Identifier: AGPL-3.0-only
"""Fetch pinned official local inference files, verifying SHA-256 before use."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1] / '.cache' / 'local-ai'
MODEL_SHA = '7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5'
RUNTIME_SHA = '927850725a5fb8a6b453891b324c4f2d4be17905f7c5b9a265f8f73daa3bd39d'
MODEL_URL = 'https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf'
RUNTIME_URL = 'https://github.com/ggml-org/llama.cpp/releases/download/b11554/llama-b11554-bin-win-cpu-x64.zip'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(url, path, sha):
    if path.exists() and digest(path) == sha:
        return
    partial = path.with_suffix(path.suffix + '.part')
    print('Downloading', path.name, flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'Luqnexa-local-AI/0.1'})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as output:
        count = 0
        checkpoint = 0
        while block := response.read(1024 * 1024):
            output.write(block)
            count += len(block)
            if count - checkpoint >= 128 * 1024 * 1024:
                print('Downloaded MiB:', count // (1024 * 1024), flush=True)
                checkpoint = count
    if digest(partial) != sha:
        raise ValueError('SHA-256 mismatch: ' + path.name)
    partial.replace(path)
    print('Verified', path.name, flush=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    model = ROOT / 'Qwen3-4B-Q4_K_M.gguf'
    runtime = ROOT / 'llama-b11554-bin-win-cpu-x64.zip'
    fetch(MODEL_URL, model, MODEL_SHA)
    fetch(RUNTIME_URL, runtime, RUNTIME_SHA)
    directory = ROOT / 'runtime-b11554'
    directory.mkdir(exist_ok=True)
    with zipfile.ZipFile(runtime) as archive:
        for item in archive.infolist():
            target = (directory / item.filename).resolve()
            if not target.is_relative_to(directory.resolve()):
                raise ValueError('Unsafe archive path')
        for item in archive.infolist():
            target = (directory/item.filename).resolve()
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            data = archive.read(item)
            if target.is_file() and digest(target) == hashlib.sha256(data).hexdigest():
                continue  # A running, identical DLL must not be overwritten.
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix+'.part')
            partial.write_bytes(data)
            partial.replace(target)
    license_url = 'https://huggingface.co/Qwen/Qwen3-4B-GGUF/raw/bc640142c66e1fdd12af0bd68f40445458f3869b/LICENSE'
    (ROOT / 'QWEN-LICENSE.txt').write_bytes(urllib.request.urlopen(license_url, timeout=30).read())
    (ROOT / 'LLAMA-LICENSE.txt').write_bytes(urllib.request.urlopen('https://raw.githubusercontent.com/ggml-org/llama.cpp/b11554/LICENSE', timeout=30).read())
    (ROOT / 'manifest.json').write_text(json.dumps({
        'model': str(model), 'model_sha256': MODEL_SHA, 'model_url': MODEL_URL,
        'model_license': 'Apache-2.0', 'runtime': str(next(directory.rglob('llama-server.exe'))),
        'runtime_archive_sha256': RUNTIME_SHA, 'runtime_url': RUNTIME_URL,
        'runtime_license': 'MIT',
    }, indent=2), encoding='utf-8')
    print('Local inference files ready', flush=True)


if __name__ == '__main__':
    main()
