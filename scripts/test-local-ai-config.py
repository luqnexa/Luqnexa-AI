# SPDX-License-Identifier: AGPL-3.0-only
"""Test Windows FFI configuration with a process-isolated HKCU and synthetic token."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
import winreg


def child(dll, key_name):
    from ctypes import wintypes
    api = ctypes.WinDLL('advapi32')
    api.RegOverridePredefKey.argtypes = [wintypes.HKEY, wintypes.HKEY]
    root = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_name, 0, winreg.KEY_ALL_ACCESS)
    assert api.RegOverridePredefKey(wintypes.HKEY(-2147483647), int(root)) == 0
    try:
        directory = Path(os.environ['LOCALAPPDATA'])
        key = r'Software\Luqnexa\LocalAI'
        mirror = {'enabled': True, 'token': '0' * 64, 'directory': str(directory)}

        def publish(settings):
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key) as handle:
                winreg.SetValueEx(handle, 'Configuration', 0, winreg.REG_SZ, json.dumps(settings))

        publish(mirror)
        ffi = ctypes.CDLL(dll)
        ffi.nextime_session_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                             ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
        ffi.nextime_session_text.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32,
                                           ctypes.c_void_p, ctypes.c_size_t,
                                           ctypes.POINTER(ctypes.c_size_t)]
        ffi.nextime_session_destroy.argtypes = [ctypes.c_uint64]

        def check(expected):
            session = ctypes.c_uint64()
            assert ffi.nextime_session_create(None, 0, 8, ctypes.byref(session)) == 0
            try:
                output = ctypes.create_string_buffer(256)
                size = ctypes.c_size_t()
                assert ffi.nextime_session_text(session, 9, 0, output, 256, ctypes.byref(size)) == 0
                assert output.value.decode() == expected, (output.value, expected)
            finally:
                ffi.nextime_session_destroy(session)

        check('local:on')  # Missing file, matching registry mirror.
        config = directory / 'Luqnexa/local-ai.json'
        config.parent.mkdir()
        config.write_text('{"enabled":false}', encoding='utf-8')
        check('local:config-disabled')  # File disable must override enabled mirror.
        config.write_text('malformed', encoding='utf-8')
        check('local:config-invalid')  # Never hide a malformed file with a mirror.
        config.unlink()
        publish({**mirror, 'directory': 'another-profile'})
        check('local:config-missing')
        publish({**mirror, 'enabled': False})
        check('local:config-disabled')
        publish({**mirror, 'token': 'short'})
        check('local:token-invalid')
        kernel = ctypes.WinDLL('kernel32')
        kernel.GetTempPathW.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
        temporary = ctypes.create_unicode_buffer(32768)
        assert kernel.GetTempPathW(len(temporary), temporary) > 0
        diagnostic = Path(temporary.value) / f'luqnexa-ai-config-{os.getpid()}.json'
        if diagnostic.exists():
            data = json.loads(diagnostic.read_text(encoding='utf-8'))
            assert set(data) == {'directory', 'stage', 'os_error', 'pid'}
            assert data['directory'] == str(directory)
            assert '0' * 64 not in diagnostic.read_text(encoding='utf-8')
            diagnostic.unlink()
        print('Windows local AI configuration: 6 checks passed (isolated HKCU/file)')
    finally:
        api.RegOverridePredefKey(wintypes.HKEY(-2147483647), None)
        root.Close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dll', required=True)
    parser.add_argument('--child-key')
    args = parser.parse_args()
    if args.child_key:
        child(args.dll, args.child_key)
        return
    key = r'Software\Luqnexa\Tests\config-' + uuid.uuid4().hex
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key):
        pass
    try:
        with tempfile.TemporaryDirectory(prefix='luqnexa-config-') as directory:
            subprocess.run([sys.executable, __file__, '--dll', str(Path(args.dll).resolve()),
                            '--child-key', key], env={**os.environ, 'LOCALAPPDATA': directory,
                                                    'LUQNEXA_AI_BACKEND': 'qwen'},
                           check=True)
    finally:
        # Only this newly created unique test key, deepest known children first.
        for suffix in [r'\Software\Luqnexa\LocalAI', r'\Software\Luqnexa', r'\Software', '']:
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key + suffix)
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    main()
