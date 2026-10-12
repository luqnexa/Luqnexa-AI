# SPDX-License-Identifier: AGPL-3.0-only
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$cache=Join-Path $root '.cache/app-learning'
New-Item -ItemType Directory -Path $cache -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $cache 'disabled') -Force | Out-Null
$key='HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
Remove-ItemProperty -LiteralPath $key -Name 'LuqnexaAppLearning' -ErrorAction SilentlyContinue
Write-Output 'Learning disabled; the current teacher request will finish before cleanup. User startup entry removed.'
