# SPDX-License-Identifier: AGPL-3.0-only
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$config=Join-Path $env:LOCALAPPDATA 'Luqnexa/local-ai.json'
if (Test-Path -LiteralPath $config) {
    $settings=Get-Content -LiteralPath $config -Raw | ConvertFrom-Json
    $settings.enabled=$false
    $settings | ConvertTo-Json | Set-Content -LiteralPath $config -Encoding utf8NoBOM
}
$record=Join-Path $root '.cache/local-ai/running.json'
if (-not (Test-Path -LiteralPath $record)) { return }
$running=Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
$script=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'local-ai-server.py')).Path
$parent=Get-CimInstance Win32_Process -Filter "ProcessId=$($running.pid)"
if (-not $parent -or -not $parent.CommandLine -or -not $parent.CommandLine.Contains($script)) { return }
$manifest=Get-Content -LiteralPath (Join-Path $root '.cache/local-ai/manifest.json') -Raw | ConvertFrom-Json
$runtime=Get-CimInstance Win32_Process -Filter "ProcessId=$($running.runtime_pid)"
if ($runtime -and $runtime.ExecutablePath -eq $manifest.runtime -and $runtime.ParentProcessId -eq $parent.ProcessId) {
    Stop-Process -Id $runtime.ProcessId
}
Stop-Process -Id $parent.ProcessId
Write-Output 'Stopped the verified local AI helper. Reopen input apps to disable their loaded configuration.'
