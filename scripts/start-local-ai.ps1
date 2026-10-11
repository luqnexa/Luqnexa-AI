# SPDX-License-Identifier: AGPL-3.0-only
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$script=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'local-ai-server.py')).Path
$cache=Join-Path $root '.cache/local-ai'
$record=Join-Path $cache 'running.json'
if (Test-Path -LiteralPath $record) {
    $running=Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
    $process=Get-CimInstance Win32_Process -Filter "ProcessId=$($running.pid)"
    if ($process -and $process.CommandLine -and $process.CommandLine.Contains($script)) {
        Write-Output 'Local AI is already running.'
        return
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $cache 'manifest.json'))) {
    throw 'Run python scripts/setup-local-ai.py first.'
}
$python=(Get-Command python).Source
$process=Start-Process -FilePath $python -ArgumentList @('-u', ('"'+$script+'"')) -WindowStyle Hidden -WorkingDirectory $root -RedirectStandardOutput (Join-Path $cache 'server-current.log') -RedirectStandardError (Join-Path $cache 'server-current-error.log') -PassThru
Write-Output "Starting local AI, PID $($process.Id). See .cache/local-ai/server-current.log for readiness."
