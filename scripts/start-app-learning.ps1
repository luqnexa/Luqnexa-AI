# SPDX-License-Identifier: AGPL-3.0-only
param([switch]$RegisterStartup)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$worker=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'learn-while-app.py')).Path
$cache=Join-Path $root '.cache/app-learning'
New-Item -ItemType Directory -Path $cache -Force | Out-Null
$disabled=Join-Path $cache 'disabled'
if (Test-Path -LiteralPath $disabled) { Remove-Item -LiteralPath $disabled }
if ($RegisterStartup) {
    $shell=(Get-Command pwsh.exe).Source
    $entry='"'+$shell+'" -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "'+$PSCommandPath+'"'
    $key='HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
    New-Item -Path $key -Force | Out-Null
    New-ItemProperty -LiteralPath $key -Name 'LuqnexaAppLearning' -Value $entry -PropertyType String -Force | Out-Null
}
$record=Join-Path $cache 'running.json'
if (Test-Path -LiteralPath $record) {
    $state=Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
    $existing=Get-CimInstance Win32_Process -Filter "ProcessId=$($state.pid)"
    if ($existing -and $existing.CommandLine -and $existing.CommandLine.Contains($worker)) {
        Write-Output 'App learning watcher already running.'
        return
    }
}
$python=(Get-Command python).Source
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$process=Start-Process -FilePath $python -ArgumentList @('-u',('"'+$worker+'"')) -WindowStyle Hidden -WorkingDirectory $root -RedirectStandardOutput (Join-Path $cache "worker-$stamp.log") -RedirectStandardError (Join-Path $cache "worker-$stamp-error.log") -PassThru
$process.PriorityClass='BelowNormal'
Write-Output "App learning watcher started, PID $($process.Id)."
