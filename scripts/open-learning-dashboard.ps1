# SPDX-License-Identifier: AGPL-3.0-only
param([switch]$NoOpen,[switch]$CreateShortcut)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$script=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'learning-dashboard.py')).Path
$cache=Join-Path $root '.cache/app-learning'
New-Item -ItemType Directory -Path $cache -Force | Out-Null
$record=Join-Path $cache 'dashboard.json'
$url=$null
if (Test-Path -LiteralPath $record) {
    $state=Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
    $existing=Get-CimInstance Win32_Process -Filter "ProcessId=$($state.pid)"
    if ($existing -and $existing.CommandLine -and $existing.CommandLine.Contains($script) -and $state.url -match '^http://127\.0\.0\.1:\d+/$') {
        try { $null=Invoke-RestMethod -Uri ($state.url+'api/status') -TimeoutSec 8; $url=$state.url } catch { }
    }
}
if (-not $url) {
    $python=(Get-Command python).Source
    $stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
    $process=Start-Process -FilePath $python -ArgumentList @('-u',('"'+$script+'"')) -WindowStyle Hidden -WorkingDirectory $root -RedirectStandardOutput (Join-Path $cache "dashboard-$stamp.log") -RedirectStandardError (Join-Path $cache "dashboard-$stamp-error.log") -PassThru
    for ($i=0;$i -lt 40;$i++) {
        Start-Sleep -Milliseconds 250
        if (Test-Path -LiteralPath $record) {
            $state=Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
            if ($state.pid -eq $process.Id) { $url=$state.url; break }
        }
    }
    if (-not $url) { throw 'Dashboard did not start. See .cache/app-learning/dashboard logs.' }
}
$edge=Join-Path ${env:ProgramFiles(x86)} 'Microsoft/Edge/Application/msedge.exe'
if ($CreateShortcut) {
    $desktop=[Environment]::GetFolderPath('Desktop')
    $shell=New-Object -ComObject WScript.Shell
    $link=$shell.CreateShortcut((Join-Path $desktop 'Luqnexa 学習状況.lnk'))
    $link.TargetPath=(Get-Command pwsh).Source
    $link.Arguments='-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "'+$PSCommandPath+'"'
    $link.WorkingDirectory=$root
    $link.Description='Luqnexaの学習状況を表示'
    if (Test-Path -LiteralPath $edge) { $link.IconLocation=$edge+',0' }
    $link.Save()
}
if (-not $NoOpen) {
    if (Test-Path -LiteralPath $edge) { Start-Process -FilePath $edge -ArgumentList @("--app=$url",'--new-window') }
    else { Start-Process $url }
}
Write-Output $url
