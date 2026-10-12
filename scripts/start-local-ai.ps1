# SPDX-License-Identifier: AGPL-3.0-only
param([switch]$RegisterStartup)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$script=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'local-ai-server.py')).Path
if ($RegisterStartup) {
    $starter=(Resolve-Path -LiteralPath $PSCommandPath).Path
    $pwsh=(Get-Command pwsh.exe).Source
    $runKey='HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
    if (-not (Test-Path -LiteralPath $runKey)) { New-Item -Path $runKey -Force | Out-Null }
    $command='"'+$pwsh+'" -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "'+$starter+'"'
    New-ItemProperty -LiteralPath $runKey -Name 'LuqnexaLocalAI' -Value $command -PropertyType String -Force | Out-Null
    Write-Output 'Local AI will start for this user at Windows sign-in.'
}
$cache=Join-Path $root '.cache/local-ai'
$record=Join-Path $cache 'running.json'
if (Test-Path -LiteralPath $record) {
    $running=Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
    $process=Get-CimInstance Win32_Process -Filter "ProcessId=$($running.pid)"
    if ($process -and $process.CommandLine -and $process.CommandLine.Contains($script)) {
        # Publish the current helper's settings for browser-hosted IMEs that
        # cannot see the file. Never log the authentication token.
        $config=Join-Path $env:LOCALAPPDATA 'Luqnexa/local-ai.json'
        if (Test-Path -LiteralPath $config) {
            $settings=Get-Content -LiteralPath $config -Raw | ConvertFrom-Json
            if ($settings.enabled -eq $true -and $settings.token -cmatch '^[0-9a-fA-F]{64}$') {
                $key='HKCU:\Software\Luqnexa\LocalAI'
                New-Item -Path $key -Force | Out-Null
                $mirror=@{enabled=$true;token=$settings.token;directory=$env:LOCALAPPDATA} | ConvertTo-Json -Compress
                New-ItemProperty -LiteralPath $key -Name 'Configuration' -PropertyType String -Value $mirror -Force | Out-Null
            }
        }
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
