# SPDX-License-Identifier: AGPL-3.0-only
[CmdletBinding()]
param([ValidateSet('Install','Uninstall','Update')][string]$Action = 'Install')
$ErrorActionPreference = 'Stop'
$metadata = cargo metadata --manifest-path (Join-Path $PSScriptRoot '../Cargo.toml') --no-deps --format-version 1 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'cargo metadata failed' }
$artifact = Join-Path $metadata.target_directory 'windows-x64'
if (-not (Test-Path -LiteralPath $artifact)) { throw 'Run build-windows.ps1 first' }
$installer = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'install-windows.ps1')).Path
if ($Action -eq 'Update') { $installer = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'update-windows.ps1')).Path }
$log = Join-Path $artifact "elevated-$($Action.ToLowerInvariant()).log"
# PowerShell string literals, then UTF-16 encoded command: paths are never shell code.
$installerLiteral = "'" + $installer.Replace("'", "''") + "'"
$artifactLiteral = "'" + $artifact.Replace("'", "''") + "'"
$logLiteral = "'" + $log.Replace("'", "''") + "'"
$actionArgument = if ($Action -eq 'Update') { '' } else { "-Action '$Action'" }
$command = @"
`$ErrorActionPreference = 'Stop'
try {
    & $installerLiteral $actionArgument -ArtifactDirectory $artifactLiteral -Confirm:`$false *> $logLiteral
    exit 0
} catch {
    `$_ | Out-String | Add-Content -LiteralPath $logLiteral -Encoding utf8
    exit 1
}
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
$powershell = (Get-Command pwsh.exe).Source
Write-Output "Requesting Windows UAC elevation for Luqnexa $Action"
$process = Start-Process -FilePath $powershell -Verb RunAs -WindowStyle Hidden -ArgumentList @('-NoProfile', '-EncodedCommand', $encoded) -Wait -PassThru
if (Test-Path -LiteralPath $log) { Get-Content -LiteralPath $log }
if ($process.ExitCode -ne 0) { throw "Elevated $Action failed with code $($process.ExitCode). Log: $log" }
