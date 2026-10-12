# SPDX-License-Identifier: AGPL-3.0-only
[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateSet('Install','Uninstall')][string]$Action = 'Install',
    [string]$ArtifactDirectory
)
$ErrorActionPreference = 'Stop'
if (-not [Environment]::Is64BitProcess) { throw 'Use x64 PowerShell; only x64 apps are supported' }
$destination = Join-Path $env:ProgramFiles 'NextIME'
$dll = Join-Path $destination 'nextime_tsf.dll'
$receipt = Join-Path $destination 'installation.json'
$names = @('nextime_tsf.dll', 'nextime_ffi.dll', 'LICENSE')
if (-not $PSCmdlet.ShouldProcess($destination, "$Action Luqnexa x64 TSF profile and DLLs")) { return }
$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Windows TSF machine registration requires administrator PowerShell; no changes were made'
}
function Invoke-Registrar([string]$path, [bool]$remove) {
    $arguments = if ($remove) { @('/s', '/u', ('"' + $path + '"')) } else { @('/s', ('"' + $path + '"')) }
    $process = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32/regsvr32.exe') -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw "regsvr32 failed with code $($process.ExitCode); retain files for diagnosis" }
}
if ($Action -eq 'Install') {
    if (Test-Path -LiteralPath $destination) { throw 'Luqnexa directory already exists; uninstall/review it before installing' }
    if ($ArtifactDirectory) {
        $source = (Resolve-Path -LiteralPath $ArtifactDirectory).Path
    } else {
        $metadata = cargo metadata --manifest-path (Join-Path $PSScriptRoot '../Cargo.toml') --no-deps --format-version 1 | ConvertFrom-Json
        if ($LASTEXITCODE -ne 0) { throw 'cargo metadata failed' }
        $source = Join-Path $metadata.target_directory 'windows-x64'
    }
    foreach ($name in $names) { if (-not (Test-Path -LiteralPath (Join-Path $source $name))) { throw 'Run scripts/build-windows.ps1 first' } }
    if (Test-Path -LiteralPath (Join-Path $source 'base.nud')) { $names += @('base.nud','MOZC-NOTICES.txt','mozc-source.json') }
    $names += @('TATOEBA-NOTICES.txt','TATOEBA-AUTHORS.tsv','context-training.json')
    $names += @('LITERATURE-NOTICES.txt','literature-training.json')
    $names += @('RUST-DEPENDENCIES-NOTICES.txt')
    foreach ($name in $names) { if (-not (Test-Path -LiteralPath (Join-Path $source $name))) { throw "Missing artifact: $name" } }
    # Existing registration must never be replaced by a copy of a different build.
    if (Test-Path -LiteralPath 'Registry::HKEY_CLASSES_ROOT\CLSID\{97D8A216-4D8C-4AD5-9839-AF53EA9ADB41}') { throw 'Luqnexa CLSID already registered' }
    New-Item -ItemType Directory -Path $destination | Out-Null
    foreach ($name in $names) { Copy-Item -LiteralPath (Join-Path $source $name) -Destination (Join-Path $destination $name) }
    $hashes = @{}
    foreach ($name in $names) { $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $destination $name) -Algorithm SHA256).Hash }
    # Write receipt before registration so a failed registration remains recoverable.
    @{ schema = 1; files = $hashes } | ConvertTo-Json | Set-Content -LiteralPath $receipt -Encoding utf8
    Invoke-Registrar $dll $false
    Write-Output 'Registered Luqnexa Japanese (prototype). Select it explicitly in Windows language settings. The default IME was not changed.'
} else {
    if (-not (Test-Path -LiteralPath $receipt)) { throw 'Installation receipt missing; refuse to delete untracked files' }
    $installed = Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json
    $allowedNames = $names + @('base.nud','MOZC-NOTICES.txt','mozc-source.json','TATOEBA-NOTICES.txt','TATOEBA-AUTHORS.tsv','context-training.json','LITERATURE-NOTICES.txt','literature-training.json','RUST-DEPENDENCIES-NOTICES.txt')
    $names = @($installed.files.PSObject.Properties.Name)
    foreach ($name in $names) { if ($name -notin $allowedNames) { throw 'Unexpected receipt filename' } }
    $payload = $destination
    if ($installed.schema -eq 2 -and $installed.directory -cmatch '^build-[0-9a-f]{16}$') { $payload = Join-Path $destination $installed.directory }
    elseif ($installed.schema -ne 1) { throw 'Unsupported receipt' }
    $dll = Join-Path $payload 'nextime_tsf.dll'
    foreach ($name in $names) {
        if ((Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash -ne $installed.files.$name) { throw "Modified installation file: $name" }
    }
    Invoke-Registrar $dll $true
    # Delete only these exact, receipt-verified files. Preserve dictionaries and other files.
    foreach ($name in $names) { Remove-Item -LiteralPath (Join-Path $payload $name) }
    if ($payload -ne $destination -and @(Get-ChildItem -LiteralPath $payload -Force).Count -eq 0) { Remove-Item -LiteralPath $payload }
    Remove-Item -LiteralPath $receipt
    if (@(Get-ChildItem -LiteralPath $destination -Force).Count -eq 0) { Remove-Item -LiteralPath $destination }
    Write-Output 'Luqnexa profile unregistered and installation files removed. Restart apps if they still have the DLL loaded.'
}
