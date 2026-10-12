# SPDX-License-Identifier: AGPL-3.0-only
[CmdletBinding(SupportsShouldProcess)]
param([Parameter(Mandatory)][string]$ArtifactDirectory)
$ErrorActionPreference = 'Stop'
$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not [Environment]::Is64BitProcess -or -not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Use administrator x64 PowerShell' }
$root = Join-Path $env:ProgramFiles 'NextIME'
$receipt = Join-Path $root 'installation.json'
$oldJson = Get-Content -LiteralPath $receipt -Raw
$old = $oldJson | ConvertFrom-Json
$oldDirectory = $root
if ($old.schema -eq 2 -and $old.directory -cmatch '^build-[0-9a-f]{16}$') { $oldDirectory = Join-Path $root $old.directory }
elseif ($old.schema -ne 1) { throw 'Invalid installation receipt' }
$names = @('nextime_tsf.dll','nextime_ffi.dll','LICENSE')
$allowedNames = $names + @('base.nud','MOZC-NOTICES.txt','mozc-source.json','TATOEBA-NOTICES.txt','TATOEBA-AUTHORS.tsv','context-training.json','LITERATURE-NOTICES.txt','literature-training.json','RUST-DEPENDENCIES-NOTICES.txt')
foreach ($name in $old.files.PSObject.Properties.Name) {
    if ($name -notin $allowedNames) { throw 'Unexpected receipt filename' }
    if ((Get-FileHash -LiteralPath (Join-Path $oldDirectory $name)).Hash -ne $old.files.$name) { throw "Modified installed file: $name" }
}
$oldDll = Join-Path $oldDirectory 'nextime_tsf.dll'
$classKey = 'Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Classes\CLSID\{97D8A216-4D8C-4AD5-9839-AF53EA9ADB41}\InprocServer32'
if ((Get-ItemProperty -LiteralPath $classKey).'(default)' -ne $oldDll) { throw 'Registered DLL does not match receipt' }
$source = (Resolve-Path -LiteralPath $ArtifactDirectory).Path
if (Test-Path -LiteralPath (Join-Path $source 'base.nud')) { $names += @('base.nud','MOZC-NOTICES.txt','mozc-source.json') }
$names += @('TATOEBA-NOTICES.txt','TATOEBA-AUTHORS.tsv','context-training.json')
$names += @('LITERATURE-NOTICES.txt','literature-training.json')
$names += @('RUST-DEPENDENCIES-NOTICES.txt')
$hashes = @{}
foreach ($name in $names) { $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $source $name)).Hash }
$hashMaterial = ($names | ForEach-Object { $hashes[$_] }) -join ''
$identity = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($hashMaterial))).ToLowerInvariant().Substring(0,16)
$relative = "build-$identity"
$directory = Join-Path $root $relative
$newDll = Join-Path $directory 'nextime_tsf.dll'
if ($oldDll -eq $newDll) { Write-Output 'This build is already active'; return }
if (-not $PSCmdlet.ShouldProcess($newDll,'Stage build and switch Luqnexa TSF registration')) { return }
if (-not (Test-Path -LiteralPath $directory)) { New-Item -ItemType Directory -Path $directory | Out-Null }
foreach ($name in $names) {
    $target = Join-Path $directory $name
    if (Test-Path -LiteralPath $target) {
        if ((Get-FileHash -LiteralPath $target).Hash -ne $hashes[$name]) { throw "Existing build differs: $name" }
    } else { Copy-Item -LiteralPath (Join-Path $source $name) -Destination $target }
    if ((Get-FileHash -LiteralPath $target).Hash -ne $hashes[$name]) { throw "Copy verification failed: $name" }
}
function Register-Dll([string]$path,[bool]$remove) {
    $arguments = if ($remove) { @('/s','/u',('"'+$path+'"')) } else { @('/s',('"'+$path+'"')) }
    $process = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32/regsvr32.exe') -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw "regsvr32 failed: $($process.ExitCode)" }
}
Register-Dll $oldDll $true
$newRegistered = $false
try {
    Register-Dll $newDll $false
    $newRegistered = $true
    if ((Get-ItemProperty -LiteralPath $classKey).'(default)' -ne $newDll) { throw 'New registration verification failed' }
    $oldJson | Set-Content -LiteralPath (Join-Path $root 'installation.previous.json') -Encoding utf8
    @{ schema=2; directory=$relative; files=$hashes } | ConvertTo-Json | Set-Content -LiteralPath "$receipt.new" -Encoding utf8
    Move-Item -LiteralPath "$receipt.new" -Destination $receipt -Force
} catch {
    $failure = $_
    if ($newRegistered) { Register-Dll $newDll $true }
    Register-Dll $oldDll $false
    $oldJson | Set-Content -LiteralPath $receipt -Encoding utf8
    throw $failure
}
Write-Output "Updated Luqnexa: $directory. Previous DLLs retained for apps that still use them. Restart target apps to load this build."
