# SPDX-License-Identifier: AGPL-3.0-only
$ErrorActionPreference = 'Stop'
$savedTemp = $env:TEMP
$savedTmp = $env:TMP
$savedPath = $env:PATH
Push-Location (Join-Path $PSScriptRoot '..')
try {
    if (-not [Environment]::Is64BitProcess) { throw 'Use x64 PowerShell' }
    cargo build -p nextime-ffi -p nextime-cli --release --locked
    if ($LASTEXITCODE -ne 0) { throw 'Rust build failed' }
    $metadata = cargo metadata --no-deps --format-version 1 | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'cargo metadata failed' }
    $output = Join-Path $metadata.target_directory 'windows-x64'
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    Copy-Item -LiteralPath (Join-Path $metadata.target_directory 'release/nextime_ffi.dll') -Destination $output
    Copy-Item -LiteralPath 'LICENSE' -Destination $output
    Copy-Item -LiteralPath 'models/TATOEBA-NOTICES.txt','models/TATOEBA-AUTHORS.tsv','models/context-training.json' -Destination $output
    Copy-Item -LiteralPath 'models/LITERATURE-NOTICES.txt','models/literature-training.json' -Destination $output
    Copy-Item -LiteralPath 'models/RUST-DEPENDENCIES-NOTICES.txt' -Destination $output
    if (Test-Path -LiteralPath 'dictionaries/mozc-ja.tsv') {
        $dictionaryHash = (Get-FileHash -LiteralPath 'dictionaries/mozc-ja.tsv').Hash.ToLowerInvariant()
        $cachedDictionary = Join-Path $output "base-$dictionaryHash.nud"
        if (-not (Test-Path -LiteralPath $cachedDictionary)) {
            & (Join-Path $metadata.target_directory 'release/nextime.exe') dict build dictionaries/mozc-ja.tsv $cachedDictionary 2026101102
            if ($LASTEXITCODE -ne 0) { throw 'Large Japanese dictionary build failed' }
        }
        Copy-Item -LiteralPath $cachedDictionary -Destination (Join-Path $output 'base.nud')
        Copy-Item -LiteralPath 'dictionaries/MOZC-NOTICES.txt','dictionaries/mozc-source.json' -Destination $output
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) { throw 'Visual Studio C++ Build Tools and Windows SDK are required' }
    Import-Module (Join-Path $vsPath 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
    $env:TEMP = $output
    $env:TMP = $output
    $objects = @()
    foreach ($source in @('service', 'registration', 'guids')) {
        $object = Join-Path $output "$source.obj"
        cl /nologo /c /std:c++17 /utf-8 /W4 /WX /EHsc /DUNICODE /D_UNICODE '/Icrates/nextime-ffi/include' "platforms/windows/src/$source.cpp" "/Fo$object"
        if ($LASTEXITCODE -ne 0) { throw "$source compilation failed" }
        $objects += $object
    }
    link /nologo /DLL $objects /DEF:platforms/windows/nextime.def "/OUT:$output/nextime_tsf.dll" ole32.lib oleaut32.lib uuid.lib user32.lib gdi32.lib advapi32.lib
    if ($LASTEXITCODE -ne 0) { throw 'TSF DLL link failed' }
    Write-Output "Windows x64 build: $output"
} finally {
    $env:TEMP = $savedTemp
    $env:TMP = $savedTmp
    $env:PATH = $savedPath
    Pop-Location
}
