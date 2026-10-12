# SPDX-License-Identifier: AGPL-3.0-only
param([switch]$SkipBuild, [switch]$Registered)
$ErrorActionPreference = 'Stop'
if (-not $SkipBuild) { & (Join-Path $PSScriptRoot 'build-windows.ps1') }
$metadata = cargo metadata --manifest-path (Join-Path $PSScriptRoot '../Cargo.toml') --no-deps --format-version 1 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'cargo metadata failed' }
$output = Join-Path $metadata.target_directory 'windows-x64'
python (Join-Path $PSScriptRoot 'test-local-ai-config.py') --dll (Join-Path $output 'nextime_ffi.dll')
if ($LASTEXITCODE -ne 0) { throw 'Local AI configuration integration failed' }
$savedTemp = $env:TEMP
$savedTmp = $env:TMP
$savedPath = $env:PATH
Push-Location (Join-Path $PSScriptRoot '..')
try {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    Import-Module (Join-Path $vsPath 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
    $env:TEMP = $output
    $env:TMP = $output
    cl /nologo /std:c++17 /utf-8 /W4 /WX /EHsc /DUNICODE /D_UNICODE '/Icrates/nextime-ffi/include' '/Iplatforms/windows/src' platforms/windows/tests/tsf.cpp "/Fo$output/tsf-test.obj" "/Fe$output/tsf-test.exe" /link ole32.lib oleaut32.lib uuid.lib user32.lib
    if ($LASTEXITCODE -ne 0) { throw 'TSF integration compilation failed' }
    if ($Registered) {
        $registeredDll = (Get-ItemProperty -LiteralPath 'Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Classes\CLSID\{97D8A216-4D8C-4AD5-9839-AF53EA9ADB41}\InprocServer32').'(default)'
        & (Join-Path $output 'tsf-test.exe') $registeredDll --registered
    } else {
        & (Join-Path $output 'tsf-test.exe') (Join-Path $output 'nextime_tsf.dll')
    }
    if ($LASTEXITCODE -ne 0) { throw 'TSF integration failed' }
} finally {
    $env:TEMP = $savedTemp
    $env:TMP = $savedTmp
    $env:PATH = $savedPath
    Pop-Location
}
