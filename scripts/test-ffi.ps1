# SPDX-License-Identifier: AGPL-3.0-only
$ErrorActionPreference = 'Stop'
$ffiOriginalTemp = $env:TEMP
$ffiOriginalTmp = $env:TMP
Push-Location (Join-Path $PSScriptRoot '..')
try {
    cargo build -p nextime-ffi --release --locked
    if ($LASTEXITCODE -ne 0) { throw 'Rust FFI build failed' }
    $ffiMetadata = cargo metadata --no-deps --format-version 1 | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'cargo metadata failed' }
    $ffiOutput = Join-Path $ffiMetadata.target_directory 'release'
    $ffiExecutable = Join-Path $ffiOutput 'nextime-ffi-demo.exe'
    $ffiImportLibrary = Join-Path $ffiOutput 'nextime_ffi.dll.lib'
    $ffiVswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $ffiVsPath = & $ffiVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $ffiVsPath) { throw 'Visual Studio C++ Build Tools are required' }
    Import-Module (Join-Path $ffiVsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $ffiVsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
    $ffiObject = Join-Path $ffiOutput 'nextime-ffi-demo.obj'
    # Keep compiler response files beside build artifacts. Some Windows compiler
    # versions fail to create them under a non-ASCII user temp directory.
    $env:TEMP = $ffiOutput
    $env:TMP = $ffiOutput
    cl /nologo /std:c11 /utf-8 /W4 /WX '/Icrates/nextime-ffi/include' 'platforms/common/ffi-demo.c' $ffiImportLibrary "/Fe$ffiExecutable" "/Fo$ffiObject"
    if ($LASTEXITCODE -ne 0) { throw 'C ABI demo compilation failed' }
    & $ffiExecutable
    if ($LASTEXITCODE -ne 0) { throw 'C ABI demo execution failed' }
} finally {
    $env:TEMP = $ffiOriginalTemp
    $env:TMP = $ffiOriginalTmp
    Pop-Location
}
