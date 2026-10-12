# SPDX-License-Identifier: AGPL-3.0-only
param([Parameter(Mandatory = $true)][string]$DictionaryPath)
$ErrorActionPreference = 'Stop'
$metadata = cargo metadata --manifest-path (Join-Path $PSScriptRoot '../Cargo.toml') --no-deps --format-version 1 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'cargo metadata failed' }
$cli = Join-Path $metadata.target_directory 'release/nextime.exe'
function Convert-Candidates([string]$reading) {
    $lines = & $cli --nud $DictionaryPath --limit 24 $reading
    if ($LASTEXITCODE -ne 0) { throw "Conversion failed: $reading" }
    @($lines | Where-Object { $_ -match '^\d+\. ' } | ForEach-Object { ($_ -replace '^\d+\. ', '') -split '\t' | Select-Object -First 1 })
}
$request = Convert-Candidates 'たいさくしてほしいです'
if ($request[0] -ne '対策してほしいです') { throw "Wrong request: $request" }
if (@($request | Where-Object { $_ -match '仕手|志手|シテ|大作|太作|泰作' }).Count) { throw "Broken request candidates: $request" }
$word = Convert-Candidates 'あなる'
if ($word -notcontains 'アナル' -or $word -notcontains 'あなる') { throw 'Whole word or kana fallback lost' }
if (@($word | Where-Object { $_ -match '亜成|亜鳴|奈留|奈瑠|穴る' }).Count) { throw "Broken word candidates: $word" }
$sentence = Convert-Candidates 'にほんごをにゅうりょくします'
if ($sentence[0] -ne '日本語を入力します' -or @($sentence | Where-Object { $_ -match '死ます|視ます|市ます|ヲ' }).Count) { throw "Broken sentence: $sentence" }
$homophones = Convert-Candidates 'はし'
foreach ($surface in @('橋', '箸', '端', 'はし')) {
    if ($homophones -notcontains $surface) { throw "Homophone lost: $surface" }
}
$noun = Convert-Candidates 'して'
if ($noun -notcontains '仕手') { throw 'Valid standalone noun lost' }
Write-Output 'Japanese regressions passed: screenshot examples, sentence, homophones, standalone noun, kana fallback.'
foreach ($inputWord in @('sex', 'SEX', 'sexが', 'sexga', 'SEXga', 'sexyが', 'sexyga', 'rustde')) {
    $expected = switch -CaseSensitive ($inputWord) {
        'sexga' { 'sexが' }
        'SEXga' { 'SEXが' }
        'sexyga' { 'sexyが' }
        'rustde' { 'rustで' }
        default { $inputWord }
    }
    $mixed = @(Convert-Candidates $inputWord)
    if ($mixed[0] -cne $expected) { throw "Mixed language failed: $inputWord -> $mixed (expected $expected)" }
}
Write-Output 'Mixed-language regressions passed: sex, casing, Japanese script and romaji particles.'
foreach ($reading in @('sexgadekinai', 'sexができない', 'SEXgadekinai', 'sexganai', 'rustnodekinai')) {
    $candidates = @(Convert-Candidates $reading)
    if (@($candidates | Where-Object { $_ -match '(sex|SEX)(我|画|蛾|賀|ガ|駕|臥)|デキナイ|rust野' }).Count) {
        throw "Particle/negative predicate corruption: $reading -> $candidates"
    }
    if ($reading -eq 'sexgadekinai' -and $candidates[0] -ne 'sexができない') { throw 'Expected natural predicate lost' }
    if ($reading -eq 'rustnodekinai' -and $candidates[0] -ne 'rustのできない') { throw 'Mixed の was lost' }
    if ($reading -eq 'sexganai' -and @($candidates | Where-Object { $_ -notin @('sexがない', 'sexが無い') }).Count) { throw 'Fragmented negative expression' }
}
Write-Output 'Particle and negative-predicate regressions passed.'
