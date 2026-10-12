# SPDX-License-Identifier: AGPL-3.0-only
param([Parameter(Mandatory = $true)][string]$DictionaryPath)
$ErrorActionPreference = 'Stop'
$metadata = cargo metadata --manifest-path (Join-Path $PSScriptRoot '../Cargo.toml') --no-deps --format-version 1 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'cargo metadata failed' }
$cli = Join-Path $metadata.target_directory 'release/nextime.exe'
function Run-Session([string[]]$commands) {
    $output = $commands | & $cli session --nud $DictionaryPath --limit 8
    if ($LASTEXITCODE -ne 0) { throw 'AI session failed' }
    $output -join "`n"
}
foreach ($case in @(
    @('あなるセックスしたい', 'アナルセックスしたい'),
    @('anarusekkusushitai', 'アナルセックスしたい'),
    @('べんきょうしたい', '勉強したい'),
    @('けんさくしたいです', '検索したいです'),
    @('せっくすしたくない', 'セックスしたくない')
)) {
    foreach ($ai in @($true, $false)) {
        $commands=@()
        if (-not $ai) { $commands += 'ai' }
        # Session CLI uses 'ai' for the explicit F9 equivalent.
        foreach ($character in $case[0].ToCharArray()) { $commands += "text $character" }
        $commands += @('space', 'quit')
        $result=Run-Session $commands
        if (-not $result.Contains("preedit: $($case[1])")) { throw "Desire conjugation failed: $($case[0])" }
        if ($result -match 'した(位|医|胃)') { throw "Desire conjugation produced a split noun: $($case[0])" }
    }
}
foreach ($editorInput in @('vscode', 'VSCode', 'vscodede', 'vscodeを')) {
    $editor = Run-Session @("text $editorInput", 'space', 'quit')
    $editorExpected = if ($editorInput -ceq 'vscodede') { 'vscodeで' } else { $editorInput }
    if (-not $editor.Contains("preedit: $editorExpected")) { throw "VSCode mixed input failed: $editorInput" }
}
$acronym = Run-Session @('text AIによる', 'space', 'quit')
foreach ($projectInput in @('purozyekuto', 'purojyekuto', 'ぷろzyえくと', 'ぷろzyエクト')) {
    $project = Run-Session @("text $projectInput", 'space', 'quit')
    if (-not $project.Contains('preedit: プロジェクト')) { throw "Project spelling failed: $projectInput" }
}
if (-not $acronym.Contains('preedit: AIによる')) { throw 'Explicit AI was lost in Japanese text' }
$romajiAcronym = Run-Session @('text AIniyoru', 'space', 'quit')
if (-not $romajiAcronym.Contains('preedit: AIによる')) { throw 'AI romaji particle boundary was lost' }
$kanaAi = Run-Session @('text あいによる', 'space', 'quit')
if (-not $kanaAi.Contains('preedit: AIによる')) { throw 'Requested kana AI phrase was not converted to AI' }
if ($kanaAi.Contains('夜') -or $kanaAi.Contains('会いによる')) { throw 'による still generates broken combinations' }
foreach ($aiSwitch in @($false, $true)) {
    foreach ($case in @(
        @('ChatGPTde', 'ChatGPTで'), @('ちゃtgpt', 'ChatGPT'),
        @('GitHubwo', 'GitHubを'), @('OpenAIga', 'OpenAIが'),
        @('YouTubede', 'YouTubeで'), @('あおぞらぶんこ', '青空文庫'),
        @('なつめそうせき', '夏目漱石'), @('にんてんどう', '任天堂'),
        @('あいによる', 'AIによる'), @('ainiyoru', 'AIによる'),
        @('あい', 'AI'), @('ai', 'AI'),
        @('あいが', 'AIが'), @('あいを', 'AIを'), @('aiga', 'AIが'), @('えーあい', 'AI'),
        @('あいをつかい', 'AIを使い'), @('aiwotsukai', 'AIを使い'),
        @('あいをつかって', 'AIを使って'), @('あいをつかいたい', 'AIを使いたい'),
        @('あいをつかいます', 'AIを使います'), @('あいを使い', 'AIを使い'),
        @('あいをりよう', 'AIを利用'),
        @('あいをたしかめたい', 'AIを確かめたい'),
        @('aiwotashikametai', 'AIを確かめたい'),
        @('あいをかくにんしたい', 'AIを確認したい')
    )) {
        $commands = @()
        if ($aiSwitch) { $commands += 'ai' }
        foreach ($ch in $case[0].ToCharArray()) { $commands += "text $ch" }
        $commands += @('space', 'quit')
        $result = Run-Session $commands
        $lastPreedit = @($result -split "`n" | Where-Object { $_.StartsWith('preedit:') })[-1]
        if ($case[0] -in @('あいをたしかめたい','aiwotashikametai')) {
            if ($lastPreedit -cnotin @('preedit: AIを確かめたい','preedit: 愛を確かめたい') -or
                -not $result.Contains('AIを確かめたい') -or -not $result.Contains('愛を確かめたい')) {
                throw "Ambiguous love/AI choices lost: $($case[0])"
            }
        } elseif ($lastPreedit -cne "preedit: $($case[1])") { throw "Proper noun regression: $($case[0]), AI disabled=$aiSwitch, actual=$lastPreedit" }
        if ($result.Contains('commit:')) { throw 'Proper noun correction auto-committed text' }
    }
}
Write-Output 'Proper noun regressions passed: character input, mixed particles, kana AI phrase, AI on/off, no auto-commit.'
foreach ($loveInput in @('あいをつたえる', 'あいする', 'あいをこめて')) {
    $love = Run-Session @("text $loveInput", 'space', 'quit')
    if ($love -match '(?m)^\s*\d+\. .*AI') { throw "Love expression was changed to AI: $loveInput" }
}
$love=Run-Session @('text あいをつかめ', 'space', 'quit')
if (($love -notmatch 'preedit: 愛を(掴め|つかめ)') -or -not $love.Contains('AIを掴め')) { throw 'Grasp-love collocation or alternative lost' }
foreach ($sentence in @('あいをつかって文章を書く', 'あいをつかいたいです', 'あいをつかいます。', 'あいをりようする')) {
    $result = Run-Session @("text $sentence", 'space', 'quit')
    $lastPreedit = @($result -split "`n" | Where-Object { $_.StartsWith('preedit:') })[-1]
    $prefix = if ($sentence -eq 'あいをりようする') { 'preedit: AIを利用する' } else { 'preedit: AIを使' }
    if (-not $lastPreedit.StartsWith($prefix)) { throw "AI sentence prefix failed: $sentence, $lastPreedit" }
    if ($result.Contains('commit:')) { throw 'AI phrase auto-committed text' }
}
$shortPhrase = Run-Session @('text yoroshi', 'quit')
if ($shortPhrase.Contains('よろしくお願いします')) { throw 'Short input still predicts a whole sentence' }
$shortAi = Run-Session @('text ai', 'quit')
if ($shortAi.Contains('preedit: AI') -or $shortAi -notmatch '(?m)^[ >]*\d+\. AI$') { throw 'AI candidate missing, or raw preedit was unexpectedly replaced' }
$food = Run-Session @('text 食事に', 'enter', 'text hashi', 'space', 'quit')
if (-not $food.Contains('preedit: 箸')) { throw 'Context did not favor chopsticks' }
$previousSentence = Run-Session @('text 夕食の準備ができました。', 'enter', 'text hashi', 'space', 'quit')
if (-not $previousSentence.Contains('preedit: 箸')) { throw 'Previous sentence topic was ignored' }
$riverSentence = Run-Session @('text 川の向こうに駅があります。', 'enter', 'text hashi', 'space', 'quit')
if (-not $riverSentence.Contains('preedit: 橋')) { throw 'River topic was ignored' }
$changedTopic = Run-Session @('text 食事の話は終わりです。川を見に行きます。', 'enter', 'text hashi', 'space', 'quit')
if (-not $changedTopic.Contains('preedit: 橋')) { throw 'Most recent topic was ignored' }
$off = Run-Session @('ai', 'text 食事に', 'enter', 'text hashi', 'space', 'quit')
if (-not $off.Contains('preedit: 橋')) { throw 'Disabled AI changed ordinary conversion' }
$cleared = Run-Session @('text 食事に', 'enter', 'clear-context', 'text hashi', 'space', 'quit')
if (-not $cleared.Contains('preedit: 橋')) { throw 'Cleared context was still used' }
$on = Run-Session @('text せいきででてこない', 'space', 'quit')
if (-not $on.Contains('preedit: 正規ででてこない')) { throw 'Local model did not rerank spelling' }
$baseline = Run-Session @('ai', 'text せいきででてこない', 'space', 'quit')
if (-not $baseline.Contains('preedit: 世紀ででてこない')) { throw 'Baseline changed unexpectedly' }
if ($on.Contains('commit:')) { throw 'Model must never auto-commit' }
$learned = Run-Session @('text hashi','space','next','enter','clear-context','text はし','space','quit')
if (-not $learned.Contains('commit: 箸')) { throw 'Selected candidate was not committed' }
if (-not $learned.Contains("commit: 箸`npreedit:")) { throw 'Unexpected session output' }
$lastPreedit = @($learned -split "`n" | Where-Object { $_.StartsWith('preedit:') })[-1]
if ($lastPreedit -ne 'preedit: 箸') { throw 'Personal choice was not reused after clearing context' }
$forgotten = Run-Session @('text hashi','space','next','enter','clear-learning','text はし','space','quit')
$lastPreedit = @($forgotten -split "`n" | Where-Object { $_.StartsWith('preedit:') })[-1]
if ($lastPreedit -ne 'preedit: 橋') { throw 'Cleared personal choice was still used' }
Write-Output 'Context AI regressions passed: contextual ranking, disable, clear, real dictionary, no auto-commit.'
Write-Output 'Personal learning regressions passed: explicit choice, normalized reading, context independence, forget.'
$typed = Run-Session @('text 食','text 事','text に','text 箸','text を','text 使','text い','text ま','text す','text 。','enter','quit')
if (-not $typed.Contains('commit: 食事に箸を使います。')) { throw 'Character-by-character input was not preserved' }
$lastLearning = @($typed -split "`n" | Where-Object { $_.StartsWith('learning:') })[-1]
if ($lastLearning -notmatch '^learning: on features=[1-9][0-9]*$') { throw 'Confirmed sentence was not learned' }
$stopped = Run-Session @('training','text 学習しない文章','enter','quit')
if (-not $stopped.Contains('learning: off features=0')) { throw 'Disabled sentence learning was used' }
$cancelled = Run-Session @('text 未確定の文章','escape','quit')
if ($cancelled -notmatch 'learning: on features=0') { throw 'Cancelled input was learned' }
Write-Output 'Human sentence learning regressions passed: character input, confirmed text, disable, cancel.'
