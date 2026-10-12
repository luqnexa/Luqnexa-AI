# SPDX-License-Identifier: AGPL-3.0-only
# Synthetic isolated sessions only. Never inserts returned edits into an app.
param([string]$Cases='tests/sentence-suite.tsv', [string]$Output='benchmarks/results/meltype-sentence-suite.json')
$ErrorActionPreference='Stop'
$sid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$pipe=[IO.Pipes.NamedPipeClientStream]::new('.','Meltype.Tip.'+$sid,[IO.Pipes.PipeDirection]::InOut,[IO.Pipes.PipeOptions]::Asynchronous)
function Send-Request($request) {
    $bytes=[Text.Encoding]::UTF8.GetBytes(($request | ConvertTo-Json -Compress))
    $pipe.Write($bytes,0,$bytes.Length)
    $pipe.Flush()
    $stream=[IO.MemoryStream]::new()
    try {
        do {
            $buffer=[byte[]]::new(32768)
            $cancel=[Threading.CancellationTokenSource]::new(5000)
            try {$n=$pipe.ReadAsync($buffer,0,$buffer.Length,$cancel.Token).GetAwaiter().GetResult()}
            finally {$cancel.Dispose()}
            if($n -eq 0){throw 'Meltype pipe closed'}
            $stream.Write($buffer,0,$n)
            if($stream.Length -gt 1048576){throw 'Oversized reply'}
        } while(-not $pipe.IsMessageComplete)
        return [Text.Encoding]::UTF8.GetString($stream.ToArray()) | ConvertFrom-Json
    } finally {$stream.Dispose()}
}
try {
    $pipe.Connect(2000)
    $pipe.ReadMode=[IO.Pipes.PipeTransmissionMode]::Message
    $hello=Send-Request @{op='hello';process='LuqnexaBenchmark'}
    if(-not $hello.active){throw 'Meltype is inactive; no user settings were changed'}
    $results=@()
    foreach($line in Get-Content -LiteralPath $Cases -Encoding utf8) {
        if($line.StartsWith('#') -or -not $line){continue}
        $fields=$line.Split("`t")
        # Shared comparison convention: remove spaces inside roman composition.
        $inputText=$fields[0].Replace(' ','')
        $session=[Guid]::NewGuid().ToString('N')
        $timer=[Diagnostics.Stopwatch]::StartNew()
        $commitCount=0
        $documentText=''
        $passedThrough=''
        foreach($ch in $inputText.ToCharArray()) {
            $vk=if([int]$ch -lt 128){[int][char]::ToUpperInvariant($ch)}else{0}
            $reply=Send-Request @{op='key';sid=$session;process='LuqnexaBenchmark';vk=$vk;ch=[int]$ch;mods=0;before='';after=''}
            $commitCount+=@($reply.commits).Count
            foreach($edit in $reply.commits) {
                $delete=[Math]::Min($documentText.Length,[int]$edit.deleteBefore)
                $documentText=$documentText.Substring(0,$documentText.Length-$delete)+$edit.text
            }
            if(-not $reply.consumed){$documentText+=$ch; $passedThrough+=$ch}
        }
        $reply=Send-Request @{op='key';sid=$session;process='LuqnexaBenchmark';vk=32;ch=32;mods=0;before='';after=''}
        $timer.Stop()
        $commitCount+=@($reply.commits).Count
        foreach($edit in $reply.commits) {
            $delete=[Math]::Min($documentText.Length,[int]$edit.deleteBefore)
            $documentText=$documentText.Substring(0,$documentText.Length-$delete)+$edit.text
        }
        $actual=$documentText+$reply.view.text
        $results += @{input=$fields[0];comparison_input=$inputText;expected=$fields[1];actual=$actual;passed=($actual -ceq $fields[1]);elapsed_ms=$timer.Elapsed.TotalMilliseconds;returned_commits=$commitCount;passed_through=$passedThrough}
        # Cancel before close, so Close cannot commit the pending composition.
        $null=Send-Request @{op='key';sid=$session;process='LuqnexaBenchmark';vk=27;ch=0;mods=0;before='';after=''}
        $null=Send-Request @{op='close';sid=$session;process='LuqnexaBenchmark'}
    }
    $report=@{product='Meltype';installed_version=(Get-Item "$env:LOCALAPPDATA/Programs/Meltype/Meltype.exe").VersionInfo.ProductVersion;
        scope='Live named-pipe synthetic sessions; original user settings and learning state; spaces removed; Text characters then Space; no Enter, app edits or external input transmission. Latency includes IPC and UI thread; not comparable to Luqnexa in-process FFI latency.';
        passed=@($results | Where-Object passed).Count;total=$results.Count;results=$results}
    New-Item -ItemType Directory -Force (Split-Path $Output -Parent) | Out-Null
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Output -Encoding utf8
    Write-Output "Meltype synthetic isolated session results: $($report.passed)/$($report.total); saved $Output"
} finally {$pipe.Dispose()}
