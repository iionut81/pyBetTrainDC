# Isolated integration checks: fake child processes, no research or task changes.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'monitored_process.ps1')
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('pred-process-test-' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($testRoot) | Out-Null
$nodePath = (Get-Command node -ErrorAction Stop).Source
$fixturePath = Join-Path $testRoot 'worker.cjs'
[IO.File]::WriteAllText($fixturePath, @'
process.stdin.resume();
process.stdin.on('end', () => {
  process.stdout.write('partial-without-newline');
  process.stderr.write('diagnostic-without-newline');
  if(process.argv[2]==='timeout') setTimeout(()=>{}, 20000);
  else setTimeout(()=>{process.stdout.write('\nfinished\n');process.exit(Number(process.argv[2]||0))},1800);
});
'@)
function New-TestStartInfo([string]$Mode) {
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $nodePath
    $info.Arguments = '"' + $fixturePath + '" ' + $Mode
    $info.UseShellExecute = $false; $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true; $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
    return $info
}
$script:streamSeenBeforeExit = $false
$script:heartbeatCount = 0
$outFile = Join-Path $testRoot 'stdout.log'; $errFile = Join-Path $testRoot 'stderr.log'
$exitCode = Invoke-PredMonitoredProcess -StartInfo (New-TestStartInfo '0') -InputText 'fixture' -StdoutPath $outFile -StderrPath $errFile -HeartbeatSeconds 1 -OnHeartbeat {
    param($beat)
    $script:heartbeatCount++
    if ($beat.stdout_bytes -gt 0 -and $beat.stderr_bytes -gt 0) { $script:streamSeenBeforeExit = $true }
    Write-PredAtomicJson (Join-Path $testRoot 'heartbeat.json') $beat
}
if ($exitCode -ne 0 -or -not $script:streamSeenBeforeExit -or $script:heartbeatCount -lt 2) { throw ('Streaming before exit/heartbeat failed: exit=' + ($exitCode | ConvertTo-Json -Compress) + ', seen=' + $script:streamSeenBeforeExit + ', beats=' + $script:heartbeatCount) }
if (-not (Get-Content $outFile -Raw).Contains('finished')) { throw 'Final stdout was lost' }
$identity = Get-Content (Join-Path $testRoot 'heartbeat.json') -Raw | ConvertFrom-Json
if (Test-PredProcessIdentity $identity.worker_pid $identity.worker_started_at_utc) { throw 'Exited worker incorrectly considered active' }
if (-not (Test-PredProcessIdentity $PID (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o'))) { throw 'Live identity not recognized' }
if (Test-PredProcessIdentity $PID '2000-01-01T00:00:00Z') { throw 'PID reuse guard failed' }
$failedExit = Invoke-PredMonitoredProcess -StartInfo (New-TestStartInfo '73') -InputText 'fixture' -StdoutPath $outFile -StderrPath $errFile
if ($failedExit -ne 73 -or -not (Get-Content $errFile -Raw).Contains('diagnostic')) { throw 'Failure diagnostics lost' }
$timeoutCaught = $false
try { Invoke-PredMonitoredProcess -StartInfo (New-TestStartInfo 'timeout') -InputText 'fixture' -StdoutPath $outFile -StderrPath $errFile -TimeoutSeconds 1 | Out-Null }
catch { if ($_.Exception.Message -notlike '*execution timeout*') { throw }; $timeoutCaught = $true }
if (-not $timeoutCaught -or -not (Get-Content $outFile -Raw).Contains('partial')) { throw 'Timeout evidence lost' }
foreach ($scriptName in @('run_daily_research.ps1','publish_daily_research.ps1','repair_daily_research_task.ps1','monitored_process.ps1')) {
    $parseErrors = $null; $parseTokens = $null
    [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot $scriptName), [ref]$parseTokens, [ref]$parseErrors) | Out-Null
    if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
}
. (Join-Path $PSScriptRoot 'daily_research_policy.ps1')
if ((Get-PredDailyDecision -NowRo ([datetime]'2026-11-06T07:59:00') -NotBeforeDate '2026-11-04' -AlreadyFinished $false -ChatPending $false) -ne 'before_0800') { throw 'Early run allowed' }
if ((Get-PredDailyDecision -NowRo ([datetime]'2026-11-06T08:30:00') -NotBeforeDate '2026-11-04' -AlreadyFinished $true -ChatPending $false) -ne 'already_finished') { throw 'Completed research would repeat' }
if ((Get-PredDailyDecision -NowRo ([datetime]'2026-11-06T08:30:00') -NotBeforeDate '2026-11-04' -AlreadyFinished $true -ChatPending $true) -ne 'deliver_only') { throw 'Delivery retry would repeat research' }
if (Test-PredChatNeedsRetry 'not_required_manual') { throw 'Manual recovery would create a duplicate automatic chat delivery' }
[pscustomobject]@{status='PASS';checks=@('output flushed before child exit without newline','heartbeat and atomic status','PID plus creation-time guard','nonzero exit retains diagnostics','timeout retains output and stops owned child','all changed PowerShell scripts parse','before08 and completion/delivery guards');fixture_root=$testRoot} | ConvertTo-Json
