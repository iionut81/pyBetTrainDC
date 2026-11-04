#requires -Version 5.1
[CmdletBinding()]
param([switch]$Preflight, [switch]$PrepareOnly, [string]$ConfigPath)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$configFile = if ($ConfigPath) { $ConfigPath } else { Join-Path $PSScriptRoot 'daily_research_config.json' }
$config = Get-Content -LiteralPath $configFile -Raw -Encoding UTF8 | ConvertFrom-Json
. (Join-Path $PSScriptRoot 'daily_research_policy.ps1')
. (Join-Path $PSScriptRoot 'publish_daily_research.ps1')
$codexExe = $config.codex_exe
if (-not (Test-Path -LiteralPath $codexExe)) {
    $cmd = Get-Command codex -ErrorAction SilentlyContinue
    if ($cmd) { $codexExe = $cmd.Source }
    else {
        $extensionRoot = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.vscode/extensions'
        $candidate = Get-ChildItem -LiteralPath $extensionRoot -Directory -Filter 'openai.chatgpt-*' |
            Sort-Object LastWriteTime -Descending | ForEach-Object {
                $exe = Join-Path $_.FullName 'bin/windows-x86_64/codex.exe'
                if (Test-Path -LiteralPath $exe) { $exe }
            } | Select-Object -First 1
        if (-not $candidate) { throw 'Codex executable unavailable.' }
        $codexExe = $candidate
    }
}
foreach ($relative in @('AGENTS.md','Prompts/Football_Goal70_Corners1H_Master.md','Analysis/Selection_History/user_selections.csv','Previews/Pred_Research_Terminal/build.mjs','automation/daily_research_prompt.md')) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectDir $relative))) { throw "Required file missing: $relative" }
}
if (-not (Test-Path -LiteralPath $config.node_exe)) { throw 'Node executable unavailable.' }
if ($config.chat_thread_id -notmatch '^[a-fA-F0-9-]{36}$') { throw 'Destination chat is not configured.' }
if ($config.not_before_date -notmatch '^\d{4}-\d{2}-\d{2}$') { throw 'First scheduled date is not configured.' }
$roZone = [TimeZoneInfo]::FindSystemTimeZoneById('GTB Standard Time')
$nowRo = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $roZone)
$today = $nowRo.ToString('yyyy-MM-dd')
$yesterday = $nowRo.AddDays(-1).ToString('yyyy-MM-dd')
$prompt = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'daily_research_prompt.md'),[Text.Encoding]::UTF8).
    Replace('{{TODAY}}',$today).Replace('{{YESTERDAY}}',$yesterday)
if ($Preflight -or $PrepareOnly) {
    $authInfo = New-Object Diagnostics.ProcessStartInfo
    $authInfo.FileName = $codexExe
    $authInfo.Arguments = 'login status'
    $authInfo.UseShellExecute = $false
    $authInfo.CreateNoWindow = $true
    $authInfo.RedirectStandardOutput = $true
    $authInfo.RedirectStandardError = $true
    $authProcess = [Diagnostics.Process]::Start($authInfo)
    $authOutput = $authProcess.StandardOutput.ReadToEndAsync()
    $authError = $authProcess.StandardError.ReadToEndAsync()
    $authProcess.WaitForExit()
    if ($authProcess.ExitCode -ne 0) { throw 'Codex authentication is unavailable.' }
    [pscustomobject]@{status='READY';project=$projectDir;date_ro=$today;review_date_ro=$yesterday;
        codex_exe=$codexExe;node_exe=$config.node_exe;prompt_characters=$prompt.Length;
        launches_agent=$false;time_zone='Europe/Bucharest';schedule='08:00 daily';chat_thread_id=$config.chat_thread_id;not_before_date=$config.not_before_date} | ConvertTo-Json
    return
}
$runDir = Join-Path $projectDir ('Analysis/Automation/Runs/' + $nowRo.ToString('yyyy-MM-dd_HHmmss'))
$autoDir = Join-Path $projectDir 'Analysis/Automation'
[IO.Directory]::CreateDirectory($autoDir) | Out-Null
try { $lock = [IO.File]::Open((Join-Path $autoDir 'daily_research.lock'), 'OpenOrCreate', 'ReadWrite', 'None') }
catch [IO.IOException] { Write-Output 'A PRED research run is already active; duplicate skipped.'; exit 0 }
$utf8 = New-Object Text.UTF8Encoding($false)
$completionFile = Join-Path $autoDir ('daily_completion_' + $today + '.json')
function Sync-PredConversation {
    try {
        & $config.node_exe (Join-Path $projectDir 'automation/export_project_conversation.mjs') --thread $config.chat_thread_id
        if ($LASTEXITCODE -ne 0) { Write-Warning 'Conversation export incomplete; native chat and analytical journals remain available.' }
    } catch { Write-Warning ('Conversation export: ' + $_.Exception.Message) }
}
$completion = if (Test-Path -LiteralPath $completionFile) { Get-Content -LiteralPath $completionFile -Raw -Encoding UTF8 | ConvertFrom-Json } else { $null }
$decision = Get-PredDailyDecision -NowRo $nowRo -NotBeforeDate $config.not_before_date -AlreadyFinished ([bool]$completion) -ChatPending ([bool]($completion -and (Test-PredChatNeedsRetry -Status $completion.chat_status)))
if ($decision -in @('before_0800','not_started','already_finished')) {
    $lock.Dispose()
    Write-Output "Daily run skipped: $decision"
    exit 0
}
if ($decision -eq 'deliver_only') {
    try {
        $delivery = Publish-PredDailyReport -CodexExe $codexExe -ThreadId $config.chat_thread_id -RunDir $completion.run_dir -ProjectDir $projectDir -Date $today
        $completion.chat_status = $delivery.status
        $completion | ConvertTo-Json -Depth 8 | ForEach-Object { [IO.File]::WriteAllText($completionFile,$_,$utf8) }
        $delivery | ConvertTo-Json | ForEach-Object { [IO.File]::WriteAllText((Join-Path $completion.run_dir 'chat_delivery.json'),$_,$utf8) }
    } finally { Sync-PredConversation; $lock.Dispose() }
    exit 0
}
function Save-Status([string]$state,[string]$message) {
    $status = [ordered]@{task_name=$config.task_name;schedule='08:00 daily';timezone='Europe/Bucharest';
        status=$state;run_date=$today;review_date=$yesterday;updated_at_utc=[DateTime]::UtcNow.ToString('o');
        message=$message;run_dir=$runDir;local_computer_required=$true}
    $text = $status | ConvertTo-Json -Depth 6
    [IO.File]::WriteAllText((Join-Path $autoDir 'daily_status.json'),$text,$utf8)
    [IO.File]::WriteAllText((Join-Path $runDir 'run.json'),$text,$utf8)
}
function Rebuild-Dashboard {
    & $config.node_exe (Join-Path $projectDir 'Previews/Pred_Research_Terminal/build.mjs')
    if ($LASTEXITCODE -ne 0) { throw 'Dashboard rebuild failed.' }
}
try {
    [IO.Directory]::CreateDirectory($runDir) | Out-Null
    [IO.File]::WriteAllText((Join-Path $runDir 'prompt.md'),$prompt,$utf8)
    Save-Status 'running' 'Review previous outcomes, then research both markets.'
    Rebuild-Dashboard
    # All arguments are fixed flags or validated local paths, passed to the executable directly.
    $summaryFile = Join-Path $runDir 'summary.md'
    $agentStartedAtUtc = [DateTime]::UtcNow
    $argsList = @('--search','exec','--approve-for-me','--cd',$projectDir,'--skip-git-repo-check','--color','never','--output-last-message',$summaryFile,'-')
    $quotedArgs = $argsList | ForEach-Object { '"' + $_.Replace('"','\"') + '"' }
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $codexExe
    $info.Arguments = $quotedArgs -join ' '
    $info.WorkingDirectory = $projectDir
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    if ($info.PSObject.Properties.Name -contains 'StandardInputEncoding') { $info.StandardInputEncoding = $utf8 }
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    if (-not $process.Start()) { throw 'Could not start Codex.' }
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    $promptBytes = $utf8.GetBytes($prompt + "`n")
    $process.StandardInput.BaseStream.Write($promptBytes,0,$promptBytes.Length)
    $process.StandardInput.BaseStream.Flush()
    $process.StandardInput.Close()
    $process.WaitForExit()
    [IO.File]::WriteAllText((Join-Path $runDir 'agent.log'),$stdout.Result,$utf8)
    [IO.File]::WriteAllText((Join-Path $runDir 'diagnostics.log'),$stderr.Result,$utf8)
    if ($process.ExitCode -ne 0) { throw ('Codex exited with code ' + $process.ExitCode) }
    $reportFile = Join-Path $autoDir ('run_report_' + $today + '.json')
    $report = if (Test-Path -LiteralPath $reportFile) { Get-Content -LiteralPath $reportFile -Raw -Encoding UTF8 | ConvertFrom-Json } else { $null }
    $freshReport = $report -and (Get-Item -LiteralPath $reportFile).LastWriteTimeUtc -ge $agentStartedAtUtc
    $full = $freshReport -and $report.date -eq $today -and $report.review_status -eq 'complete' -and
        $report.goal_research_status -eq 'complete' -and $report.corner_research_status -eq 'complete' -and $report.dashboard_status -eq 'complete'
    if ($full) { Save-Status 'completed' 'Review, both research markets and dashboard reported complete.' }
    else { Save-Status 'partial' 'Agent finished; inspect report for missing data or incomplete stages.' }
    Rebuild-Dashboard
    # Preserve once-per-day research even if only chat delivery needs retrying.
    $completion = [pscustomobject]@{date=$today;run_dir=$runDir;research_status=$(if($full){'completed'}else{'partial'});chat_status='pending';completed_at_utc=[DateTime]::UtcNow.ToString('o')}
    [IO.File]::WriteAllText($completionFile,($completion | ConvertTo-Json),$utf8)
    try {
        $delivery = Publish-PredDailyReport -CodexExe $codexExe -ThreadId $config.chat_thread_id -RunDir $runDir -ProjectDir $projectDir -Date $today
        $completion.chat_status = $delivery.status
        [IO.File]::WriteAllText($completionFile,($completion | ConvertTo-Json),$utf8)
        [IO.File]::WriteAllText((Join-Path $runDir 'chat_delivery.json'),($delivery | ConvertTo-Json),$utf8)
    } catch {
        [IO.File]::WriteAllText((Join-Path $runDir 'chat_delivery_error.txt'),$_.Exception.Message,$utf8)
        Write-Warning $_.Exception.Message
    }
} catch {
    Save-Status 'failed' $_.Exception.Message
    try { Rebuild-Dashboard } catch { Write-Warning $_.Exception.Message }
    throw
} finally { Sync-PredConversation; if ($lock) { $lock.Dispose() } }
