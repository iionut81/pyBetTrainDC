#requires -Version 5.1
[CmdletBinding()]
param([switch]$PreviewOnly, [string]$ChatThreadId)
$ErrorActionPreference = 'Stop'
$taskName = 'Pred_Daily_Research_0800'
$projectDir = Split-Path -Parent $PSScriptRoot
$codexExe = (Get-Command codex -ErrorAction Stop).Source
$nodeExe = (Get-Command node -ErrorAction Stop).Source
if ((Get-TimeZone).Id -ne 'GTB Standard Time') { throw 'Task requires the Windows Bucharest time zone (GTB Standard Time).' }
$userId = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$configPath = Join-Path $PSScriptRoot 'daily_research_config.json'
$previousConfig = if (Test-Path -LiteralPath $configPath) { Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json } else { $null }
if (-not $ChatThreadId) { $ChatThreadId = $previousConfig.chat_thread_id }
if (-not $ChatThreadId) { $ChatThreadId = $env:CODEX_THREAD_ID }
if ($ChatThreadId -notmatch '^[a-fA-F0-9-]{36}$') { throw 'A valid current-chat UUID is required for daily link delivery.' }
$runner = Join-Path $PSScriptRoot 'run_daily_research.ps1'
$powerShellExe = Join-Path $PSHOME 'powershell.exe'
$arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $runner + '"'
$action = New-ScheduledTaskAction -Execute $powerShellExe -Argument $arguments -WorkingDirectory $projectDir
$trigger = New-ScheduledTaskTrigger -Daily -At '08:00'
if ([DateTime]::Now.TimeOfDay -ge [TimeSpan]::FromHours(8)) {
    $trigger.StartBoundary = [DateTime]::Today.AddDays(1).AddHours(8).ToString('yyyy-MM-ddTHH:mm:ss')
}
$firstDate = if ($previousConfig.not_before_date) { $previousConfig.not_before_date } else { $trigger.StartBoundary.Substring(0,10) }
$startBoundary = $firstDate + 'T08:00:00'
$logon = New-ScheduledTaskTrigger -AtLogOn -User $userId
$logon.StartBoundary = $startBoundary
$sessionClass = Get-CimClass -Namespace 'Root/Microsoft/Windows/TaskScheduler' -ClassName 'MSFT_TaskSessionStateChangeTrigger'
$eventClass = Get-CimClass -Namespace 'Root/Microsoft/Windows/TaskScheduler' -ClassName 'MSFT_TaskEventTrigger'
$unlock = New-CimInstance -CimClass $sessionClass -ClientOnly -Property @{Enabled=$true;StateChange=[uint32]8;UserId=$userId;StartBoundary=$startBoundary}
$network = New-CimInstance -CimClass $eventClass -ClientOnly -Property @{Enabled=$true;StartBoundary=$startBoundary;Subscription="<QueryList><Query Id='0' Path='Microsoft-Windows-NetworkProfile/Operational'><Select Path='Microsoft-Windows-NetworkProfile/Operational'>*[System[EventID=10000]]</Select></Query></QueryList>"}
$wake = New-CimInstance -CimClass $eventClass -ClientOnly -Property @{Enabled=$true;StartBoundary=$startBoundary;Subscription="<QueryList><Query Id='0' Path='System'><Select Path='System'>*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]</Select></Query></QueryList>"}
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 4)
$definition = New-ScheduledTask -Action $action -Trigger @($trigger,$logon,$unlock,$network,$wake) -Principal $principal -Settings $settings -Description 'PRED daily 08:00 Europe/Bucharest, or catch up at logon/unlock/network/wake; once-per-day review then both markets, dashboard and link in the user chat.'
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    $existingArgs = ($existing.Actions | Select-Object -First 1).Arguments
    if (-not $existingArgs.Contains($runner)) { throw 'A different task already uses this name; refusing to replace it.' }
}
if ($PreviewOnly) {
    [pscustomobject]@{task=$taskName;user=$userId;execute=$powerShellExe;arguments=$arguments;
        first_run=$trigger.StartBoundary;timezone='Europe/Bucharest';overwrites_other_tasks=$false;requires_logged_in_user=$true;triggers=5;not_before_date=$firstDate;chat_thread_id=$ChatThreadId} | ConvertTo-Json
    exit 0
}
$utf8 = New-Object Text.UTF8Encoding($false)
$config = @{task_name=$taskName;codex_exe=$codexExe;node_exe=$nodeExe;timezone='Europe/Bucharest';not_before_date=$firstDate;chat_thread_id=$ChatThreadId} | ConvertTo-Json
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'daily_research_config.json'),$config,$utf8)
$preflightJson = & $powerShellExe -NoProfile -NonInteractive -File $runner -Preflight
$preflightExitCode = $LASTEXITCODE
if ($preflightExitCode -ne 0) { throw "Research runner preflight failed with exit code $preflightExitCode. Task was not registered." }
$preflight = $preflightJson | ConvertFrom-Json
if ($preflight.status -ne 'READY' -or $preflight.project -ne $projectDir -or $preflight.time_zone -ne 'Europe/Bucharest' -or $preflight.launches_agent -ne $false) {
    throw 'Research runner preflight returned invalid readiness data. Task was not registered.'
}
$preflightJson | Write-Output
Register-ScheduledTask -TaskName $taskName -InputObject $definition -Force | Out-Null
$task = Get-ScheduledTask -TaskName $taskName
$info = Get-ScheduledTaskInfo -TaskName $taskName
if ($task.State -eq 'Disabled' -or $info.NextRunTime -eq [DateTime]::MinValue) { throw 'Task registration did not produce an enabled next run.' }
$autoDir = Join-Path $projectDir 'Analysis/Automation'
[IO.Directory]::CreateDirectory($autoDir) | Out-Null
$status = [ordered]@{task_name=$taskName;schedule='08:00 daily';timezone='Europe/Bucharest';
    status='scheduled';registered_at_utc=[DateTime]::UtcNow.ToString('o');next_run_local=$info.NextRunTime.ToString('s');
    local_computer_required=$true;logged_in_user_required=$true;message='Enabled task verified; full research has not run yet.'}
$status['catch_up_triggers'] = @('logon','unlock','network_connected','resume_from_sleep')
$status['chat_thread_id'] = $ChatThreadId
[IO.File]::WriteAllText((Join-Path $autoDir 'daily_status.json'),($status | ConvertTo-Json),$utf8)
[IO.File]::WriteAllText((Join-Path $autoDir 'scheduled_task.xml'),(Export-ScheduledTask -TaskName $taskName),$utf8)
$status | ConvertTo-Json
