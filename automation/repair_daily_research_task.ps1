# Repair only failure recovery on the EXISTING authorized PRED task.
# Preserve actions, triggers, principal, idle/battery/network settings and schedule.
[CmdletBinding()]
param([switch]$PreviewOnly)
$ErrorActionPreference = 'Stop'
$taskName = 'Pred_Daily_Research_0800'
$task = Get-ScheduledTask -TaskName $taskName
$runnerPath = Join-Path $PSScriptRoot 'run_daily_research.ps1'
if ($task.Actions.Count -ne 1 -or -not $task.Actions[0].Arguments.Contains($runnerPath)) { throw 'Unexpected PRED task action; refusing to modify it.' }
$before = [xml](Export-ScheduledTask -TaskName $taskName)
$settings = $task.Settings
$settings.RestartCount = 3
$settings.RestartInterval = 'PT5M'
if ($PreviewOnly) { [pscustomobject]@{task=$taskName;restart_count=3;restart_interval='PT5M';triggers_unchanged=$true;starts_research=$false} | ConvertTo-Json; return }
$evidenceRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'Analysis/Automation/Repairs'
[IO.Directory]::CreateDirectory($evidenceRoot) | Out-Null
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$before.Save((Join-Path $evidenceRoot ($stamp + '_task_before.xml')))
Set-ScheduledTask -TaskName $taskName -Settings $settings | Out-Null
$after = [xml](Export-ScheduledTask -TaskName $taskName)
$after.Save((Join-Path $evidenceRoot ($stamp + '_task_after.xml')))
foreach ($section in @('Triggers','Actions','Principals')) {
    if ($before.Task.$section.OuterXml -ne $after.Task.$section.OuterXml) { throw ('Unexpected task change: ' + $section) }
}
foreach ($setting in $before.Task.Settings.ChildNodes) {
    if ($setting.LocalName -eq 'RestartOnFailure') { continue }
    $afterSetting = $after.Task.Settings.ChildNodes | Where-Object LocalName -eq $setting.LocalName
    if ($setting.OuterXml -ne $afterSetting.OuterXml) { throw ('Unexpected setting change: ' + $setting.LocalName) }
}
$verified = Get-ScheduledTask -TaskName $taskName
if ($verified.Settings.RestartCount -ne 3 -or $verified.Settings.RestartInterval -ne 'PT5M') { throw 'Retry settings were not persisted.' }
[pscustomobject]@{task=$taskName;status='repaired';restart_count=3;restart_interval='PT5M';triggers_preserved=$true;actions_preserved=$true;principal_preserved=$true;starts_research=$false} | ConvertTo-Json
