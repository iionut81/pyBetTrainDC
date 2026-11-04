#requires -Version 5.1
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../automation/daily_research_policy.ps1')
$cases = @(
    @{at='2026-11-03T12:00:00';done=$false;pending=$false;expected='not_started'},
    @{at='2026-11-04T07:59:59';done=$false;pending=$false;expected='before_0800'},
    @{at='2026-11-04T08:00:00';done=$false;pending=$false;expected='research'},
    @{at='2026-11-04T11:30:00';done=$false;pending=$false;expected='research'},
    @{at='2026-11-04T11:30:00';done=$true;pending=$false;expected='already_finished'},
    @{at='2026-11-04T11:30:00';done=$true;pending=$true;expected='deliver_only'}
)
foreach ($case in $cases) {
    $actual = Get-PredDailyDecision -NowRo ([datetime]$case.at) -NotBeforeDate '2026-11-04' -AlreadyFinished $case.done -ChatPending $case.pending
    if ($actual -ne $case.expected) { throw "Decision mismatch at $($case.at): $actual, expected $($case.expected)" }
}
$zone = [TimeZoneInfo]::FindSystemTimeZoneById('GTB Standard Time')
foreach ($status in @('delivered','queued','pending','failed')) {
    $expectedRetry = $status -in @('pending','failed')
    if ((Test-PredChatNeedsRetry -Status $status) -ne $expectedRetry) { throw "Invalid delivery retry decision for $status" }
}
foreach ($utc in @('2026-11-04T06:00:00Z','2027-07-01T05:00:00Z')) {
    $ro = [TimeZoneInfo]::ConvertTimeFromUtc(([datetime]::Parse($utc)).ToUniversalTime(),$zone)
    if ($ro.Hour -ne 8) { throw "Romanian DST/time-zone conversion failed: $utc" }
}
[pscustomobject]@{status='PASS';cases=$cases.Count;dst_cases=2;agent_launched=$false;task_modified=$false} | ConvertTo-Json
