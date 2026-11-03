# Pure decision helper: no processes, files, scheduling or network access.
function Get-PredDailyDecision {
    param([datetime]$NowRo, [string]$NotBeforeDate, [bool]$AlreadyFinished, [bool]$ChatPending)
    $date = $NowRo.ToString('yyyy-MM-dd')
    if ($NotBeforeDate -and [string]::CompareOrdinal($date,$NotBeforeDate) -lt 0) { return 'not_started' }
    if ($NowRo.TimeOfDay -lt [TimeSpan]::FromHours(8)) { return 'before_0800' }
    if ($AlreadyFinished) {
        if ($ChatPending) { return 'deliver_only' }
        return 'already_finished'
    }
    return 'research'
}
