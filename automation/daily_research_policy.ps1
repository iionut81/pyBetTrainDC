# Pure decision helper: no processes, files, scheduling or network access.
function Test-PredChatNeedsRetry {
    param([string]$Status)
    # A manual recovery is reported in its already-active user turn. It has no
    # separate automatic delivery to retry; this is NOT a delivery receipt.
    return $Status -notin @('delivered','queued','not_required_manual')
}
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
