param(
    [string]$UserId = "$env:USERDOMAIN\$env:USERNAME"
)

$taskNames = @(
    "Pred_Daily_History_Update",
    "Pred_Daily_Flashscore_Predictions",
    "Pred_Weekly_League_Refresh",
    "Pred_Weekly_Retrain_Ratings"
)

$principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType S4U -RunLevel Highest

foreach ($taskName in $taskNames) {
    try {
        Set-ScheduledTask -TaskName $taskName -Principal $principal | Out-Null
        Write-Host "Updated: $taskName -> S4U ($UserId)"
    } catch {
        Write-Host "Failed: $taskName -> $($_.Exception.Message)"
    }
}

Write-Host "Done."
