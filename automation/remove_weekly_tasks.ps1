$refreshTask = "Pred_Weekly_League_Refresh"
$retrainTask = "Pred_Weekly_Retrain_Ratings"
$weeklyTask = "Pred_Weekly_Refresh_And_Retrain"

Unregister-ScheduledTask -TaskName $refreshTask -Confirm:$false -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $retrainTask -Confirm:$false -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $weeklyTask -Confirm:$false -ErrorAction SilentlyContinue

Write-Host "Removed task(s) if present:"
Write-Host " - $refreshTask"
Write-Host " - $retrainTask"
Write-Host " - $weeklyTask"
