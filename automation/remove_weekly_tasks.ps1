$refreshTask = "Pred_Weekly_League_Refresh"
$retrainTask = "Pred_Weekly_Retrain_Ratings"

Unregister-ScheduledTask -TaskName $refreshTask -Confirm:$false -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $retrainTask -Confirm:$false -ErrorAction SilentlyContinue

Write-Host "Removed task(s) if present:"
Write-Host " - $refreshTask"
Write-Host " - $retrainTask"
