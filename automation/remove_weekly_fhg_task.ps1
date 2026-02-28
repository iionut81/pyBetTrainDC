$taskName = "Pred_Weekly_FHG_Refresh"

Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

Write-Host "Removed task if present:"
Write-Host " - $taskName"
