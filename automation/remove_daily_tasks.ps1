$historyTask = "Pred_Daily_History_Update"
$predictTask = "Pred_Daily_Flashscore_Predictions"

schtasks /Delete /F /TN $historyTask | Out-Null
schtasks /Delete /F /TN $predictTask | Out-Null

Write-Host "Removed task(s) if present:"
Write-Host " - $historyTask"
Write-Host " - $predictTask"
