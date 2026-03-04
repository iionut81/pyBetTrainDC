$historyTask = "Pred_Daily_History_Update"
$predictTask = "Pred_Daily_Flashscore_Predictions"
$cornersTask = "Pred_Daily_Corners_U12_5"

schtasks /Delete /F /TN $historyTask | Out-Null
schtasks /Delete /F /TN $predictTask | Out-Null
schtasks /Delete /F /TN $cornersTask | Out-Null

Write-Host "Removed task(s) if present:"
Write-Host " - $historyTask"
Write-Host " - $predictTask"
Write-Host " - $cornersTask"
