param(
    [string]$ProjectDir = "C:\Users\Ionut.Iordache\OneDrive - LucaNet AG\Desktop\Pred",
    [string]$PythonExe = "python",
    [string]$RatingsPkl = "data\historical\team_ratings.pkl"
)

$historyTask = "Pred_Daily_History_Update"
$predictTask = "Pred_Daily_Flashscore_Predictions"

$historyAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ProjectDir\automation\run_daily_history_update.ps1`" -ProjectDir `"$ProjectDir`" -PythonExe `"$PythonExe`""

$predictAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ProjectDir\automation\run_daily_predictions_flashscore.ps1`" -ProjectDir `"$ProjectDir`" -PythonExe `"$PythonExe`" -RatingsPkl `"$RatingsPkl`" -DaysAhead 1"

$historyTrigger = New-ScheduledTaskTrigger -Daily -At 06:30
$predictTrigger = New-ScheduledTaskTrigger -Daily -At 07:00

Register-ScheduledTask -TaskName $historyTask -Action $historyAction -Trigger $historyTrigger -Force | Out-Null
Register-ScheduledTask -TaskName $predictTask -Action $predictAction -Trigger $predictTrigger -Force | Out-Null

Write-Host "Installed tasks:"
Write-Host " - $historyTask @ 06:30"
Write-Host " - $predictTask @ 07:00"
