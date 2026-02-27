param(
    [string]$ProjectDir = "C:\Users\Ionut.Iordache\OneDrive - LucaNet AG\Desktop\Pred",
    [string]$PythonExe = "python",
    [string]$HistoryCsv = "data\historical\historical_matches_transfermarkt.csv",
    [string]$OutputPkl = "data\historical\team_ratings.pkl"
)

$refreshTask = "Pred_Weekly_League_Refresh"
$retrainTask = "Pred_Weekly_Retrain_Ratings"

$refreshAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ProjectDir\automation\run_weekly_league_refresh.ps1`" -ProjectDir `"$ProjectDir`" -PythonExe `"$PythonExe`""

$retrainAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ProjectDir\automation\run_weekly_retrain_ratings.ps1`" -ProjectDir `"$ProjectDir`" -PythonExe `"$PythonExe`" -HistoryCsv `"$HistoryCsv`" -OutputPkl `"$OutputPkl`""

# Weekly Sunday schedule: refresh first, retrain after.
$refreshTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 05:30
$retrainTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 06:15

Register-ScheduledTask -TaskName $refreshTask -Action $refreshAction -Trigger $refreshTrigger -Force | Out-Null
Register-ScheduledTask -TaskName $retrainTask -Action $retrainAction -Trigger $retrainTrigger -Force | Out-Null

Write-Host "Installed tasks:"
Write-Host " - $refreshTask @ Sunday 05:30"
Write-Host " - $retrainTask @ Sunday 06:15"
