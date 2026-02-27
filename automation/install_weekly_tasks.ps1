param(
    [string]$ProjectDir = "C:\Users\Ionut.Iordache\OneDrive - LucaNet AG\Desktop\Pred",
    [string]$PythonExe = "python",
    [string]$HistoryCsv = "data\historical\historical_matches_transfermarkt.csv",
    [string]$OutputPkl = "data\historical\team_ratings.pkl"
)

$weeklyTask = "Pred_Weekly_Refresh_And_Retrain"

$weeklyAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ProjectDir\automation\run_weekly_refresh_and_retrain.ps1`" -ProjectDir `"$ProjectDir`" -PythonExe `"$PythonExe`" -HistoryCsv `"$HistoryCsv`" -OutputPkl `"$OutputPkl`""

$weeklyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 05:30

Register-ScheduledTask -TaskName $weeklyTask -Action $weeklyAction -Trigger $weeklyTrigger -Force | Out-Null

Write-Host "Installed tasks:"
Write-Host " - $weeklyTask @ Sunday 05:30"
