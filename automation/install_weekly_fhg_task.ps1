param(
    [string]$ProjectDir = "C:\Users\Ionut.Iordache\OneDrive - LucaNet AG\Desktop\Pred",
    [string]$PythonExe = "python"
)

$taskName = "Pred_Weekly_FHG_Refresh"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ProjectDir\automation\run_weekly_fhg_refresh.ps1`" -ProjectDir `"$ProjectDir`" -PythonExe `"$PythonExe`""

# Weekly Sunday after DC retrain.
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 07:00

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Force | Out-Null

Write-Host "Installed task:"
Write-Host " - $taskName @ Sunday 07:00"
