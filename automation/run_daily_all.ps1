param(
    [string]$ProjectDir = "C:\Users\Ionut.Iordache\OneDrive - LucaNet AG\Desktop\Pred",
    [string]$PythonExe = "python",
    [string]$RatingsPkl = "data\historical\team_ratings.pkl",
    [int]$DaysAhead = 1,
    [double]$MinOdds = 1.10,
    [double]$MaxOdds = 1.25,
    [string]$Series = "1"
)

& "$ProjectDir\automation\run_daily_history_update.ps1" -ProjectDir $ProjectDir -PythonExe $PythonExe
& "$ProjectDir\automation\run_daily_predictions_flashscore.ps1" -ProjectDir $ProjectDir -PythonExe $PythonExe -RatingsPkl $RatingsPkl -DaysAhead $DaysAhead -MinOdds $MinOdds -MaxOdds $MaxOdds -Series $Series

Write-Host "Daily pipeline completed (Series $Series)."
