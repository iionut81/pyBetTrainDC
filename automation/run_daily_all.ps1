param(
    [string]$ProjectDir = "C:\Users\Ionut.Iordache\OneDrive - LucaNet AG\Desktop\Pred",
    [string]$PythonExe = "python",
    [string]$RatingsPkl = "data\historical\team_ratings.pkl",
    [int]$DaysAhead = 1,
    [double]$MinOdds = 1.10,
    [double]$MaxOdds = 1.25,
    [double]$CornersMinProb = 0.78,
    [double]$CornersMaxFairOdds = 1.30,
    [string]$Series = "1"
)

& "$ProjectDir\automation\run_daily_history_update.ps1" -ProjectDir $ProjectDir -PythonExe $PythonExe
if ($LASTEXITCODE -ne 0) {
    throw "Daily history update failed with exit code $LASTEXITCODE"
}

& "$ProjectDir\automation\run_daily_predictions_flashscore.ps1" -ProjectDir $ProjectDir -PythonExe $PythonExe -RatingsPkl $RatingsPkl -DaysAhead $DaysAhead -MinOdds $MinOdds -MaxOdds $MaxOdds -Series $Series
if ($LASTEXITCODE -ne 0) {
    throw "Daily DC predictions failed with exit code $LASTEXITCODE"
}

& "$ProjectDir\automation\run_daily_corners_u12_5.ps1" -ProjectDir $ProjectDir -PythonExe $PythonExe -DaysAhead $DaysAhead -MinProb $CornersMinProb -MinOdds $MinOdds -MaxOdds $MaxOdds -MaxFairOdds $CornersMaxFairOdds -Series $Series
if ($LASTEXITCODE -ne 0) {
    throw "Daily corners predictions failed with exit code $LASTEXITCODE"
}

Write-Host "Daily pipeline completed: history + DC + corners (Series $Series)."
