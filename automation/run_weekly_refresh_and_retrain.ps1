param(
    [string]$ProjectDir = "C:\Users\Ionut.Iordache\OneDrive - LucaNet AG\Desktop\Pred",
    [string]$PythonExe = "python",
    [string]$HistoryCsv = "data\historical\historical_matches_transfermarkt.csv",
    [string]$OutputPkl = "data\historical\team_ratings.pkl"
)

Set-Location $ProjectDir

& "$ProjectDir\automation\run_weekly_league_refresh.ps1" -ProjectDir $ProjectDir -PythonExe $PythonExe
if ($LASTEXITCODE -ne 0) {
    throw "Weekly refresh step failed with exit code $LASTEXITCODE"
}

& "$ProjectDir\automation\run_weekly_retrain_ratings.ps1" -ProjectDir $ProjectDir -PythonExe $PythonExe -HistoryCsv $HistoryCsv -OutputPkl $OutputPkl
if ($LASTEXITCODE -ne 0) {
    throw "Weekly retrain step failed with exit code $LASTEXITCODE"
}

Write-Host "Weekly pipeline completed: refresh + retrain."
